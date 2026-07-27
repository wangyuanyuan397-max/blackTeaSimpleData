"""Isolated MixNet-S explicit cross-scale interaction backbone.

This module is intentionally kept under ``temp/``.  Import it from the local
runner to register ``mixnet_s_explicit_scale_interaction`` only for this
experiment family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import BACKBONES


ALL_BLOCKS = (
    "S0B0",
    "S1B0", "S1B1",
    "S2B0", "S2B1", "S2B2", "S2B3",
    "S3B0", "S3B1", "S3B2",
    "S4B0", "S4B1", "S4B2",
    "S5B0", "S5B1", "S5B2",
)

STAGE_BLOCKS = {
    0: ("S0B0",),
    1: ("S1B0", "S1B1"),
    2: ("S2B0", "S2B1", "S2B2", "S2B3"),
    3: ("S3B0", "S3B1", "S3B2"),
    4: ("S4B0", "S4B1", "S4B2"),
    5: ("S5B0", "S5B1", "S5B2"),
}

PLACEMENTS = {
    "NONE": (),
    "ALL": ALL_BLOCKS,
    "ONLY_S0": STAGE_BLOCKS[0],
    "ONLY_S1": STAGE_BLOCKS[1],
    "ONLY_S2": STAGE_BLOCKS[2],
    "ONLY_S3": STAGE_BLOCKS[3],
    "ONLY_S4": STAGE_BLOCKS[4],
    "ONLY_S5": STAGE_BLOCKS[5],
    "EARLY_S01": STAGE_BLOCKS[0] + STAGE_BLOCKS[1],
    "MIDDLE_S23": STAGE_BLOCKS[2] + STAGE_BLOCKS[3],
    "LAST2_S45": STAGE_BLOCKS[4] + STAGE_BLOCKS[5],
    "MIDLATE_S2345": STAGE_BLOCKS[2] + STAGE_BLOCKS[3] + STAGE_BLOCKS[4] + STAGE_BLOCKS[5],
    "LATE_S345": STAGE_BLOCKS[3] + STAGE_BLOCKS[4] + STAGE_BLOCKS[5],
    "FIRST_BLOCK": ("S0B0", "S1B0", "S2B0", "S3B0", "S4B0", "S5B0"),
    "REPEAT_ONLY": ("S1B1", "S2B1", "S2B2", "S2B3", "S3B1", "S3B2", "S4B1", "S4B2", "S5B1", "S5B2"),
    "STRIDE2": ("S1B0", "S2B0", "S3B0", "S5B0"),
    "STRIDE1": tuple(block for block in ALL_BLOCKS if block not in {"S1B0", "S2B0", "S3B0", "S5B0"}),
}

INTERACTION_TYPES = {
    "baseline",
    "scale_attention",
    "small_to_large_guidance",
    "large_to_small_guidance",
    "weighted_sum",
    "cross_residual_bidir",
    "full_concat_interaction",
    "full_weighted_interaction",
}


@dataclass(frozen=True)
class InteractionConfig:
    interaction_type: str = "baseline"
    placement: str = "ALL"
    stage_mask: tuple[bool, ...] | None = None
    block_indices: tuple[int | str, ...] | None = None
    edge_mode: str = "adjacent"
    residual_strength: float = 1.0
    attention_hidden_dim: int = 8


def parse_block_name(block_name: str) -> tuple[int, int]:
    value = str(block_name).upper()
    if not value.startswith("S") or "B" not in value:
        raise ValueError(f"Invalid MixNet-S block name: {block_name}")
    stage_text, block_text = value[1:].split("B", 1)
    return int(stage_text), int(block_text)


def normalize_stage_mask(stage_mask: Sequence[Any] | None) -> tuple[bool, ...] | None:
    if stage_mask is None:
        return None
    mask = tuple(bool(value) for value in stage_mask)
    if len(mask) != len(STAGE_BLOCKS):
        raise ValueError(f"stage_mask must have {len(STAGE_BLOCKS)} entries, got {len(mask)}")
    return mask


def normalize_block_indices(block_indices: Sequence[int | str] | None) -> tuple[int | str, ...] | None:
    if block_indices is None:
        return None
    return tuple(block_indices)


def selected_blocks_from_config(config: InteractionConfig) -> tuple[str, ...]:
    if config.block_indices is not None:
        selected = []
        for item in config.block_indices:
            if isinstance(item, int):
                try:
                    selected.append(ALL_BLOCKS[item])
                except IndexError as exc:
                    raise ValueError(f"block index out of range: {item}") from exc
            else:
                block_name = str(item).upper()
                if block_name not in ALL_BLOCKS:
                    raise ValueError(f"Unknown MixNet-S block: {item}")
                selected.append(block_name)
        return tuple(dict.fromkeys(selected))

    if config.stage_mask is not None:
        selected = []
        for stage_index, enabled in enumerate(config.stage_mask):
            if enabled:
                selected.extend(STAGE_BLOCKS[stage_index])
        return tuple(selected)

    placement = str(config.placement).upper()
    if placement not in PLACEMENTS:
        raise ValueError(f"Unknown placement: {config.placement}. Available: {sorted(PLACEMENTS)}")
    return tuple(PLACEMENTS[placement])


def first_conv(module: nn.Module) -> nn.Conv2d:
    if isinstance(module, nn.Conv2d):
        return module
    if hasattr(module, "values"):
        values = list(module.values())
        if values and isinstance(values[0], nn.Conv2d):
            return values[0]
    branches = getattr(module, "branches", None)
    if branches is not None and len(branches) > 0 and isinstance(branches[0], nn.Conv2d):
        return branches[0]
    raise TypeError(f"Unsupported depthwise module: {module.__class__.__name__}")


def extract_depthwise_branches(module: nn.Module) -> list[nn.Conv2d]:
    if isinstance(module, nn.Conv2d):
        return [module]
    if hasattr(module, "values"):
        branches = list(module.values())
    else:
        branches = list(getattr(module, "branches", []))
    if not branches:
        raise TypeError(f"Unsupported depthwise module: {module.__class__.__name__}")
    if not all(isinstance(branch, nn.Conv2d) for branch in branches):
        branch_types = [branch.__class__.__name__ for branch in branches]
        raise TypeError(f"Depthwise branches must be Conv2d modules, got {branch_types}")
    return branches


def validate_depthwise_branch(branch: nn.Conv2d, block_name: str) -> None:
    if branch.groups != branch.in_channels:
        raise ValueError(f"{block_name} branch is not depthwise: groups={branch.groups}, in={branch.in_channels}")
    if branch.in_channels != branch.out_channels:
        raise ValueError(
            f"{block_name} branch must preserve channels for split-wise interaction, "
            f"got in={branch.in_channels}, out={branch.out_channels}"
        )


def build_edges(num_scales: int, direction: str, edge_mode: str) -> tuple[tuple[int, int], ...]:
    edge_mode = str(edge_mode).lower()
    if edge_mode not in {"adjacent", "all"}:
        raise ValueError("edge_mode must be adjacent or all.")

    edges = []
    for src in range(num_scales):
        for dst in range(num_scales):
            if src == dst:
                continue
            if edge_mode == "adjacent" and abs(src - dst) != 1:
                continue
            if direction == "small_to_large" and src < dst:
                edges.append((src, dst))
            elif direction == "large_to_small" and src > dst:
                edges.append((src, dst))
            elif direction == "bidirectional":
                edges.append((src, dst))
    return tuple(edges)


class BranchScaleAttention(nn.Module):
    """Per-sample scalar attention over scale branches.

    The final projection is zero-initialized and weights are returned as
    ``num_scales * softmax(logits)``, so each branch starts with scale 1.0.
    """

    def __init__(self, num_scales: int, hidden_dim: int) -> None:
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.num_scales = int(num_scales)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_scales, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, self.num_scales),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, branch_outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        descriptors = [
            output.mean(dim=(1, 2, 3))
            for output in branch_outputs
        ]
        logits = self.mlp(torch.stack(descriptors, dim=1))
        return self.num_scales * torch.softmax(logits, dim=1)


class SliceToFullProjection(nn.Module):
    """Project one scale branch into the full post-MixConv channel space."""

    def __init__(self, in_channels: int, out_channels: int, start: int, end: int) -> None:
        super().__init__()
        self.start = int(start)
        self.end = int(end)
        self.proj = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=False)
        self.reset_identity_slice()

    def reset_identity_slice(self) -> None:
        with torch.no_grad():
            self.proj.weight.zero_()
            width = min(self.end - self.start, self.proj.in_channels)
            for channel in range(width):
                self.proj.weight[self.start + channel, channel, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ExplicitScaleInteractionDepthwiseConv2d(nn.Module):
    """Depthwise MixConv wrapper with explicit pre-concat scale interaction."""

    def __init__(
        self,
        old_module: nn.Module,
        interaction_type: str,
        block_name: str,
        edge_mode: str = "adjacent",
        residual_strength: float = 1.0,
        attention_hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        interaction_type = str(interaction_type).lower()
        if interaction_type not in INTERACTION_TYPES - {"baseline"}:
            raise ValueError(f"Unsupported interaction_type: {interaction_type}")

        branches = extract_depthwise_branches(old_module)
        if len(branches) < 2:
            raise ValueError("Explicit scale interaction requires at least two MixConv branches.")
        for index, branch in enumerate(branches):
            validate_depthwise_branch(branch, f"{block_name}/branch_{index}")

        self.branches = nn.ModuleList(branches)
        self.interaction_type = interaction_type
        self.edge_mode = str(edge_mode).lower()
        self.residual_strength = float(residual_strength)
        self.in_splits = [int(branch.in_channels) for branch in self.branches]
        self.out_splits = [int(branch.out_channels) for branch in self.branches]
        self.splits = list(self.in_splits)
        self.in_channels = sum(self.in_splits)
        self.out_channels = sum(self.out_splits)
        self.kernel_sizes = tuple(int(branch.kernel_size[0]) for branch in self.branches)
        self.num_scales = len(self.branches)

        self.scale_attention = None
        if interaction_type in {"scale_attention", "weighted_sum", "full_concat_interaction", "full_weighted_interaction"}:
            self.scale_attention = BranchScaleAttention(self.num_scales, int(attention_hidden_dim))

        direction = None
        if interaction_type == "small_to_large_guidance":
            direction = "small_to_large"
        elif interaction_type == "large_to_small_guidance":
            direction = "large_to_small"
        elif interaction_type in {"cross_residual_bidir", "full_concat_interaction", "full_weighted_interaction"}:
            direction = "bidirectional"

        self.residual_edges: tuple[tuple[int, int], ...] = ()
        self.residual_projections = nn.ModuleDict()
        if direction is not None:
            self.residual_edges = build_edges(self.num_scales, direction=direction, edge_mode=self.edge_mode)
            for src, dst in self.residual_edges:
                key = f"{src}_to_{dst}"
                projection = nn.Conv2d(
                    self.out_splits[src],
                    self.out_splits[dst],
                    kernel_size=1,
                    bias=False,
                )
                nn.init.zeros_(projection.weight)
                self.residual_projections[key] = projection

        self.full_projections = nn.ModuleList()
        if interaction_type in {"weighted_sum", "full_weighted_interaction"}:
            start = 0
            for channels in self.out_splits:
                end = start + channels
                self.full_projections.append(
                    SliceToFullProjection(channels, self.out_channels, start=start, end=end)
                )
                start = end

    def _apply_attention(self, outputs: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.scale_attention is None:
            return outputs
        weights = self.scale_attention(outputs)
        return [
            output * weights[:, index].view(-1, 1, 1, 1)
            for index, output in enumerate(outputs)
        ]

    def _apply_residual_edges(self, outputs: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.residual_edges:
            return outputs
        updates = [torch.zeros_like(output) for output in outputs]
        for src, dst in self.residual_edges:
            key = f"{src}_to_{dst}"
            updates[dst] = updates[dst] + self.residual_projections[key](outputs[src])
        return [
            output + self.residual_strength * update
            for output, update in zip(outputs, updates)
        ]

    def _weighted_sum(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        weights = self.scale_attention(outputs) if self.scale_attention is not None else None
        full_outputs = []
        for index, (projection, output) in enumerate(zip(self.full_projections, outputs)):
            full = projection(output)
            if weights is not None:
                full = full * weights[:, index].view(-1, 1, 1, 1)
            full_outputs.append(full)
        return torch.stack(full_outputs, dim=0).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_split = torch.split(x, self.in_splits, dim=1)
        outputs = [
            branch(x_part)
            for branch, x_part in zip(self.branches, x_split)
        ]
        if self.interaction_type in {"scale_attention", "full_concat_interaction"}:
            outputs = self._apply_attention(outputs)
        outputs = self._apply_residual_edges(outputs)
        if self.interaction_type in {"weighted_sum", "full_weighted_interaction"}:
            return self._weighted_sum(outputs)
        return torch.cat(outputs, dim=1)

    def interaction_summary(self) -> dict[str, Any]:
        return {
            "interaction_type": self.interaction_type,
            "edge_mode": self.edge_mode,
            "residual_strength": self.residual_strength,
            "kernel_sizes": list(self.kernel_sizes),
            "in_splits": list(self.in_splits),
            "out_splits": list(self.out_splits),
            "num_scales": self.num_scales,
            "residual_edges": [list(edge) for edge in self.residual_edges],
            "uses_scale_attention": self.scale_attention is not None,
            "uses_weighted_sum": bool(self.full_projections),
        }


@BACKBONES.register("mixnet_s_explicit_scale_interaction")
class MixNetSExplicitScaleInteractionBackbone(nn.Module):
    """MixNet-S with explicit interactions between MixConv scale branches."""

    def __init__(
        self,
        pretrained: bool = True,
        input_size: int = 408,
        model_name: str = "mixnet_s",
        interaction_type: str = "baseline",
        placement: str = "ALL",
        stage_mask: Sequence[Any] | None = None,
        block_indices: Sequence[int | str] | None = None,
        edge_mode: str = "adjacent",
        residual_strength: float = 1.0,
        attention_hidden_dim: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        interaction_type = str(interaction_type).lower()
        if interaction_type not in INTERACTION_TYPES:
            raise ValueError(f"interaction_type must be one of {sorted(INTERACTION_TYPES)}, got {interaction_type}")
        self.interaction_config = InteractionConfig(
            interaction_type=interaction_type,
            placement=str(placement).upper(),
            stage_mask=normalize_stage_mask(stage_mask),
            block_indices=normalize_block_indices(block_indices),
            edge_mode=str(edge_mode).lower(),
            residual_strength=float(residual_strength),
            attention_hidden_dim=int(attention_hidden_dim),
        )
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        self.model_name = str(model_name)
        self.selected_blocks = selected_blocks_from_config(self.interaction_config)
        self.applied_blocks: dict[str, dict[str, Any]] = {}
        self.skipped_blocks: dict[str, dict[str, Any]] = {}
        self._apply_interaction_plan()
        self.out_features = self._infer_out_features(int(input_size))

    def _apply_interaction_plan(self) -> None:
        if self.interaction_config.interaction_type == "baseline":
            return
        for block_name in self.selected_blocks:
            stage_index, block_index = parse_block_name(block_name)
            try:
                block = self.model.blocks[stage_index][block_index]
            except IndexError as exc:
                raise ValueError(f"MixNet-S block does not exist: {block_name}") from exc

            old_conv = block.conv_dw
            branches = extract_depthwise_branches(old_conv)
            if len(branches) < 2:
                self.skipped_blocks[block_name] = {
                    "reason": "single_scale_depthwise",
                    "branch_count": len(branches),
                }
                continue

            new_conv = ExplicitScaleInteractionDepthwiseConv2d(
                old_conv,
                interaction_type=self.interaction_config.interaction_type,
                block_name=block_name,
                edge_mode=self.interaction_config.edge_mode,
                residual_strength=self.interaction_config.residual_strength,
                attention_hidden_dim=self.interaction_config.attention_hidden_dim,
            )
            block.conv_dw = new_conv
            self.applied_blocks[block_name] = new_conv.interaction_summary()

    def interaction_summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "interaction_type": self.interaction_config.interaction_type,
            "placement": self.interaction_config.placement,
            "stage_mask": list(self.interaction_config.stage_mask) if self.interaction_config.stage_mask is not None else None,
            "block_indices": list(self.interaction_config.block_indices) if self.interaction_config.block_indices is not None else None,
            "edge_mode": self.interaction_config.edge_mode,
            "residual_strength": self.interaction_config.residual_strength,
            "attention_hidden_dim": self.interaction_config.attention_hidden_dim,
            "selected_blocks": list(self.selected_blocks),
            "modified_block_count": len(self.applied_blocks),
            "skipped_block_count": len(self.skipped_blocks),
            "applied_blocks": self.applied_blocks,
            "skipped_blocks": self.skipped_blocks,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._extract_feature_vector(x)

    def _extract_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features") and hasattr(self.model, "forward_head"):
            features = self.model.forward_features(x)
            try:
                features = self.model.forward_head(features, pre_logits=True)
            except TypeError:
                features = self.model.forward_head(features)
        else:
            features = self.model(x)
        return self._to_feature_vector(features)

    def _to_feature_vector(self, features: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.ndim == 2:
            return features
        if features.ndim == 4:
            num_features = getattr(self.model, "num_features", None)
            if num_features is not None and features.shape[-1] == num_features:
                return features.mean(dim=(1, 2))
            return torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
        if features.ndim == 3:
            return features.mean(dim=1)
        return torch.flatten(features, 1)

    def _infer_out_features(self, input_size: int) -> int:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, input_size, input_size)
                features = self._extract_feature_vector(dummy)
        finally:
            self.model.train(was_training)
        if features.ndim != 2 or features.shape[1] <= 0:
            raise ValueError(
                "mixnet_s_explicit_scale_interaction must return a non-empty 2D feature "
                f"tensor, got {tuple(features.shape)}"
            )
        return int(features.shape[1])
