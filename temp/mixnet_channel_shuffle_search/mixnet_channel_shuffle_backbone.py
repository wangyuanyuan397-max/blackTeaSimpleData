"""Isolated MixNet-S channel-shuffle fusion search backbone.

This module is intentionally kept outside ``src/``.  Import it from the runner
in this directory to register ``mixnet_s_channel_shuffle_search`` only for these
search experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

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

REPLACEMENT_TYPES = {"baseline", "group", "shuffle_group", "group_shuffle", "shuffle_dense"}
ADDITIVE_TYPES = {"extra_group", "extra_shuffle_group", "partial_mix"}
FUSION_TYPES = REPLACEMENT_TYPES | ADDITIVE_TYPES


@dataclass(frozen=True)
class FusionConfig:
    fusion_type: str = "baseline"
    target_groups: int = 1
    partial_ratio: float = 0.5
    placement: str = "ALL"
    stage_mask: tuple[bool, ...] | None = None
    block_indices: tuple[int | str, ...] | None = None
    shuffle_mode: str = "scale"
    insertion_mode: str = "auto"
    random_permutation_seed: int = 2026


def parse_block_name(block_name: str) -> tuple[int, int]:
    value = str(block_name).upper()
    if not value.startswith("S") or "B" not in value:
        raise ValueError(f"Invalid MixNet block name: {block_name}")
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


def selected_blocks_from_config(config: FusionConfig) -> tuple[str, ...]:
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


def infer_num_scales(depthwise_module: nn.Module) -> int:
    splits = getattr(depthwise_module, "splits", None)
    if splits is not None:
        try:
            return max(1, len(tuple(splits)))
        except TypeError:
            pass
    branches = getattr(depthwise_module, "branches", None)
    if branches is not None:
        try:
            return max(1, len(branches))
        except TypeError:
            pass
    if hasattr(depthwise_module, "values"):
        try:
            return max(1, len(list(depthwise_module.values())))
        except Exception:
            pass
    return 1


def resolve_groups(
    mid_channels: int,
    out_channels: int,
    num_scales: int,
    target_groups: int,
    require_scale_balance: bool = True,
) -> int:
    target_groups = max(1, int(target_groups))
    candidates = [
        group
        for group in (target_groups, 8, 4, 2, 1)
        if group <= target_groups
    ]
    candidates = list(dict.fromkeys(candidates))
    for group in candidates:
        legal = mid_channels % group == 0 and out_channels % group == 0
        if require_scale_balance:
            legal = legal and mid_channels % (max(1, num_scales) * group) == 0
        if legal:
            return group
    return 1


def resolve_shuffle_groups(channels: int, requested_groups: int) -> int:
    requested_groups = max(1, int(requested_groups))
    if requested_groups <= 1:
        return 1
    candidates = [
        group
        for group in (requested_groups, 8, 4, 2, 1)
        if group <= requested_groups
    ]
    candidates = list(dict.fromkeys(candidates))
    for group in candidates:
        if channels % group == 0:
            return group
    return 1


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = int(groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.groups <= 1:
            return x
        batch, channels, height, width = x.shape
        if channels % self.groups != 0:
            raise ValueError(f"Cannot shuffle {channels} channels into {self.groups} groups.")
        x = x.view(batch, self.groups, channels // self.groups, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, channels, height, width)


class FixedChannelPermutation(nn.Module):
    def __init__(self, channels: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        self.register_buffer("permutation", torch.randperm(int(channels), generator=generator), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(1, self.permutation.to(device=x.device))


class ShuffleDenseProjection(nn.Module):
    def __init__(self, shuffle: nn.Module, dense_projection: nn.Module) -> None:
        super().__init__()
        self.shuffle = shuffle
        self.dense_projection = dense_projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_projection(self.shuffle(x))


class ShuffleGroupProjection(nn.Module):
    def __init__(self, shuffle: nn.Module, group_projection: nn.Module) -> None:
        super().__init__()
        self.shuffle = shuffle
        self.group_projection = group_projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_projection(self.shuffle(x))


class GroupShuffleProjection(nn.Module):
    def __init__(self, group_projection: nn.Module, shuffle: nn.Module) -> None:
        super().__init__()
        self.group_projection = group_projection
        self.shuffle = shuffle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.group_projection(x))


class ExtraGroupProjection(nn.Module):
    def __init__(
        self,
        pre_shuffle: nn.Module,
        group_mix: nn.Module,
        dense_projection: nn.Module,
        channels: int,
    ) -> None:
        super().__init__()
        self.pre_shuffle = pre_shuffle
        self.group_mix = group_mix
        self.norm = nn.BatchNorm2d(int(channels))
        self.act = nn.SiLU(inplace=True)
        self.dense_projection = dense_projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_shuffle(x)
        x = self.group_mix(x)
        x = self.act(self.norm(x))
        return self.dense_projection(x)


class PartialMixProjection(nn.Module):
    def __init__(
        self,
        mix_channels: int,
        partial_mix: nn.Module,
        dense_projection: nn.Module,
    ) -> None:
        super().__init__()
        self.mix_channels = int(mix_channels)
        self.partial_mix = partial_mix
        self.dense_projection = dense_projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.partial_mix(x[:, : self.mix_channels])
        if self.mix_channels >= x.shape[1]:
            fused = mixed
        else:
            fused = torch.cat((mixed, x[:, self.mix_channels :]), dim=1)
        return self.dense_projection(fused)


def make_conv1x1_like(
    reference: nn.Module,
    in_channels: int,
    out_channels: int,
    groups: int,
    bias: bool | None = None,
) -> nn.Conv2d:
    ref_conv = first_conv(reference)
    use_bias = ref_conv.bias is not None if bias is None else bool(bias)
    conv = nn.Conv2d(
        int(in_channels),
        int(out_channels),
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=int(groups),
        bias=use_bias,
    )
    return conv.to(device=ref_conv.weight.device, dtype=ref_conv.weight.dtype)


def copy_group_projection_from_dense(group_conv: nn.Conv2d, dense_conv: nn.Conv2d) -> None:
    groups = int(group_conv.groups)
    if dense_conv.weight.shape[-2:] != (1, 1):
        raise ValueError("Only 1x1 dense projection weights can initialize grouped projection.")
    with torch.no_grad():
        group_conv.weight.zero_()
        in_per_group = group_conv.in_channels // groups
        out_per_group = group_conv.out_channels // groups
        for group_index in range(groups):
            out_start = group_index * out_per_group
            out_end = out_start + out_per_group
            in_start = group_index * in_per_group
            in_end = in_start + in_per_group
            group_conv.weight[out_start:out_end].copy_(
                dense_conv.weight[out_start:out_end, in_start:in_end]
            )
        if group_conv.bias is not None:
            if dense_conv.bias is None:
                group_conv.bias.zero_()
            else:
                group_conv.bias.copy_(dense_conv.bias)


def init_group_projection_from_reference(group_conv: nn.Conv2d, reference: nn.Module) -> None:
    if isinstance(reference, nn.Conv2d):
        copy_group_projection_from_dense(group_conv, reference)
        return
    nn.init.kaiming_normal_(group_conv.weight, mode="fan_out", nonlinearity="relu")
    if group_conv.bias is not None:
        nn.init.zeros_(group_conv.bias)


def init_group_conv_identity(conv: nn.Conv2d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        in_per_group = conv.in_channels // conv.groups
        out_per_group = conv.out_channels // conv.groups
        for group_index in range(conv.groups):
            out_start = group_index * out_per_group
            for local_channel in range(min(in_per_group, out_per_group)):
                conv.weight[out_start + local_channel, local_channel, 0, 0] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


def init_dense_identity(conv: nn.Conv2d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        for channel in range(min(conv.in_channels, conv.out_channels)):
            conv.weight[channel, channel, 0, 0] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


def make_shuffle(
    mode: str,
    channels: int,
    num_scales: int,
    conv_groups: int,
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    mode = str(mode).lower()
    if mode in {"none", "off", "identity"}:
        return nn.Identity(), {"mode": "none", "groups": 1}
    if mode == "scale":
        groups = resolve_shuffle_groups(channels, num_scales)
        return ChannelShuffle(groups), {"mode": "scale", "groups": groups}
    if mode == "fusion":
        groups = resolve_shuffle_groups(channels, conv_groups)
        return ChannelShuffle(groups), {"mode": "fusion", "groups": groups}
    if mode == "random":
        return FixedChannelPermutation(channels, seed), {"mode": "random", "groups": None, "seed": int(seed)}
    if mode == "double_scale":
        groups = resolve_shuffle_groups(channels, num_scales)
        return nn.Sequential(ChannelShuffle(groups), ChannelShuffle(groups)), {"mode": "double_scale", "groups": groups}
    raise ValueError("shuffle_mode must be one of: none, scale, fusion, random, double_scale.")


def resolve_partial_channels(channels: int, ratio: float) -> int:
    ratio = min(1.0, max(0.0, float(ratio)))
    if ratio <= 0:
        raise ValueError("partial_ratio must be positive.")
    return max(1, min(int(channels), int(round(int(channels) * ratio))))


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
    raise TypeError(f"Unsupported projection module: {module.__class__.__name__}")


def iter_projection_convs(module: nn.Module) -> Iterable[nn.Conv2d]:
    if isinstance(module, nn.Conv2d):
        yield module
        return
    if hasattr(module, "values"):
        for child in module.values():
            if not isinstance(child, nn.Conv2d):
                raise TypeError(f"Projection child must be nn.Conv2d, got {child.__class__.__name__}")
            yield child
        return
    branches = getattr(module, "branches", None)
    if branches is not None:
        for child in branches:
            if not isinstance(child, nn.Conv2d):
                raise TypeError(f"Projection branch must be nn.Conv2d, got {child.__class__.__name__}")
            yield child
        return
    raise TypeError(f"Unsupported projection module: {module.__class__.__name__}")


def projection_in_channels(module: nn.Module) -> int:
    value = getattr(module, "in_channels", None)
    if value is not None:
        return int(value)
    return sum(int(conv.in_channels) for conv in iter_projection_convs(module))


def projection_out_channels(module: nn.Module) -> int:
    value = getattr(module, "out_channels", None)
    if value is not None:
        return int(value)
    return sum(int(conv.out_channels) for conv in iter_projection_convs(module))


def projection_attr_name(block: nn.Module, block_name: str) -> str:
    if hasattr(block, "conv_pwl"):
        return "conv_pwl"
    if hasattr(block, "conv_pw"):
        # timm's first MixNet-S block is DepthwiseSeparableConv.  There,
        # conv_pw is the pointwise projection after depthwise/MixConv.  In
        # InvertedResidual blocks conv_pw is expansion and conv_pwl is used
        # above instead.
        return "conv_pw"
    raise AttributeError(f"{block_name} has neither conv_pwl nor conv_pw projection.")


def validate_projection_module(module: nn.Module, block_name: str) -> nn.Module:
    for conv in iter_projection_convs(module):
        if conv.kernel_size != (1, 1):
            raise ValueError(f"{block_name} projection must use only 1x1 convs, got {conv.kernel_size}")
    return module


def build_projection(
    dense_projection: nn.Module,
    block_name: str,
    fusion_config: FusionConfig,
    num_scales: int,
    block_seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    fusion_type = fusion_config.fusion_type
    mid_channels = projection_in_channels(dense_projection)
    out_channels = projection_out_channels(dense_projection)
    target_groups = int(fusion_config.target_groups)
    metadata: dict[str, Any] = {
        "block": block_name,
        "fusion_type": fusion_type,
        "target_groups": target_groups,
        "num_scales": int(num_scales),
        "mid_channels": mid_channels,
        "out_channels": out_channels,
    }

    if fusion_type == "baseline":
        metadata.update({"actual_groups": 1, "insertion_mode": "replace"})
        return dense_projection, metadata

    if fusion_type == "shuffle_dense":
        shuffle, shuffle_meta = make_shuffle(
            fusion_config.shuffle_mode,
            channels=mid_channels,
            num_scales=num_scales,
            conv_groups=1,
            seed=block_seed,
        )
        metadata.update({"actual_groups": 1, "pre_shuffle": shuffle_meta, "insertion_mode": "replace"})
        return ShuffleDenseProjection(shuffle, dense_projection), metadata

    if fusion_type in {"group", "shuffle_group", "group_shuffle"}:
        groups = resolve_groups(mid_channels, out_channels, num_scales, target_groups)
        group_projection = make_conv1x1_like(dense_projection, mid_channels, out_channels, groups)
        init_group_projection_from_reference(group_projection, dense_projection)
        metadata.update({"actual_groups": groups, "insertion_mode": "replace"})

        if fusion_type == "group":
            return group_projection, metadata
        if fusion_type == "shuffle_group":
            shuffle, shuffle_meta = make_shuffle(
                fusion_config.shuffle_mode,
                channels=mid_channels,
                num_scales=num_scales,
                conv_groups=groups,
                seed=block_seed,
            )
            metadata["pre_shuffle"] = shuffle_meta
            return ShuffleGroupProjection(shuffle, group_projection), metadata

        shuffle, shuffle_meta = make_shuffle(
            fusion_config.shuffle_mode,
            channels=out_channels,
            num_scales=num_scales,
            conv_groups=groups,
            seed=block_seed + 1,
        )
        metadata["post_shuffle"] = shuffle_meta
        return GroupShuffleProjection(group_projection, shuffle), metadata

    if fusion_type in {"extra_group", "extra_shuffle_group"}:
        groups = resolve_groups(mid_channels, mid_channels, num_scales, target_groups)
        group_mix = make_conv1x1_like(dense_projection, mid_channels, mid_channels, groups, bias=False)
        init_group_conv_identity(group_mix)
        if fusion_type == "extra_shuffle_group":
            pre_shuffle, shuffle_meta = make_shuffle(
                fusion_config.shuffle_mode,
                channels=mid_channels,
                num_scales=num_scales,
                conv_groups=groups,
                seed=block_seed,
            )
        else:
            pre_shuffle, shuffle_meta = nn.Identity(), {"mode": "none", "groups": 1}
        metadata.update(
            {
                "actual_groups": groups,
                "pre_shuffle": shuffle_meta,
                "insertion_mode": "add",
            }
        )
        return ExtraGroupProjection(pre_shuffle, group_mix, dense_projection, mid_channels), metadata

    if fusion_type == "partial_mix":
        mix_channels = resolve_partial_channels(mid_channels, fusion_config.partial_ratio)
        partial_conv = make_conv1x1_like(dense_projection, mix_channels, mix_channels, groups=1, bias=False)
        init_dense_identity(partial_conv)
        partial_mix = nn.Sequential(
            partial_conv,
            nn.BatchNorm2d(mix_channels),
            nn.SiLU(inplace=True),
        )
        metadata.update(
            {
                "actual_groups": 1,
                "partial_ratio": float(fusion_config.partial_ratio),
                "partial_channels": mix_channels,
                "insertion_mode": "add",
            }
        )
        return PartialMixProjection(mix_channels, partial_mix, dense_projection), metadata

    raise ValueError(f"Unknown fusion_type: {fusion_type}")


@BACKBONES.register("mixnet_s_channel_shuffle_search")
class MixNetSChannelShuffleSearchBackbone(nn.Module):
    """MixNet-S with configurable channel fusion after MixConv projection input."""

    def __init__(
        self,
        pretrained: bool = True,
        input_size: int = 408,
        model_name: str = "mixnet_s",
        fusion_type: str = "baseline",
        target_groups: int = 1,
        partial_ratio: float = 0.5,
        placement: str = "ALL",
        stage_mask: Sequence[Any] | None = None,
        block_indices: Sequence[int | str] | None = None,
        shuffle_mode: str = "scale",
        insertion_mode: str = "auto",
        random_permutation_seed: int = 2026,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        fusion_type = str(fusion_type).lower()
        if fusion_type not in FUSION_TYPES:
            raise ValueError(f"fusion_type must be one of {sorted(FUSION_TYPES)}, got {fusion_type}")
        expected_mode = "add" if fusion_type in ADDITIVE_TYPES else "replace"
        insertion_mode = str(insertion_mode).lower()
        if insertion_mode not in {"auto", expected_mode}:
            raise ValueError(f"{fusion_type} expects insertion_mode={expected_mode!r}, got {insertion_mode!r}")

        self.fusion_config = FusionConfig(
            fusion_type=fusion_type,
            target_groups=int(target_groups),
            partial_ratio=float(partial_ratio),
            placement=str(placement).upper(),
            stage_mask=normalize_stage_mask(stage_mask),
            block_indices=normalize_block_indices(block_indices),
            shuffle_mode=str(shuffle_mode).lower(),
            insertion_mode=expected_mode,
            random_permutation_seed=int(random_permutation_seed),
        )
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        self.model_name = str(model_name)
        self.selected_blocks = selected_blocks_from_config(self.fusion_config)
        self.applied_blocks: dict[str, dict[str, Any]] = {}
        self._apply_fusion_plan()
        self.out_features = self._infer_out_features(int(input_size))

    def _apply_fusion_plan(self) -> None:
        if self.fusion_config.fusion_type == "baseline":
            return
        for selection_index, block_name in enumerate(self.selected_blocks):
            stage_index, block_index = parse_block_name(block_name)
            try:
                block = self.model.blocks[stage_index][block_index]
            except IndexError as exc:
                raise ValueError(f"MixNet-S block does not exist: {block_name}") from exc

            projection_attr = projection_attr_name(block, block_name)
            old_projection = validate_projection_module(getattr(block, projection_attr), block_name)
            num_scales = infer_num_scales(block.conv_dw)
            block_seed = self.fusion_config.random_permutation_seed + selection_index * 1009
            new_projection, metadata = build_projection(
                old_projection,
                block_name,
                self.fusion_config,
                num_scales=num_scales,
                block_seed=block_seed,
            )
            metadata["projection_attr"] = projection_attr
            setattr(block, projection_attr, new_projection)
            self.applied_blocks[block_name] = metadata

    def fusion_summary(self) -> dict[str, Any]:
        actual_groups_histogram: dict[str, int] = {}
        for record in self.applied_blocks.values():
            group_key = str(record.get("actual_groups", 1))
            actual_groups_histogram[group_key] = actual_groups_histogram.get(group_key, 0) + 1
        return {
            "model_name": self.model_name,
            "fusion_type": self.fusion_config.fusion_type,
            "target_groups": self.fusion_config.target_groups,
            "partial_ratio": self.fusion_config.partial_ratio,
            "placement": self.fusion_config.placement,
            "stage_mask": list(self.fusion_config.stage_mask) if self.fusion_config.stage_mask is not None else None,
            "block_indices": list(self.fusion_config.block_indices) if self.fusion_config.block_indices is not None else None,
            "shuffle_mode": self.fusion_config.shuffle_mode,
            "insertion_mode": self.fusion_config.insertion_mode,
            "selected_blocks": list(self.selected_blocks),
            "modified_block_count": len(self.applied_blocks),
            "actual_groups_histogram": actual_groups_histogram,
            "applied_blocks": self.applied_blocks,
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
                "mixnet_s_channel_shuffle_search must return a non-empty 2D feature "
                f"tensor, got {tuple(features.shape)}"
            )
        return int(features.shape[1])
