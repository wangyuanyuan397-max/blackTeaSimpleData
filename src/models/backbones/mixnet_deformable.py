"""MixNet-S backbone with optional structure search kernels and deformable attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...utils.registry import BACKBONES
from .deformable_attention import DeformableAttention2d
from .mixnet_search import (
    ALL_BLOCKS,
    SearchMixedDepthwiseConv2d,
    build_kernel_plan,
    copy_resized_depthwise_weights,
    normalize_kernel_sizes,
    parse_block_name,
    _first_conv,
)


@dataclass(frozen=True)
class StageInfo:
    stage_id: int
    timm_stage_index: int
    block_index: int
    channels: int
    height: int
    width: int


def normalize_stage_ids(stage_ids: Sequence[int] | None) -> tuple[int, ...]:
    if stage_ids is None:
        return ()
    normalized = tuple(sorted({int(stage_id) for stage_id in stage_ids}))
    invalid = [stage_id for stage_id in normalized if stage_id < 0 or stage_id > 4]
    if invalid:
        raise ValueError(f"deform_stage_ids must be in [0, 4], got {invalid}.")
    return normalized


@BACKBONES.register("mixnet_s_deformable")
class MixNetSDeformableBackbone(nn.Module):
    """MixNet-S with optional MixConv search plan and deformable attention."""

    def __init__(
        self,
        pretrained: bool = True,
        input_size: int = 408,
        deform_stage_ids: Sequence[int] | None = None,
        model_name: str = "mixnet_s",
        num_heads: int = 4,
        num_points: int = 4,
        max_offset: float = 2.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        layer_scale_init: float = 1e-3,
        deform_checkpoint: bool = False,
        placement: str = "ORIGINAL",
        kernel_sizes: Sequence[int] | None = None,
        kernel_plan: Mapping[str, Sequence[int]] | None = None,
        gate_type: str = "none",
        gate_reduction: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        if model_name != "mixnet_s":
            raise ValueError("This backbone fixes model_name to mixnet_s.")

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        self.input_size = int(input_size)
        self.placement = str(placement).upper()
        self.gate_type = str(gate_type).lower()
        self.kernel_plan = self._resolve_kernel_plan(kernel_plan, kernel_sizes)
        self.applied_blocks: dict[str, tuple[int, ...]] = {}
        self._apply_kernel_plan(gate_reduction=int(gate_reduction))

        self.deform_stage_ids = normalize_stage_ids(deform_stage_ids)
        self.deform_checkpoint = bool(deform_checkpoint)
        self.stage_infos = self._discover_stage_infos(self.input_size)
        self.deform_blocks = nn.ModuleDict()
        for stage_info in self.stage_infos:
            if stage_info.stage_id not in self.deform_stage_ids:
                continue
            self.deform_blocks[str(stage_info.stage_id)] = DeformableAttention2d(
                channels=stage_info.channels,
                num_heads=int(num_heads),
                num_points=int(num_points),
                max_offset=float(max_offset),
                attention_dropout=float(attention_dropout),
                projection_dropout=float(projection_dropout),
                layer_scale_init=float(layer_scale_init),
            )

        self._insert_after = {
            (info.timm_stage_index, info.block_index): str(info.stage_id)
            for info in self.stage_infos
            if info.stage_id in self.deform_stage_ids
        }
        self.out_features = self._infer_out_features(self.input_size)

    def _resolve_kernel_plan(
        self,
        kernel_plan: Mapping[str, Sequence[int]] | None,
        kernel_sizes: Sequence[int] | None,
    ) -> dict[str, tuple[int, ...]]:
        if kernel_plan:
            return {
                str(block_name).upper(): normalize_kernel_sizes(kernels)
                for block_name, kernels in kernel_plan.items()
            }
        if self.placement == "ORIGINAL":
            return {}
        return build_kernel_plan(self.placement, kernel_sizes or (3, 5, 7))

    def _apply_kernel_plan(self, gate_reduction: int) -> None:
        for block_name, kernels in self.kernel_plan.items():
            if block_name not in ALL_BLOCKS:
                raise ValueError(f"Unknown MixNet-S block in kernel_plan: {block_name}")
            stage_index, block_index = parse_block_name(block_name)
            try:
                block = self.model.blocks[stage_index][block_index]
            except IndexError as exc:
                raise ValueError(f"MixNet-S block does not exist: {block_name}") from exc

            old_conv = block.conv_dw
            first = _first_conv(old_conv)
            bias = first.bias is not None
            new_conv = SearchMixedDepthwiseConv2d(
                in_channels=int(
                    first.in_channels
                    if isinstance(old_conv, nn.Conv2d)
                    else old_conv.in_channels
                ),
                out_channels=int(
                    first.out_channels
                    if isinstance(old_conv, nn.Conv2d)
                    else old_conv.out_channels
                ),
                kernel_sizes=kernels,
                stride=first.stride,
                padding="",
                dilation=first.dilation,
                bias=bias,
                gate_type=self.gate_type,
                gate_reduction=gate_reduction,
                device=first.weight.device,
                dtype=first.weight.dtype,
            )
            copy_resized_depthwise_weights(old_conv, new_conv)
            block.conv_dw = new_conv
            self.applied_blocks[block_name] = tuple(kernels)

    def _discover_stage_infos(self, input_size: int) -> tuple[StageInfo, ...]:
        required = ("conv_stem", "bn1", "blocks", "conv_head", "bn2", "forward_head")
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise ValueError(f"Unexpected timm MixNet-S structure; missing {missing}.")

        was_training = self.model.training
        self.model.eval()
        records: list[tuple[int, int, int, int, int]] = []
        try:
            with torch.no_grad():
                x = torch.zeros(1, 3, int(input_size), int(input_size))
                x = self.model.conv_stem(x)
                x = self.model.bn1(x)
                for timm_stage_index, stage in enumerate(self.model.blocks):
                    for block_index, block in enumerate(stage):
                        x = block(x)
                        records.append(
                            (
                                timm_stage_index,
                                block_index,
                                int(x.shape[1]),
                                int(x.shape[2]),
                                int(x.shape[3]),
                            )
                        )
        finally:
            self.model.train(was_training)

        grouped: list[list[tuple[int, int, int, int, int]]] = []
        for record in records:
            _, _, _, height, width = record
            if not grouped or grouped[-1][-1][3:] != (height, width):
                grouped.append([record])
            else:
                grouped[-1].append(record)

        if len(grouped) != 5:
            shapes = [(group[-1][3], group[-1][4]) for group in grouped]
            raise ValueError(
                "Expected 5 MixNet-S spatial resolution groups after blocks, "
                f"got {len(grouped)} groups: {shapes}."
            )

        infos: list[StageInfo] = []
        for stage_id, group in enumerate(grouped):
            timm_stage_index, block_index, channels, height, width = group[-1]
            infos.append(
                StageInfo(
                    stage_id=stage_id,
                    timm_stage_index=timm_stage_index,
                    block_index=block_index,
                    channels=channels,
                    height=height,
                    width=width,
                )
            )
        return tuple(infos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._extract_feature_vector(x)

    def _extract_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        for timm_stage_index, stage in enumerate(self.model.blocks):
            for block_index, block in enumerate(stage):
                x = block(x)
                stage_key = self._insert_after.get((timm_stage_index, block_index))
                if stage_key is not None:
                    x = self._run_deform_block(stage_key, x)
        x = self.model.conv_head(x)
        x = self.model.bn2(x)
        try:
            x = self.model.forward_head(x, pre_logits=True)
        except TypeError:
            x = self.model.forward_head(x)
        return self._to_feature_vector(x)

    def _run_deform_block(self, stage_key: str, x: torch.Tensor) -> torch.Tensor:
        block = self.deform_blocks[stage_key]
        if self.deform_checkpoint and self.training and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def _to_feature_vector(self, features) -> torch.Tensor:
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
                dummy = torch.zeros(1, 3, int(input_size), int(input_size))
                features = self._extract_feature_vector(dummy)
        finally:
            self.model.train(was_training)
        if features.ndim != 2 or features.shape[1] <= 0:
            raise ValueError(
                "mixnet_s_deformable must return a non-empty 2D feature tensor, "
                f"got {tuple(features.shape)}."
            )
        return int(features.shape[1])

    def describe_deformable_stages(self) -> list[dict[str, int]]:
        return [asdict(info) for info in self.stage_infos]

    def collect_deformable_diagnostics(self) -> dict[str, dict[str, float]]:
        diagnostics: dict[str, dict[str, float]] = {}
        for stage_key, block in self.deform_blocks.items():
            values: dict[str, float] = {}
            for attr_name in (
                "last_offset_mean",
                "last_offset_max",
                "last_attention_entropy",
                "last_layer_scale_mean",
            ):
                value = getattr(block, attr_name, None)
                if value is not None:
                    values[attr_name] = float(value.detach().cpu().item())
            diagnostics[f"S{stage_key}"] = values
        return diagnostics
