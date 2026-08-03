"""Lite-Mono-inspired MixNet-S variants.

The modules in this file keep `mixnet_s` as the baseline backbone and add
optional low-risk switches for stage-level LGFI, hybrid dilated MixConv, CDC
stage rewrites, pooled RGB injection, cross-stage residual fusion, and the
existing partial Fourier filter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import MixedConv2d
from torch.utils.checkpoint import checkpoint

from ...utils.registry import BACKBONES
from .mixnet_fourier import (
    FourierBlockInfo,
    FourierBeforeSE,
    HighLowFourierFilter2d,
    _block_feature_channels,
    normalize_mixnet_block_names,
)
from .mixnet_search import ALL_BLOCKS, parse_block_name, resize_depthwise_kernel, split_channels


def _first_depthwise_conv(module: nn.Module) -> nn.Conv2d:
    if isinstance(module, MixedConv2d):
        return next(iter(module.values()))
    if isinstance(module, HybridDilatedMixedDepthwiseConv2d):
        return module.branches[0]
    if isinstance(module, nn.Conv2d):
        return module
    raise TypeError(f"Unsupported depthwise conv module: {module.__class__.__name__}")


def _conv_records(module: nn.Module):
    if isinstance(module, MixedConv2d):
        start = 0
        for split, conv in zip(module.splits, module.values()):
            yield start, start + int(split), conv
            start += int(split)
        return
    if isinstance(module, HybridDilatedMixedDepthwiseConv2d):
        start = 0
        for split, conv in zip(module.out_splits, module.branches):
            yield start, start + int(split), conv
            start += int(split)
        return
    if isinstance(module, nn.Conv2d):
        yield 0, int(module.out_channels), module
        return
    raise TypeError(f"Unsupported depthwise conv module: {module.__class__.__name__}")


def _find_record(records, channel_index: int):
    for start, end, conv in records:
        if start <= channel_index < end:
            return start, end, conv
    raise IndexError(f"Channel {channel_index} not found in depthwise conv.")


def _copy_depthwise_weights_by_channel(old_module: nn.Module, new_module: nn.Module) -> None:
    old_records = list(_conv_records(old_module))
    new_records = list(_conv_records(new_module))
    with torch.no_grad():
        for new_start, new_end, new_conv in new_records:
            new_kernel_size = int(new_conv.weight.shape[-1])
            for channel_index in range(new_start, new_end):
                old_start, _, old_conv = _find_record(old_records, channel_index)
                old_local = channel_index - old_start
                new_local = channel_index - new_start
                resized = resize_depthwise_kernel(
                    old_conv.weight.detach()[old_local:old_local + 1],
                    new_kernel_size,
                )
                new_conv.weight[new_local:new_local + 1].copy_(resized)
                if new_conv.bias is not None and old_conv.bias is not None:
                    new_conv.bias[new_local].copy_(old_conv.bias.detach()[old_local])


def _same_padding(kernel_size: int, dilation: int) -> int:
    return int(dilation) * (int(kernel_size) - 1) // 2


def _effective_kernel_to_dilation(kernel_size: int) -> int:
    if int(kernel_size) < 3 or int(kernel_size) % 2 != 1:
        raise ValueError(f"Expected odd kernel >= 3, got {kernel_size}.")
    return (int(kernel_size) - 1) // 2


def _normalize_stage_indices(stage_indices: Sequence[int] | None) -> tuple[int, ...]:
    if stage_indices is None:
        return ()
    normalized = tuple(dict.fromkeys(int(stage_index) for stage_index in stage_indices))
    invalid = [stage_index for stage_index in normalized if stage_index < 0 or stage_index > 5]
    if invalid:
        raise ValueError(f"MixNet-S block stage indices must be in [0, 5], got {invalid}.")
    return normalized


def _normalize_block_stage_plan(
    stage_plan: Mapping[int | str, Sequence[int]] | None,
) -> dict[int, tuple[int, ...]]:
    if not stage_plan:
        return {}
    normalized: dict[int, tuple[int, ...]] = {}
    for key, values in stage_plan.items():
        stage_index = int(key)
        if stage_index < 0 or stage_index > 5:
            raise ValueError(f"CDC stage index must be in [0, 5], got {stage_index}.")
        dilations = tuple(int(value) for value in values)
        if any(dilation <= 0 for dilation in dilations):
            raise ValueError(f"CDC dilations must be positive, got {dilations}.")
        normalized[stage_index] = dilations
    return normalized


def _normalize_skip_pairs(skip_pairs: Sequence[Sequence[int]] | None) -> tuple[tuple[int, int], ...]:
    if skip_pairs is None:
        return ()
    normalized: list[tuple[int, int]] = []
    for pair in skip_pairs:
        if len(pair) != 2:
            raise ValueError(f"stage_skip_pairs entries must have two values, got {pair}.")
        source, target = int(pair[0]), int(pair[1])
        if source < 0 or source > 5 or target < 0 or target > 5:
            raise ValueError(f"stage_skip_pairs stages must be in [0, 5], got {pair}.")
        if source >= target:
            raise ValueError(f"stage_skip_pairs must go from shallow to deep, got {pair}.")
        normalized.append((source, target))
    return tuple(dict.fromkeys(normalized))


@dataclass(frozen=True)
class StageInfo:
    stage_index: int
    block_count: int
    channels: int
    height: int
    width: int


@dataclass(frozen=True)
class HybridConvInfo:
    block_name: str
    old_kernels: tuple[int, ...]
    new_kernels: tuple[int, ...]
    new_dilations: tuple[int, ...]


@dataclass(frozen=True)
class CdcBlockInfo:
    block_name: str
    dilation: int
    channels: int


class HybridDilatedMixedDepthwiseConv2d(nn.Module):
    """Mixed depthwise conv where large kernels can become 3x3 dilated branches."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_specs: Sequence[tuple[int, int]],
        stride,
        bias: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not branch_specs:
            raise ValueError("branch_specs cannot be empty.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.branch_specs = tuple((int(k), int(d)) for k, d in branch_specs)
        self.in_splits = split_channels(self.in_channels, len(self.branch_specs))
        self.out_splits = split_channels(self.out_channels, len(self.branch_specs))
        self.splits = list(self.in_splits)

        dd = {"device": device, "dtype": dtype}
        self.branches = nn.ModuleList()
        for (kernel_size, dilation), in_chs, out_chs in zip(
            self.branch_specs,
            self.in_splits,
            self.out_splits,
        ):
            self.branches.append(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=_same_padding(kernel_size, dilation),
                    dilation=dilation,
                    groups=in_chs,
                    bias=bias,
                    **dd,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_split = torch.split(x, self.in_splits, dim=1)
        outputs = [branch(x_part) for branch, x_part in zip(self.branches, x_split)]
        return torch.cat(outputs, dim=1)


class LiteMonoLGFI2d(nn.Module):
    """Stage-level channel interaction with zero-initialized residual scaling."""

    def __init__(
        self,
        channels: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        layer_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError("channels must be positive.")
        self.channels = channels
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.attn_dropout = nn.Dropout(float(attn_dropout))
        self.proj_dropout = nn.Dropout2d(float(proj_dropout))
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale_init)))

        self.last_attention_entropy: torch.Tensor | None = None
        self.last_layer_scale_mean: torch.Tensor | None = None

        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}.")
        batch_size, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")

        normalized = self.norm(x)
        q, k, v = self.qkv(normalized).chunk(3, dim=1)
        q = q.flatten(2)
        k = k.flatten(2)
        v = v.flatten(2)
        scale = float(height * width) ** -0.5
        attention = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        self.last_attention_entropy = (
            -attention.detach().clamp_min(1e-8).log().mul(attention.detach()).sum(dim=-1).mean()
        )
        self.last_layer_scale_mean = self.layer_scale.detach().mean()
        attention = self.attn_dropout(attention)
        mixed = torch.bmm(attention, v).view(batch_size, channels, height, width)
        mixed = self.proj(mixed)
        mixed = self.proj_dropout(mixed)
        scale_param = self.layer_scale.to(dtype=x.dtype)
        return x + scale_param * mixed.to(dtype=x.dtype)


class PooledRgbInjection2d(nn.Module):
    """Fuse adaptively pooled input RGB into a stage feature with residual gating."""

    def __init__(
        self,
        channels: int,
        projection_channels: int | None = None,
        layer_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError("channels must be positive.")
        projection_channels = (
            max(8, channels // 8)
            if projection_channels is None
            else int(projection_channels)
        )
        if projection_channels <= 0:
            raise ValueError("projection_channels must be positive.")

        self.channels = channels
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(3, projection_channels, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + projection_channels, channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale_init)))

    def forward(self, x: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        pooled_rgb = F.adaptive_avg_pool2d(original, output_size=x.shape[-2:])
        rgb = self.rgb_proj(pooled_rgb.to(dtype=x.dtype))
        delta = self.fuse(torch.cat([x, rgb], dim=1))
        scale_param = self.layer_scale.to(dtype=x.dtype)
        return x + scale_param * delta


class StageSkipFusion2d(nn.Module):
    """Residual fusion from a shallower stage into a deeper stage."""

    def __init__(
        self,
        source_channels: int,
        target_channels: int,
        layer_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.source_channels = int(source_channels)
        self.target_channels = int(target_channels)
        self.proj = nn.Conv2d(self.source_channels, self.target_channels, kernel_size=1)
        self.layer_scale = nn.Parameter(
            torch.full((1, self.target_channels, 1, 1), float(layer_scale_init))
        )
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        source = F.adaptive_avg_pool2d(source, output_size=target.shape[-2:])
        delta = self.proj(source.to(dtype=target.dtype))
        scale_param = self.layer_scale.to(dtype=target.dtype)
        return target + scale_param * delta


@BACKBONES.register("mixnet_s_litemono")
class MixNetSLiteMonoBackbone(nn.Module):
    """Official MixNet-S with Lite-Mono-inspired optional modules."""

    def __init__(
        self,
        pretrained: bool = True,
        input_size: int = 408,
        model_name: str = "mixnet_s",
        mixedconv_replacement: str = "none",
        hybrid_replace_blocks: Sequence[str] | None = None,
        keep_dense_kernel_size: int = 5,
        cdc_stage_plans: Mapping[int | str, Sequence[int]] | None = None,
        lgfi_after_blocks: Sequence[int] | None = None,
        lgfi_attn_dropout: float = 0.0,
        lgfi_proj_dropout: float = 0.0,
        lgfi_layer_scale_init: float = 0.0,
        lgfi_checkpoint: bool = False,
        remove_se_blocks: Sequence[str] | None = None,
        rgb_injection_after_blocks: Sequence[int] | None = None,
        rgb_projection_channels: int | None = None,
        rgb_layer_scale_init: float = 0.0,
        stage_skip_pairs: Sequence[Sequence[int]] | None = None,
        skip_layer_scale_init: float = 0.0,
        fourier_blocks: Sequence[str] | None = None,
        frequency_ratio: float = 0.25,
        low_frequency_radius_ratio: float = 0.35,
        fourier_reduction: int = 4,
        fourier_residual_scale_init: float = 0.0,
        min_frequency_channels: int = 1,
        fourier_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if model_name != "mixnet_s":
            raise ValueError("mixnet_s_litemono fixes model_name to mixnet_s.")

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        self.input_size = int(input_size)
        self.mixedconv_replacement = str(mixedconv_replacement).lower()
        self.keep_dense_kernel_size = int(keep_dense_kernel_size)
        self.hybrid_replace_blocks = normalize_mixnet_block_names(hybrid_replace_blocks)
        self.cdc_stage_plans = _normalize_block_stage_plan(cdc_stage_plans)
        self.lgfi_after_blocks = _normalize_stage_indices(lgfi_after_blocks)
        self.lgfi_checkpoint = bool(lgfi_checkpoint)
        self.remove_se_blocks = normalize_mixnet_block_names(remove_se_blocks)
        self.rgb_injection_after_blocks = _normalize_stage_indices(rgb_injection_after_blocks)
        self.stage_skip_pairs = _normalize_skip_pairs(stage_skip_pairs)
        self.fourier_blocks = normalize_mixnet_block_names(fourier_blocks)
        self.fourier_checkpoint = bool(fourier_checkpoint)

        self.hybrid_conv_infos: dict[str, HybridConvInfo] = {}
        self.cdc_block_infos: dict[str, CdcBlockInfo] = {}
        self.applied_fourier_blocks: dict[str, FourierBlockInfo] = {}
        self.removed_se_blocks: tuple[str, ...] = ()

        self._apply_hybrid_mixedconv_replacement()
        self._apply_cdc_stage_plans()
        self._remove_selected_se_blocks()
        self._insert_fourier_filters(
            frequency_ratio=frequency_ratio,
            low_frequency_radius_ratio=low_frequency_radius_ratio,
            reduction=fourier_reduction,
            residual_scale_init=fourier_residual_scale_init,
            min_frequency_channels=min_frequency_channels,
        )

        self.stage_infos = self._discover_stage_infos(self.input_size)
        self.stage_info_by_index = {info.stage_index: info for info in self.stage_infos}
        self.lgfi_blocks = nn.ModuleDict()
        for stage_index in self.lgfi_after_blocks:
            channels = self.stage_info_by_index[stage_index].channels
            self.lgfi_blocks[str(stage_index)] = LiteMonoLGFI2d(
                channels=channels,
                attn_dropout=lgfi_attn_dropout,
                proj_dropout=lgfi_proj_dropout,
                layer_scale_init=lgfi_layer_scale_init,
            )

        self.rgb_blocks = nn.ModuleDict()
        for stage_index in self.rgb_injection_after_blocks:
            channels = self.stage_info_by_index[stage_index].channels
            self.rgb_blocks[str(stage_index)] = PooledRgbInjection2d(
                channels=channels,
                projection_channels=rgb_projection_channels,
                layer_scale_init=rgb_layer_scale_init,
            )

        self.skip_blocks = nn.ModuleDict()
        for source_stage, target_stage in self.stage_skip_pairs:
            source_channels = self.stage_info_by_index[source_stage].channels
            target_channels = self.stage_info_by_index[target_stage].channels
            self.skip_blocks[f"{source_stage}->{target_stage}"] = StageSkipFusion2d(
                source_channels=source_channels,
                target_channels=target_channels,
                layer_scale_init=skip_layer_scale_init,
            )

        self.out_features = self._infer_out_features(self.input_size)

    def _selected_hybrid_blocks(self) -> tuple[str, ...]:
        if self.hybrid_replace_blocks:
            return self.hybrid_replace_blocks
        return ALL_BLOCKS

    def _branch_specs_for_kernels(self, kernels: Sequence[int]) -> tuple[tuple[int, int], ...]:
        mode = self.mixedconv_replacement
        if mode not in {"none", "large_only", "all_dilated"}:
            raise ValueError(
                "mixedconv_replacement must be one of: none, large_only, all_dilated."
            )
        specs: list[tuple[int, int]] = []
        for kernel in kernels:
            kernel = int(kernel)
            if mode == "none":
                specs.append((kernel, 1))
            elif mode == "large_only" and kernel <= self.keep_dense_kernel_size:
                specs.append((kernel, 1))
            else:
                specs.append((3, _effective_kernel_to_dilation(kernel)))
        return tuple(specs)

    def _apply_hybrid_mixedconv_replacement(self) -> None:
        if self.mixedconv_replacement == "none":
            return
        selected_blocks = set(self._selected_hybrid_blocks())
        for block_name in ALL_BLOCKS:
            if block_name not in selected_blocks:
                continue
            stage_index, block_index = parse_block_name(block_name)
            block = self.model.blocks[stage_index][block_index]
            old_conv = getattr(block, "conv_dw", None)
            if not isinstance(old_conv, MixedConv2d):
                continue

            old_kernels = tuple(int(conv.kernel_size[0]) for conv in old_conv.values())
            branch_specs = self._branch_specs_for_kernels(old_kernels)
            if all((old_kernel, 1) == spec for old_kernel, spec in zip(old_kernels, branch_specs)):
                continue

            first = _first_depthwise_conv(old_conv)
            new_conv = HybridDilatedMixedDepthwiseConv2d(
                in_channels=int(old_conv.in_channels),
                out_channels=int(old_conv.out_channels),
                branch_specs=branch_specs,
                stride=first.stride,
                bias=first.bias is not None,
                device=first.weight.device,
                dtype=first.weight.dtype,
            )
            _copy_depthwise_weights_by_channel(old_conv, new_conv)
            block.conv_dw = new_conv
            self.hybrid_conv_infos[block_name] = HybridConvInfo(
                block_name=block_name,
                old_kernels=old_kernels,
                new_kernels=tuple(spec[0] for spec in branch_specs),
                new_dilations=tuple(spec[1] for spec in branch_specs),
            )

    def _apply_cdc_stage_plans(self) -> None:
        for stage_index, dilations in self.cdc_stage_plans.items():
            stage = self.model.blocks[stage_index]
            if len(dilations) != len(stage):
                raise ValueError(
                    f"CDC stage {stage_index} has {len(stage)} blocks, "
                    f"but got {len(dilations)} dilations."
                )
            for block_index, dilation in enumerate(dilations):
                block_name = f"S{stage_index}B{block_index}"
                block = stage[block_index]
                old_conv = block.conv_dw
                first = _first_depthwise_conv(old_conv)
                channels = int(first.in_channels if isinstance(old_conv, nn.Conv2d) else old_conv.in_channels)
                new_conv = nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=first.stride,
                    padding=_same_padding(3, dilation),
                    dilation=int(dilation),
                    groups=channels,
                    bias=first.bias is not None,
                    device=first.weight.device,
                    dtype=first.weight.dtype,
                )
                _copy_depthwise_weights_by_channel(old_conv, new_conv)
                block.conv_dw = new_conv
                self.cdc_block_infos[block_name] = CdcBlockInfo(
                    block_name=block_name,
                    dilation=int(dilation),
                    channels=channels,
                )

    def _remove_selected_se_blocks(self) -> None:
        removed: list[str] = []
        for block_name in self.remove_se_blocks:
            stage_index, block_index = parse_block_name(block_name)
            block = self.model.blocks[stage_index][block_index]
            if not hasattr(block, "se"):
                raise ValueError(f"MixNet-S block has no SE slot: {block_name}.")
            block.se = nn.Identity()
            removed.append(block_name)
        self.removed_se_blocks = tuple(removed)

    def _insert_fourier_filters(
        self,
        frequency_ratio: float,
        low_frequency_radius_ratio: float,
        reduction: int,
        residual_scale_init: float,
        min_frequency_channels: int,
    ) -> None:
        for block_name in self.fourier_blocks:
            stage_index, block_index = parse_block_name(block_name)
            block = self.model.blocks[stage_index][block_index]
            if not hasattr(block, "se"):
                raise ValueError(f"MixNet-S block has no SE slot: {block_name}.")

            channels = _block_feature_channels(block)
            fourier = HighLowFourierFilter2d(
                channels=channels,
                frequency_ratio=frequency_ratio,
                low_frequency_radius_ratio=low_frequency_radius_ratio,
                reduction=reduction,
                residual_scale_init=residual_scale_init,
                min_frequency_channels=min_frequency_channels,
            )
            old_se = block.se
            block.se = FourierBeforeSE(
                fourier=fourier,
                se=old_se,
                use_checkpoint=self.fourier_checkpoint,
            )
            self.applied_fourier_blocks[block_name] = FourierBlockInfo(
                block_name=block_name,
                timm_stage_index=stage_index,
                block_index=block_index,
                channels=channels,
                frequency_channels=fourier.frequency_channels,
                local_channels=fourier.local_channels,
            )

    def _discover_stage_infos(self, input_size: int) -> tuple[StageInfo, ...]:
        required = ("conv_stem", "bn1", "blocks", "conv_head", "bn2", "forward_head")
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise ValueError(f"Unexpected timm MixNet-S structure; missing {missing}.")

        was_training = self.model.training
        self.model.eval()
        infos: list[StageInfo] = []
        try:
            with torch.no_grad():
                x = torch.zeros(1, 3, int(input_size), int(input_size))
                x = self.model.conv_stem(x)
                x = self.model.bn1(x)
                for stage_index, stage in enumerate(self.model.blocks):
                    for block in stage:
                        x = block(x)
                    infos.append(
                        StageInfo(
                            stage_index=stage_index,
                            block_count=len(stage),
                            channels=int(x.shape[1]),
                            height=int(x.shape[2]),
                            width=int(x.shape[3]),
                        )
                    )
        finally:
            self.model.train(was_training)
        return tuple(infos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._extract_feature_vector(x)

    def _extract_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        original = x
        stage_outputs: dict[int, torch.Tensor] = {}
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        for stage_index, stage in enumerate(self.model.blocks):
            for block in stage:
                x = block(x)
            stage_outputs[stage_index] = x
            for source_stage, target_stage in self.stage_skip_pairs:
                if target_stage == stage_index:
                    x = self.skip_blocks[f"{source_stage}->{target_stage}"](
                        x,
                        stage_outputs[source_stage],
                    )
                    stage_outputs[stage_index] = x
            if str(stage_index) in self.rgb_blocks:
                x = self.rgb_blocks[str(stage_index)](x, original)
                stage_outputs[stage_index] = x
            if str(stage_index) in self.lgfi_blocks:
                x = self._run_lgfi_block(str(stage_index), x)
                stage_outputs[stage_index] = x

        x = self.model.conv_head(x)
        x = self.model.bn2(x)
        try:
            x = self.model.forward_head(x, pre_logits=True)
        except TypeError:
            x = self.model.forward_head(x)
        return self._to_feature_vector(x)

    def _run_lgfi_block(self, stage_key: str, x: torch.Tensor) -> torch.Tensor:
        block = self.lgfi_blocks[stage_key]
        if self.lgfi_checkpoint and self.training and x.requires_grad:
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
                "mixnet_s_litemono must return a non-empty 2D feature tensor, "
                f"got {tuple(features.shape)}."
            )
        return int(features.shape[1])

    def describe_stage_infos(self) -> list[dict[str, int]]:
        return [asdict(info) for info in self.stage_infos]

    def describe_hybrid_convs(self) -> list[dict[str, object]]:
        return [asdict(info) for info in self.hybrid_conv_infos.values()]

    def describe_cdc_blocks(self) -> list[dict[str, int | str]]:
        return [asdict(info) for info in self.cdc_block_infos.values()]

    def describe_fourier_blocks(self) -> list[dict[str, int | str]]:
        return [asdict(info) for info in self.applied_fourier_blocks.values()]

    def collect_lgfi_diagnostics(self) -> dict[str, dict[str, float]]:
        diagnostics: dict[str, dict[str, float]] = {}
        for stage_key, block in self.lgfi_blocks.items():
            values: dict[str, float] = {}
            if block.last_attention_entropy is not None:
                values["last_attention_entropy"] = float(
                    block.last_attention_entropy.detach().cpu().item()
                )
            if block.last_layer_scale_mean is not None:
                values["last_layer_scale_mean"] = float(
                    block.last_layer_scale_mean.detach().cpu().item()
                )
            diagnostics[f"blocks[{stage_key}]"] = values
        return diagnostics
