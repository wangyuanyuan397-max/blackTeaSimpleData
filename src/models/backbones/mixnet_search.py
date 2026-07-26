"""Configurable MixNet-S MixConv placement search backbone."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import MixedConv2d, create_conv2d

from ...utils.registry import BACKBONES


ALL_BLOCKS = (
    "S0B0",
    "S1B0", "S1B1",
    "S2B0", "S2B1", "S2B2", "S2B3",
    "S3B0", "S3B1", "S3B2",
    "S4B0", "S4B1", "S4B2",
    "S5B0", "S5B1", "S5B2",
)

STAGE_BLOCKS = {
    "S0": ("S0B0",),
    "S1": ("S1B0", "S1B1"),
    "S2": ("S2B0", "S2B1", "S2B2", "S2B3"),
    "S3": ("S3B0", "S3B1", "S3B2"),
    "S4": ("S4B0", "S4B1", "S4B2"),
    "S5": ("S5B0", "S5B1", "S5B2"),
}

PLACEMENTS = {
    "NONE": (),
    "ALL": ALL_BLOCKS,
    "STRIDE2": ("S1B0", "S2B0", "S3B0", "S5B0"),
    "STRIDE1": tuple(block for block in ALL_BLOCKS if block not in {"S1B0", "S2B0", "S3B0", "S5B0"}),
    "EARLY_S01": STAGE_BLOCKS["S0"] + STAGE_BLOCKS["S1"],
    "MIDDLE_S23": STAGE_BLOCKS["S2"] + STAGE_BLOCKS["S3"],
    "LAST2_S45": STAGE_BLOCKS["S4"] + STAGE_BLOCKS["S5"],
    "MIDLATE_S2345": STAGE_BLOCKS["S2"] + STAGE_BLOCKS["S3"] + STAGE_BLOCKS["S4"] + STAGE_BLOCKS["S5"],
    "LATE_S345": STAGE_BLOCKS["S3"] + STAGE_BLOCKS["S4"] + STAGE_BLOCKS["S5"],
    "ONLY_S0": STAGE_BLOCKS["S0"],
    "ONLY_S1": STAGE_BLOCKS["S1"],
    "ONLY_S2": STAGE_BLOCKS["S2"],
    "ONLY_S3": STAGE_BLOCKS["S3"],
    "ONLY_S4": STAGE_BLOCKS["S4"],
    "ONLY_S5": STAGE_BLOCKS["S5"],
    "FIRST_BLOCK": ("S0B0", "S1B0", "S2B0", "S3B0", "S4B0", "S5B0"),
    "REPEAT_ONLY": ("S1B1", "S2B1", "S2B2", "S2B3", "S3B1", "S3B2", "S4B1", "S4B2", "S5B1", "S5B2"),
    "LATE_STRIDE2": ("S2B0", "S3B0", "S5B0"),
    "FINAL_DOWNSAMPLE": ("S5B0",),
}


def split_channels(total_channels: int, num_groups: int) -> list[int]:
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")
    base = total_channels // num_groups
    splits = [base] * num_groups
    splits[0] += total_channels - sum(splits)
    if any(split <= 0 for split in splits):
        raise ValueError(
            f"Cannot split {total_channels} channels into {num_groups} non-empty groups."
        )
    return splits


def normalize_kernel_sizes(kernel_sizes: Iterable[int]) -> tuple[int, ...]:
    kernels = tuple(int(kernel) for kernel in kernel_sizes)
    if not kernels:
        raise ValueError("kernel_sizes cannot be empty.")
    if len(set(kernels)) != len(kernels):
        raise ValueError(f"kernel_sizes contain duplicates: {kernels}")
    if any(kernel < 1 or kernel % 2 != 1 for kernel in kernels):
        raise ValueError(f"kernel_sizes must be positive odd integers: {kernels}")
    return kernels


def build_kernel_plan(
    placement: str,
    kernel_sizes: Iterable[int],
) -> dict[str, tuple[int, ...]]:
    placement = str(placement).upper()
    if placement == "ORIGINAL":
        return {}
    if placement not in PLACEMENTS:
        raise ValueError(f"Unknown MixNet placement: {placement}")
    mix_kernels = normalize_kernel_sizes(kernel_sizes)
    selected = set(PLACEMENTS[placement])
    return {
        block: mix_kernels if block in selected else (3,)
        for block in ALL_BLOCKS
    }


def resize_depthwise_kernel(
    old_weight: torch.Tensor,
    new_kernel_size: int,
) -> torch.Tensor:
    if old_weight.ndim != 4:
        raise ValueError(f"Expected [C, 1, K, K], got {tuple(old_weight.shape)}")
    old_k = int(old_weight.shape[-1])
    if old_k != int(old_weight.shape[-2]):
        raise ValueError("Only square depthwise kernels are supported.")
    if old_k == new_kernel_size:
        return old_weight.clone()

    channels = int(old_weight.shape[0])
    if new_kernel_size > old_k:
        new_weight = old_weight.new_zeros(channels, 1, new_kernel_size, new_kernel_size)
        offset = (new_kernel_size - old_k) // 2
        new_weight[:, :, offset:offset + old_k, offset:offset + old_k] = old_weight
        return new_weight

    offset = (old_k - new_kernel_size) // 2
    return old_weight[
        :, :, offset:offset + new_kernel_size, offset:offset + new_kernel_size
    ].clone()


class SearchMixedDepthwiseConv2d(nn.Module):
    """Depthwise MixConv with optional static or dynamic scale gates."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int],
        stride,
        padding,
        dilation,
        bias: bool,
        gate_type: str = "none",
        gate_reduction: int = 4,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.kernel_sizes = normalize_kernel_sizes(kernel_sizes)
        self.gate_type = str(gate_type).lower()
        if len(self.kernel_sizes) == 1:
            self.gate_type = "none"
        if self.gate_type not in {"none", "static", "sigmoid", "softmax"}:
            raise ValueError("gate_type must be one of: none, static, sigmoid, softmax.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.in_splits = split_channels(self.in_channels, len(self.kernel_sizes))
        self.out_splits = split_channels(self.out_channels, len(self.kernel_sizes))
        dd = {"device": device, "dtype": dtype}

        branches = []
        for kernel, in_chs, out_chs in zip(self.kernel_sizes, self.in_splits, self.out_splits):
            branches.append(
                create_conv2d(
                    in_chs,
                    out_chs,
                    int(kernel),
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=in_chs,
                    bias=bias,
                    **dd,
                )
            )
        self.branches = nn.ModuleList(branches)
        # Keep a timm-compatible attribute name for downstream introspection.
        self.splits = list(self.in_splits)

        if self.gate_type == "static":
            self.static_scale = nn.Parameter(torch.ones(len(self.kernel_sizes), **dd))
            self.gate = None
        elif self.gate_type in {"sigmoid", "softmax"}:
            hidden_channels = max(8, self.in_channels // max(1, int(gate_reduction)))
            self.static_scale = None
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=True, **dd),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, len(self.kernel_sizes), kernel_size=1, bias=True, **dd),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        else:
            self.static_scale = None
            self.gate = None

    def _branch_weights(self, x: torch.Tensor) -> torch.Tensor | None:
        branch_count = len(self.kernel_sizes)
        if self.gate_type == "none":
            return None
        if self.gate_type == "static":
            return self.static_scale.view(1, branch_count)

        logits = self.gate(x).flatten(1)
        if self.gate_type == "sigmoid":
            return 2.0 * torch.sigmoid(logits)
        return branch_count * torch.softmax(logits, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_split = torch.split(x, self.in_splits, dim=1)
        branch_outputs = [branch(x_part) for branch, x_part in zip(self.branches, x_split)]
        weights = self._branch_weights(x)
        if weights is not None:
            branch_outputs = [
                output * weights[:, index].view(-1, 1, 1, 1)
                for index, output in enumerate(branch_outputs)
            ]
        return torch.cat(branch_outputs, dim=1)


def _first_conv(module: nn.Module) -> nn.Conv2d:
    if isinstance(module, MixedConv2d):
        return next(iter(module.values()))
    if isinstance(module, SearchMixedDepthwiseConv2d):
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
    if isinstance(module, SearchMixedDepthwiseConv2d):
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
    raise IndexError(f"Channel {channel_index} not found in old depthwise conv.")


def copy_resized_depthwise_weights(old_module: nn.Module, new_module: SearchMixedDepthwiseConv2d) -> None:
    old_records = list(_conv_records(old_module))
    new_start = 0
    with torch.no_grad():
        for branch in new_module.branches:
            new_k = int(branch.weight.shape[-1])
            for local_index in range(int(branch.out_channels)):
                channel_index = new_start + local_index
                old_start, _, old_conv = _find_record(old_records, channel_index)
                old_local = channel_index - old_start
                resized = resize_depthwise_kernel(
                    old_conv.weight.detach()[old_local:old_local + 1],
                    new_k,
                )
                branch.weight[local_index:local_index + 1].copy_(resized)
                if branch.bias is not None and old_conv.bias is not None:
                    branch.bias[local_index].copy_(old_conv.bias.detach()[old_local])
            new_start += int(branch.out_channels)


def parse_block_name(block_name: str) -> tuple[int, int]:
    value = str(block_name).upper()
    if not value.startswith("S") or "B" not in value:
        raise ValueError(f"Invalid MixNet block name: {block_name}")
    stage_text, block_text = value[1:].split("B", 1)
    return int(stage_text), int(block_text)


@BACKBONES.register("mixnet_s_search")
class MixNetSSearchBackbone(nn.Module):
    """MixNet-S backbone with block-level MixConv placement and scale gates."""

    def __init__(
        self,
        pretrained: bool = True,
        input_size: int = 408,
        placement: str = "ORIGINAL",
        kernel_sizes: Sequence[int] | None = None,
        kernel_plan: Mapping[str, Sequence[int]] | None = None,
        gate_type: str = "none",
        gate_reduction: int = 4,
        model_name: str = "mixnet_s",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        self.placement = str(placement).upper()
        self.gate_type = str(gate_type).lower()
        self.kernel_plan = self._resolve_kernel_plan(kernel_plan, kernel_sizes)
        self.applied_blocks: dict[str, tuple[int, ...]] = {}
        self._apply_kernel_plan(gate_reduction=int(gate_reduction))
        self.out_features = self._infer_out_features(int(input_size))

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
                in_channels=int(first.in_channels if isinstance(old_conv, nn.Conv2d) else old_conv.in_channels),
                out_channels=int(first.out_channels if isinstance(old_conv, nn.Conv2d) else old_conv.out_channels),
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
                dummy = torch.zeros(1, 3, input_size, input_size)
                features = self._extract_feature_vector(dummy)
        finally:
            self.model.train(was_training)
        if features.ndim != 2 or features.shape[1] <= 0:
            raise ValueError(
                f"mixnet_s_search must return a non-empty 2D feature tensor, got {tuple(features.shape)}"
            )
        return int(features.shape[1])
