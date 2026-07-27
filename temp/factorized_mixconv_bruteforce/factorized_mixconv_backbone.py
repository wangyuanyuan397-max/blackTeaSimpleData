"""Local registry module for Factorized MixConv brute-force experiments."""

from __future__ import annotations

from typing import Iterable, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import MixedConv2d

from src.utils.registry import BACKBONES


def normalize_factorized_kernels(kernels: Iterable[int] | None) -> tuple[int, ...]:
    if kernels is None:
        return ()
    normalized = tuple(sorted({int(kernel) for kernel in kernels}))
    invalid = [kernel for kernel in normalized if kernel < 1 or kernel % 2 != 1]
    if invalid:
        raise ValueError(f"factorized_kernels must be positive odd integers: {invalid}")
    if 3 in normalized:
        raise ValueError("Keep 3x3 unchanged for this search; do not include 3.")
    return normalized


def _pair(value, field_name: str) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{field_name} must have length 2, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _square_kernel_size(conv: nn.Conv2d) -> int:
    kernel_h, kernel_w = _pair(conv.kernel_size, "kernel_size")
    if kernel_h != kernel_w:
        raise ValueError(f"Expected a square kernel, got {conv.kernel_size}")
    if kernel_h % 2 != 1:
        raise ValueError(f"Expected an odd kernel, got {conv.kernel_size}")
    return int(kernel_h)


def _symmetric_stride(conv: nn.Conv2d) -> int:
    stride_h, stride_w = _pair(conv.stride, "stride")
    if stride_h != stride_w or stride_h not in (1, 2):
        raise ValueError(f"Only symmetric stride 1 or 2 is supported, got {conv.stride}")
    return int(stride_h)


def _symmetric_dilation(conv: nn.Conv2d) -> int:
    dilation_h, dilation_w = _pair(conv.dilation, "dilation")
    if dilation_h != dilation_w:
        raise ValueError(f"Only symmetric dilation is supported, got {conv.dilation}")
    return int(dilation_h)


def _is_depthwise_conv(conv: nn.Conv2d) -> bool:
    return conv.groups == conv.in_channels and conv.in_channels == conv.out_channels


class FactorizedDepthwiseConv(nn.Module):
    """Replace KxK depthwise conv with 1xK followed by Kx1 depthwise conv."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd.")
        if stride not in (1, 2):
            raise ValueError("Only stride 1 or 2 is supported.")

        padding = int(dilation) * (int(kernel_size) // 2)
        dd = {"device": device, "dtype": dtype}
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)

        self.horizontal = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, padding),
            dilation=(1, self.dilation),
            groups=self.channels,
            bias=bias,
            **dd,
        )
        self.vertical = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=(self.kernel_size, 1),
            stride=(self.stride, 1),
            padding=(padding, 0),
            dilation=(self.dilation, 1),
            groups=self.channels,
            bias=bias,
            **dd,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.horizontal.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.vertical.weight, mode="fan_out", nonlinearity="relu")
        if self.horizontal.bias is not None:
            nn.init.zeros_(self.horizontal.bias)
        if self.vertical.bias is not None:
            nn.init.zeros_(self.vertical.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.horizontal(x)
        x = self.vertical(x)
        return x


def _build_factorized_from_conv(conv: nn.Conv2d) -> FactorizedDepthwiseConv:
    if not _is_depthwise_conv(conv):
        raise ValueError(
            "Factorized MixConv only supports depthwise Conv2d branches; "
            f"got in={conv.in_channels}, out={conv.out_channels}, groups={conv.groups}."
        )
    return FactorizedDepthwiseConv(
        channels=int(conv.in_channels),
        kernel_size=_square_kernel_size(conv),
        stride=_symmetric_stride(conv),
        dilation=_symmetric_dilation(conv),
        bias=conv.bias is not None,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )


@BACKBONES.register("mixnet_s_factorized")
class MixNetSFactorizedBackbone(nn.Module):
    """MixNet-S with selected 5/7/9 depthwise branches factorized as 1xK -> Kx1."""

    def __init__(
        self,
        pretrained: bool = True,
        input_size: int = 408,
        factorized_kernels: Sequence[int] | None = None,
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
        self.factorized_kernels = normalize_factorized_kernels(factorized_kernels)
        self.factorized_branches: list[dict[str, object]] = []
        self._apply_factorization()
        self.out_features = self._infer_out_features(int(input_size))

    def _apply_factorization(self) -> None:
        if not self.factorized_kernels:
            return

        selected = set(self.factorized_kernels)
        for stage_index, stage in enumerate(self.model.blocks):
            for block_index, block in enumerate(stage):
                block_name = f"S{stage_index}B{block_index}"
                old_conv = getattr(block, "conv_dw", None)
                if old_conv is None:
                    continue

                if isinstance(old_conv, MixedConv2d):
                    for branch_name, branch in list(old_conv.items()):
                        if not isinstance(branch, nn.Conv2d):
                            continue
                        kernel = _square_kernel_size(branch)
                        if kernel not in selected:
                            continue
                        old_conv[branch_name] = _build_factorized_from_conv(branch)
                        self.factorized_branches.append(
                            {
                                "block": block_name,
                                "branch": str(branch_name),
                                "kernel_size": kernel,
                                "channels": int(branch.in_channels),
                                "stride": tuple(int(v) for v in _pair(branch.stride, "stride")),
                            }
                        )
                    continue

                if isinstance(old_conv, nn.Conv2d):
                    kernel = _square_kernel_size(old_conv)
                    if kernel not in selected:
                        continue
                    block.conv_dw = _build_factorized_from_conv(old_conv)
                    self.factorized_branches.append(
                        {
                            "block": block_name,
                            "branch": "conv_dw",
                            "kernel_size": kernel,
                            "channels": int(old_conv.in_channels),
                            "stride": tuple(int(v) for v in _pair(old_conv.stride, "stride")),
                        }
                    )

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
                "mixnet_s_factorized must return a non-empty 2D feature tensor, "
                f"got {tuple(features.shape)}"
            )
        return int(features.shape[1])
