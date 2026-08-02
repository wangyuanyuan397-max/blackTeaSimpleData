"""MixNet-S with partial-channel high/low Fourier filtering."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...utils.registry import BACKBONES
from .mixnet_search import ALL_BLOCKS, parse_block_name


_BLOCK_INDEX_PATTERN = re.compile(r"^BLOCKS\[(\d+)\]\[(\d+)\]$")


def normalize_mixnet_block_name(block_name: str) -> str:
    """Normalize `S2B3` or `blocks[2][3]` into the canonical `S2B3` form."""
    value = str(block_name).strip().upper().replace(" ", "")
    match = _BLOCK_INDEX_PATTERN.fullmatch(value)
    if match:
        return f"S{match.group(1)}B{match.group(2)}"
    return value


def normalize_mixnet_block_names(block_names: Sequence[str] | None) -> tuple[str, ...]:
    if block_names is None:
        return ()
    normalized = tuple(dict.fromkeys(normalize_mixnet_block_name(name) for name in block_names))
    unknown = [name for name in normalized if name not in ALL_BLOCKS]
    if unknown:
        raise ValueError(f"Unknown MixNet-S blocks for Fourier filtering: {unknown}.")
    return normalized


def _block_feature_channels(block: nn.Module) -> int:
    bn2 = getattr(block, "bn2", None)
    num_features = getattr(bn2, "num_features", None)
    if num_features is not None:
        return int(num_features)

    conv_dw = getattr(block, "conv_dw", None)
    out_channels = getattr(conv_dw, "out_channels", None)
    if out_channels is not None:
        return int(out_channels)

    raise ValueError("Cannot infer MixNet-S block channels before SE.")


@dataclass(frozen=True)
class FourierBlockInfo:
    block_name: str
    timm_stage_index: int
    block_index: int
    channels: int
    frequency_channels: int
    local_channels: int


class HighLowFourierFilter2d(nn.Module):
    """Residual partial-channel high/low Fourier filter for [B, C, H, W]."""

    def __init__(
        self,
        channels: int,
        frequency_ratio: float = 0.25,
        low_frequency_radius_ratio: float = 0.35,
        reduction: int = 4,
        residual_scale_init: float = 0.0,
        min_frequency_channels: int = 1,
    ) -> None:
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if not 0.0 < float(frequency_ratio) <= 1.0:
            raise ValueError("frequency_ratio must be in (0, 1].")
        if not 0.0 < float(low_frequency_radius_ratio) < 1.0:
            raise ValueError("low_frequency_radius_ratio must be in (0, 1).")
        if int(reduction) <= 0:
            raise ValueError("reduction must be positive.")
        if int(min_frequency_channels) <= 0:
            raise ValueError("min_frequency_channels must be positive.")

        frequency_channels = max(
            int(min_frequency_channels),
            int(round(channels * float(frequency_ratio))),
        )
        frequency_channels = min(channels, frequency_channels)

        self.channels = channels
        self.frequency_channels = frequency_channels
        self.local_channels = channels - frequency_channels
        self.low_frequency_radius_ratio = float(low_frequency_radius_ratio)

        hidden_channels = max(8, self.frequency_channels // int(reduction))
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.frequency_channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.frequency_channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, self.frequency_channels * 2, kernel_size=1),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(self.frequency_channels * 2, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, self.frequency_channels, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(
            torch.full((1, self.frequency_channels, 1, 1), float(residual_scale_init))
        )

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}.")

        if self.local_channels > 0:
            local_x, frequency_x = torch.split(
                x,
                [self.local_channels, self.frequency_channels],
                dim=1,
            )
        else:
            local_x = None
            frequency_x = x

        residual = frequency_x
        normalized = self.norm(frequency_x)
        low_x, high_x = self._split_high_low(normalized)

        gates = torch.sigmoid(self.gate(normalized))
        gates = gates.view(
            x.shape[0],
            2,
            self.frequency_channels,
            1,
            1,
        )
        low_x = low_x.to(dtype=gates.dtype) * gates[:, 0]
        high_x = high_x.to(dtype=gates.dtype) * gates[:, 1]
        delta = self.proj(torch.cat([low_x, high_x], dim=1))
        scale = self.residual_scale.to(dtype=residual.dtype)
        filtered = residual + scale * delta.to(dtype=residual.dtype)

        if local_x is None:
            return filtered
        return torch.cat([local_x, filtered], dim=1)

    def _split_high_low(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = int(x.shape[-2]), int(x.shape[-1])
        spectrum = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
        low_mask = self._low_frequency_mask(
            height=height,
            width=width,
            device=x.device,
        )
        low_spectrum = spectrum * low_mask
        high_spectrum = spectrum * (1.0 - low_mask)
        low_x = torch.fft.irfft2(low_spectrum, s=(height, width), dim=(-2, -1), norm="ortho")
        high_x = torch.fft.irfft2(high_spectrum, s=(height, width), dim=(-2, -1), norm="ortho")
        return low_x, high_x

    def _low_frequency_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        fy = torch.fft.fftfreq(height, device=device).abs().view(height, 1)
        fx = torch.fft.rfftfreq(width, device=device).abs().view(1, width // 2 + 1)
        radius = torch.sqrt(fy.square() + fx.square())
        cutoff = radius.amax().clamp_min(1e-6) * self.low_frequency_radius_ratio
        return (radius <= cutoff).to(dtype=torch.float32).view(1, 1, height, width // 2 + 1)


class FourierBeforeSE(nn.Module):
    """Place Fourier filtering immediately before the original SE module."""

    def __init__(
        self,
        fourier: HighLowFourierFilter2d,
        se: nn.Module,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.fourier = fourier
        self.se = se
        self.use_checkpoint = bool(use_checkpoint)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            x = checkpoint(self.fourier, x, use_reentrant=False)
        else:
            x = self.fourier(x)
        return self.se(x)


@BACKBONES.register("mixnet_s_fourier")
class MixNetSFourierBackbone(nn.Module):
    """Official timm MixNet-S with optional Partial High/Low Fourier filters."""

    def __init__(
        self,
        pretrained: bool = True,
        input_size: int = 408,
        model_name: str = "mixnet_s",
        fourier_blocks: Sequence[str] | None = None,
        frequency_ratio: float = 0.25,
        low_frequency_radius_ratio: float = 0.35,
        reduction: int = 4,
        residual_scale_init: float = 0.0,
        min_frequency_channels: int = 1,
        fourier_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if model_name != "mixnet_s":
            raise ValueError("mixnet_s_fourier fixes model_name to mixnet_s.")

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        self.input_size = int(input_size)
        self.fourier_blocks = normalize_mixnet_block_names(fourier_blocks)
        self.frequency_ratio = float(frequency_ratio)
        self.low_frequency_radius_ratio = float(low_frequency_radius_ratio)
        self.reduction = int(reduction)
        self.residual_scale_init = float(residual_scale_init)
        self.min_frequency_channels = int(min_frequency_channels)
        self.fourier_checkpoint = bool(fourier_checkpoint)

        self.applied_blocks: dict[str, FourierBlockInfo] = {}
        self._insert_fourier_filters()
        self.out_features = self._infer_out_features(self.input_size)

    def _insert_fourier_filters(self) -> None:
        for block_name in self.fourier_blocks:
            stage_index, block_index = parse_block_name(block_name)
            try:
                block = self.model.blocks[stage_index][block_index]
            except IndexError as exc:
                raise ValueError(f"MixNet-S block does not exist: {block_name}") from exc
            if not hasattr(block, "se"):
                raise ValueError(f"MixNet-S block has no SE slot: {block_name}.")

            channels = _block_feature_channels(block)
            fourier = HighLowFourierFilter2d(
                channels=channels,
                frequency_ratio=self.frequency_ratio,
                low_frequency_radius_ratio=self.low_frequency_radius_ratio,
                reduction=self.reduction,
                residual_scale_init=self.residual_scale_init,
                min_frequency_channels=self.min_frequency_channels,
            )
            old_se = block.se
            block.se = FourierBeforeSE(
                fourier=fourier,
                se=old_se,
                use_checkpoint=self.fourier_checkpoint,
            )
            self.applied_blocks[block_name] = FourierBlockInfo(
                block_name=block_name,
                timm_stage_index=stage_index,
                block_index=block_index,
                channels=channels,
                frequency_channels=fourier.frequency_channels,
                local_channels=fourier.local_channels,
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
                dummy = torch.zeros(1, 3, int(input_size), int(input_size))
                features = self._extract_feature_vector(dummy)
        finally:
            self.model.train(was_training)
        if features.ndim != 2 or features.shape[1] <= 0:
            raise ValueError(
                "mixnet_s_fourier must return a non-empty 2D feature tensor, "
                f"got {tuple(features.shape)}."
            )
        return int(features.shape[1])

    def describe_fourier_blocks(self) -> list[dict[str, int | str]]:
        return [asdict(info) for info in self.applied_blocks.values()]
