"""MixNet-S with scale-aware aggregation after depthwise MixConv blocks.

The official timm MixNet-S topology and pretrained layers are kept intact.
For selected multi-branch depthwise blocks, SAA is inserted after ``bn2``
(which includes the activation in timm) and immediately before SE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import BACKBONES


VALID_TARGETS = {"all", "down", "mid", "late"}
VALID_MODES = {"residual", "replace", "modulation"}


@dataclass(frozen=True)
class MixConvSAAConfig:
    enabled: bool = True
    target: str = "mid"
    keep_original_kernels: bool = True
    expansion: int = 2
    inter_groups: int = 1
    mode: str = "residual"
    gamma_init: float = 0.1
    heads: str = "auto"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "MixConvSAAConfig":
        values = dict(config or {})
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown mixconv_saa options: {unknown}")
        parsed = cls(**values)
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError(
                "mixnet_s_saa requires mixconv_saa.enabled=true; use the timm "
                "backbone for the baseline."
            )
        if str(self.target).lower() not in VALID_TARGETS:
            raise ValueError(f"target must be one of {sorted(VALID_TARGETS)}")
        if not self.keep_original_kernels:
            raise ValueError("The first SAA experiment must keep the official MixNet-S kernels.")
        if int(self.expansion) <= 0:
            raise ValueError("expansion must be a positive integer.")
        if int(self.inter_groups) <= 0:
            raise ValueError("inter_groups must be a positive integer.")
        if str(self.mode).lower() not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        if str(self.heads).lower() != "auto":
            raise ValueError("heads must be auto so it follows the original MixConv branches.")


def _depthwise_branches(module: nn.Module) -> list[nn.Conv2d]:
    if isinstance(module, nn.Conv2d):
        branches = [module]
    elif hasattr(module, "values"):
        branches = list(module.values())
    else:
        branches = list(getattr(module, "branches", ()))
    if not branches or not all(isinstance(branch, nn.Conv2d) for branch in branches):
        raise TypeError(f"Unsupported MixNet depthwise module: {module.__class__.__name__}")
    return branches


def _stride_is_two(branches: Sequence[nn.Conv2d]) -> bool:
    strides = {tuple(int(value) for value in branch.stride) for branch in branches}
    if len(strides) != 1:
        raise ValueError(f"MixConv branches have inconsistent strides: {sorted(strides)}")
    return next(iter(strides)) == (2, 2)


class ScaleAwareAggregation(nn.Module):
    """Regroup concatenated scale branches, then aggregate within/across groups."""

    def __init__(
        self,
        channels: int,
        heads: int,
        expansion: int = 2,
        inter_groups: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        channels = int(channels)
        heads = int(heads)
        expansion = int(expansion)
        inter_groups = int(inter_groups)
        if heads < 2:
            raise ValueError("SAA requires at least two MixConv scale branches.")
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}.")
        if channels % inter_groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by inter_groups={inter_groups}."
            )
        if expansion <= 0:
            raise ValueError("expansion must be positive.")

        self.channels = channels
        self.heads = heads
        self.num_groups = channels // heads
        self.expansion = expansion
        self.inter_groups = inter_groups
        hidden_channels = channels * expansion
        if hidden_channels % self.num_groups != 0:
            raise ValueError(
                f"expanded channels={hidden_channels} must be divisible by "
                f"SAA groups={self.num_groups}."
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.intra = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                groups=self.num_groups,
                bias=False,
                **factory_kwargs,
            ),
            nn.BatchNorm2d(hidden_channels, **factory_kwargs),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                groups=self.num_groups,
                bias=False,
                **factory_kwargs,
            ),
            nn.BatchNorm2d(channels, **factory_kwargs),
        )
        self.inter = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            groups=inter_groups,
            bias=False,
            **factory_kwargs,
        )

    def scale_regroup(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SAA expects BCHW input, got shape={tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"SAA expects {self.channels} channels, got {channels}.")
        channels_per_scale = channels // self.heads
        return (
            x.reshape(batch, self.heads, channels_per_scale, height, width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(batch, channels, height, width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inter(self.intra(self.scale_regroup(x)))


class ScaleAwareFusion(nn.Module):
    """Apply SAA as a residual adapter, replacement, or bounded modulator."""

    def __init__(
        self,
        channels: int,
        heads: int,
        expansion: int,
        inter_groups: int,
        mode: str,
        gamma_init: float,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        self.saa = ScaleAwareAggregation(
            channels=channels,
            heads=heads,
            expansion=expansion,
            inter_groups=inter_groups,
            device=device,
            dtype=dtype,
        )
        if self.mode in {"residual", "modulation"}:
            self.gamma = nn.Parameter(
                torch.tensor(float(gamma_init), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("gamma", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aggregated = self.saa(x)
        if self.mode == "replace":
            return aggregated
        if self.mode == "residual":
            return x + self.gamma * aggregated
        return x * (1.0 + self.gamma * torch.tanh(aggregated))


class SAAThenSE(nn.Module):
    """Preserve the original SE module while inserting SAA immediately before it."""

    def __init__(self, fusion: ScaleAwareFusion, se: nn.Module) -> None:
        super().__init__()
        self.fusion = fusion
        self.se = se

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.fusion(x))


@BACKBONES.register("mixnet_s_saa")
class MixNetSSAABackbone(nn.Module):
    """Official MixNet-S augmented with post-DW, pre-SE scale-aware fusion."""

    def __init__(
        self,
        pretrained: bool = True,
        input_size: int = 408,
        model_name: str = "mixnet_s",
        mixconv_saa: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if model_name != "mixnet_s":
            raise ValueError("mixnet_s_saa fixes model_name to mixnet_s.")
        self.input_size = int(input_size)
        self.saa_config = MixConvSAAConfig.from_mapping(mixconv_saa)
        self.model = timm.create_model(
            model_name,
            pretrained=bool(pretrained),
            num_classes=0,
            **kwargs,
        )
        self.model_name = model_name
        self.applied_blocks: dict[str, dict[str, Any]] = {}
        self.skipped_blocks: dict[str, dict[str, Any]] = {}
        self._insert_saa_modules()
        self.out_features = int(getattr(self.model, "num_features", 0))
        if self.out_features <= 0:
            raise ValueError("timm MixNet-S did not expose a valid num_features value.")

    def _target_matches(
        self,
        stage_index: int,
        branches: Sequence[nn.Conv2d],
    ) -> bool:
        target = str(self.saa_config.target).lower()
        if target == "all":
            return True
        if target == "down":
            return _stride_is_two(branches)
        if target == "mid":
            return stage_index == 4
        return stage_index == 5

    def _insert_saa_modules(self) -> None:
        for stage_index, stage in enumerate(self.model.blocks):
            for block_index, block in enumerate(stage):
                block_name = f"S{stage_index}B{block_index}"
                branches = _depthwise_branches(block.conv_dw)
                kernels = [int(branch.kernel_size[0]) for branch in branches]
                if len(branches) < 2:
                    self.skipped_blocks[block_name] = {
                        "reason": "single_scale_depthwise",
                        "kernels": kernels,
                    }
                    continue
                if not self._target_matches(stage_index, branches):
                    self.skipped_blocks[block_name] = {
                        "reason": "outside_target",
                        "kernels": kernels,
                    }
                    continue

                branch_channels = [int(branch.out_channels) for branch in branches]
                if len(set(branch_channels)) != 1:
                    raise ValueError(
                        f"{block_name} has unequal MixConv branch widths {branch_channels}; "
                        "scale regrouping requires equal widths."
                    )
                channels = sum(branch_channels)
                reference_weight = branches[0].weight
                fusion = ScaleAwareFusion(
                    channels=channels,
                    heads=len(branches),
                    expansion=int(self.saa_config.expansion),
                    inter_groups=int(self.saa_config.inter_groups),
                    mode=str(self.saa_config.mode),
                    gamma_init=float(self.saa_config.gamma_init),
                    device=reference_weight.device,
                    dtype=reference_weight.dtype,
                )
                block.se = SAAThenSE(fusion=fusion, se=block.se)
                self.applied_blocks[block_name] = {
                    "channels": channels,
                    "heads": len(branches),
                    "branch_channels": branch_channels,
                    "kernels": kernels,
                    "stride": list(branches[0].stride),
                }

        if not self.applied_blocks:
            raise ValueError(
                f"SAA target={self.saa_config.target!r} did not select any multi-scale blocks."
            )

    def saa_summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target": str(self.saa_config.target).lower(),
            "mode": str(self.saa_config.mode).lower(),
            "expansion": int(self.saa_config.expansion),
            "inter_groups": int(self.saa_config.inter_groups),
            "gamma_init": float(self.saa_config.gamma_init),
            "heads": "auto",
            "keep_original_kernels": bool(self.saa_config.keep_original_kernels),
            "modified_block_count": len(self.applied_blocks),
            "applied_blocks": self.applied_blocks,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features") and hasattr(self.model, "forward_head"):
            features = self.model.forward_features(x)
            try:
                features = self.model.forward_head(features, pre_logits=True)
            except TypeError:
                features = self.model.forward_head(features)
        else:
            features = self.model(x)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.ndim == 2:
            return features
        if features.ndim == 4:
            if features.shape[-1] == self.out_features:
                return features.mean(dim=(1, 2))
            return torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
        if features.ndim == 3:
            return features.mean(dim=1)
        return torch.flatten(features, 1)
