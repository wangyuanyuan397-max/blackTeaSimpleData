"""EfficientNetV2-S 最终特征图上的 MSR、ECA、SE 与 DCN 精炼实验。"""

import math

import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import DeformConv2d

from ...utils.registry import BACKBONES


def _build_efficientnet_v2_s(pretrained: bool) -> nn.Module:
    """创建 EfficientNetV2-S，并兼容新旧 torchvision 权重接口。"""
    if hasattr(models, "EfficientNet_V2_S_Weights"):
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        return models.efficientnet_v2_s(weights=weights)
    return models.efficientnet_v2_s(pretrained=pretrained)


class MultiScaleRefinement(nn.Module):
    """以 3/5/7 深度卷积分支精炼最终特征图，再通过残差返回原通道数。"""

    def __init__(
        self,
        in_channels: int = 1280,
        refine_channels: int = 256,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        if refine_channels <= 0:
            raise ValueError("refine_channels 必须大于 0。")
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, refine_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(refine_channels),
            nn.SiLU(inplace=True),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        refine_channels,
                        refine_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        groups=refine_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(refine_channels),
                    nn.SiLU(inplace=True),
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                refine_channels * 3,
                refine_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(refine_channels),
            nn.SiLU(inplace=True),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(refine_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        multi_scale = torch.cat(
            [branch(reduced) for branch in self.branches],
            dim=1,
        )
        refined = self.expand(self.fuse(multi_scale))
        return x + self.gamma * refined


class EfficientChannelAttention(nn.Module):
    """ECA：对 GAP 通道描述符使用自适应一维卷积进行轻量重标定。"""

    def __init__(
        self,
        channels: int = 1280,
        gamma: float = 2.0,
        bias: float = 1.0,
    ) -> None:
        super().__init__()
        kernel_size = int(abs((math.log2(channels) + bias) / gamma))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel_size = max(kernel_size, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avgpool(x).squeeze(-1).transpose(-1, -2)
        weights = self.channel_conv(weights).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(weights)


class SqueezeExcitationAttention(nn.Module):
    """SE：通过通道压缩和激励重新标定 MSR 输出。"""

    def __init__(self, channels: int = 1280, reduction: int = 16) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError("se_reduction 必须大于 0。")
        hidden_channels = max(channels // reduction, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.excitation(self.avgpool(x))


class DeformableRefinement(nn.Module):
    """用零初始化偏移的 3×3 Deformable Conv 精炼不规则叶片特征。"""

    def __init__(
        self,
        in_channels: int = 1280,
        refine_channels: int = 256,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        if refine_channels <= 0:
            raise ValueError("refine_channels 必须大于 0。")
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, refine_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(refine_channels),
            nn.SiLU(inplace=True),
        )
        self.offset = nn.Conv2d(
            refine_channels,
            2 * 3 * 3,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.deformable_conv = DeformConv2d(
            refine_channels,
            refine_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(refine_channels)
        self.activation = nn.SiLU(inplace=True)
        self.expand = nn.Sequential(
            nn.Conv2d(refine_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        offsets = self.offset(reduced)
        refined = self.deformable_conv(reduced, offsets)
        refined = self.activation(self.norm(refined))
        refined = self.expand(refined)
        return x + self.gamma * refined


@BACKBONES.register("efficientnet_v2_s_final_refinement")
class EfficientNetV2SFinalRefinementBackbone(nn.Module):
    """将指定精炼模块严格放在 EfficientNetV2-S final map 与 GAP 之间。"""

    SUPPORTED_VARIANTS = {"msr", "msr_eca", "dcn", "msr_se"}

    def __init__(
        self,
        pretrained: bool = True,
        variant: str = "msr",
        refine_channels: int = 256,
        residual_gamma_init: float = 0.1,
        se_reduction: int = 16,
        eca_gamma: float = 2.0,
        eca_bias: float = 1.0,
    ) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"未知 final refinement variant：{variant!r}，"
                f"可选值为 {sorted(self.SUPPORTED_VARIANTS)}。"
            )
        efficientnet = _build_efficientnet_v2_s(pretrained)
        self.features = efficientnet.features
        self.variant = variant
        self.out_features = 1280

        if variant == "dcn":
            self.refinement = DeformableRefinement(
                in_channels=self.out_features,
                refine_channels=refine_channels,
                gamma_init=residual_gamma_init,
            )
            self.attention = nn.Identity()
        else:
            self.refinement = MultiScaleRefinement(
                in_channels=self.out_features,
                refine_channels=refine_channels,
                gamma_init=residual_gamma_init,
            )
            if variant == "msr_eca":
                self.attention = EfficientChannelAttention(
                    channels=self.out_features,
                    gamma=eca_gamma,
                    bias=eca_bias,
                )
            elif variant == "msr_se":
                self.attention = SqueezeExcitationAttention(
                    channels=self.out_features,
                    reduction=se_reduction,
                )
            else:
                self.attention = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        final_feature_map = self.features(x)
        refined_feature_map = self.refinement(final_feature_map)
        refined_feature_map = self.attention(refined_feature_map)
        pooled = self.avgpool(refined_feature_map)
        return torch.flatten(pooled, 1)
