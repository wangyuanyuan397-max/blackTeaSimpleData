"""EfficientNetV2-S 的全局—局部多尺度双分支实验。"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from ...utils.registry import BACKBONES


def _build_efficientnet_v2_s(pretrained: bool) -> nn.Module:
    """创建 EfficientNetV2-S，并兼容新旧 torchvision 权重接口。"""
    if hasattr(models, "EfficientNet_V2_S_Weights"):
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        return models.efficientnet_v2_s(weights=weights)
    return models.efficientnet_v2_s(pretrained=pretrained)


class MultiScaleLocalBranch(nn.Module):
    """用 3×3、5×5、7×7 深度卷积提取最终特征图中的局部多尺度线索。"""

    def __init__(self, in_channels: int, local_channels: int = 256) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels 必须大于 0。")
        if local_channels <= 0:
            raise ValueError("local_channels 必须大于 0。")

        # 先用 1×1 卷积降低通道数，控制三个大卷积分支的参数量和显存占用。
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, local_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(local_channels),
            nn.SiLU(inplace=True),
        )

        # 三个分支共享同一个降维特征，但分别观察不同大小的局部感受野。
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        local_channels,
                        local_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        groups=local_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(local_channels),
                    nn.SiLU(inplace=True),
                )
                for kernel_size in (3, 5, 7)
            ]
        )

        # 拼接三个尺度后恢复到 backbone 的原始特征维度，便于和全局向量融合。
        self.fuse = nn.Sequential(
            nn.Conv2d(
                local_channels * len(self.branches),
                in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """返回形状为 [B, C] 的局部多尺度特征向量。"""
        if feature_map.ndim != 4:
            raise ValueError(
                f"局部分支期望输入 [B, C, H, W]，实际为 {tuple(feature_map.shape)}。"
            )
        reduced = self.reduce(feature_map)
        multi_scale = torch.cat(
            [branch(reduced) for branch in self.branches],
            dim=1,
        )
        local_map = self.fuse(multi_scale)
        return self.pool(local_map).flatten(1)


class GlobalLocalGate(nn.Module):
    """根据每个样本的全局与局部描述符，动态生成两个分支的融合权重。"""

    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim 必须大于 0。")
        if hidden_dim <= 0:
            raise ValueError("gate_hidden_dim 必须大于 0。")
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        global_feature: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回加权融合特征以及 [global, local] 两个归一化权重。"""
        gate_logits = self.gate(torch.cat([global_feature, local_feature], dim=1))
        gate_weights = torch.softmax(gate_logits, dim=1)
        fused = (
            gate_weights[:, 0:1] * global_feature
            + gate_weights[:, 1:2] * local_feature
        )
        return fused, gate_weights


@BACKBONES.register("efficientnet_v2_s_global_local_multiscale")
class EfficientNetV2SGlobalLocalMultiScaleBackbone(nn.Module):
    """在 EfficientNetV2-S 最终特征图上构建可消融的全局—局部双分支。

    fusion_mode 的含义：
    - global：只使用 GAP(X)，作为严格 baseline；
    - local：只使用 3/5/7 深度卷积局部分支；
    - concat：拼接全局向量与局部向量；
    - gated：按样本动态加权全局向量与局部向量。
    """

    SUPPORTED_FUSION_MODES = {"global", "local", "concat", "gated"}

    def __init__(
        self,
        pretrained: bool = True,
        fusion_mode: str = "concat",
        local_channels: int = 256,
        gate_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        fusion_mode = str(fusion_mode).lower()
        if fusion_mode not in self.SUPPORTED_FUSION_MODES:
            raise ValueError(
                f"未知 fusion_mode：{fusion_mode!r}，"
                f"可选值为 {sorted(self.SUPPORTED_FUSION_MODES)}。"
            )

        efficientnet = _build_efficientnet_v2_s(bool(pretrained))
        self.features = efficientnet.features
        self.feature_dim = int(efficientnet.classifier[1].in_features)
        self.fusion_mode = fusion_mode
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # global baseline 不创建局部分支，避免未使用参数污染参数量统计。
        self.local_branch: MultiScaleLocalBranch | None = None
        self.fusion_gate: GlobalLocalGate | None = None
        if fusion_mode != "global":
            self.local_branch = MultiScaleLocalBranch(
                in_channels=self.feature_dim,
                local_channels=int(local_channels),
            )
        if fusion_mode == "gated":
            self.fusion_gate = GlobalLocalGate(
                feature_dim=self.feature_dim,
                hidden_dim=int(gate_hidden_dim),
            )

        # concat 会将两个 1280 维向量直接拼接，其余模式仍输出 1280 维。
        self.out_features = (
            self.feature_dim * 2 if fusion_mode == "concat" else self.feature_dim
        )
        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """提取并按 YAML 指定的模式返回分类头所需特征。"""
        feature_map = self.features(images)
        global_feature = self.global_pool(feature_map).flatten(1)

        if self.fusion_mode == "global":
            self.last_gate_weights = None
            return global_feature

        if self.local_branch is None:
            raise RuntimeError("当前融合模式需要局部分支，但局部分支尚未创建。")
        local_feature = self.local_branch(feature_map)

        if self.fusion_mode == "local":
            self.last_gate_weights = None
            return local_feature
        if self.fusion_mode == "concat":
            self.last_gate_weights = None
            return torch.cat([global_feature, local_feature], dim=1)

        if self.fusion_gate is None:
            raise RuntimeError("gated 模式需要融合门控模块，但该模块尚未创建。")
        fused, gate_weights = self.fusion_gate(global_feature, local_feature)
        self.last_gate_weights = gate_weights.detach()
        return fused
