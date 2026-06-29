'''面向红茶发酵细粒度分类的 EfficientNetV2-S 门控特征骨干网络。'''

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ...utils.registry import BACKBONES


def _build_efficientnet_v2_s(pretrained: bool) -> nn.Module:
    '''创建 EfficientNetV2-S，并兼容新旧 torchvision 的权重参数接口。'''
    if hasattr(models, 'EfficientNet_V2_S_Weights'):
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        return models.efficientnet_v2_s(weights=weights)
    return models.efficientnet_v2_s(pretrained=pretrained)


class ConvNormActivation(nn.Sequential):
    '''由卷积、批归一化和 SiLU 激活组成的基础投影模块。'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        groups: int = 1,
        activate: bool = True,
    ):
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activate:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class GatedRefinementBlock(nn.Module):
    '''在全局池化前动态筛选局部与通道特征，并保留稳定的残差路径。'''

    def __init__(self, in_channels: int = 1280, refine_channels: int = 256):
        super().__init__()

        # 先用 1×1 卷积把 EfficientNet 的 1280 通道压缩到较轻量的特征空间。
        self.reduce = ConvNormActivation(in_channels, refine_channels)

        # 内容分支使用逐通道 3×3 卷积提取局部颜色、纹理和边缘变化。
        self.content_branch = ConvNormActivation(
            refine_channels,
            refine_channels,
            kernel_size=3,
            groups=refine_channels,
        )

        # 门控分支为每个通道、每个空间位置生成 0 到 1 之间的筛选权重。
        self.gate_branch = nn.Sequential(
            nn.Conv2d(refine_channels, refine_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # 融合后的特征经过 1×1 卷积重新混合通道，再与降维特征做残差相加。
        self.output_projection = ConvNormActivation(
            refine_channels,
            refine_channels,
            activate=False,
        )
        self.output_activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        content = self.content_branch(reduced)
        gate = self.gate_branch(reduced)
        refined = self.output_projection(content * gate)
        return self.output_activation(reduced + refined)


@BACKBONES.register('efficientnet_v2_s_gated_refinement')
class EfficientNetV2SGatedRefinementBackbone(nn.Module):
    '''路线 1：最终特征图经过门控精炼后再执行全局平均池化。'''

    def __init__(self, pretrained: bool = True, refine_channels: int = 256):
        super().__init__()
        efficientnet = _build_efficientnet_v2_s(pretrained)

        # 只保留官方 EfficientNetV2-S 的卷积特征提取部分，移除原池化与分类器。
        self.features = efficientnet.features
        self.refinement = GatedRefinementBlock(
            in_channels=1280,
            refine_channels=refine_channels,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ImageClassifier 会读取这个维度并自动配置后续 LinearHead。
        self.out_features = refine_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.refinement(x)
        return self.pool(x).flatten(1)


class MultiStageGatedFusion(nn.Module):
    '''把浅层、中层、深层特征对齐后，按样本学习三个尺度的融合权重。'''

    def __init__(
        self,
        stage_channels: tuple[int, int, int] = (64, 160, 1280),
        fusion_channels: int = 256,
        gate_hidden_channels: int = 128,
    ):
        super().__init__()

        # 三个 1×1 投影分别把 stage3、stage5、stage7 统一到 256 通道。
        self.projections = nn.ModuleList(
            ConvNormActivation(in_channels, fusion_channels)
            for in_channels in stage_channels
        )

        # 先汇总三个阶段的全局描述，再输出每张图像对应的三个尺度分数。
        self.scale_gate = nn.Sequential(
            nn.Linear(fusion_channels * 3, gate_hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(gate_hidden_channels, 3),
        )

        # 融合后再做一次轻量局部精炼，使不同阶段的特征响应更加协调。
        self.fusion_refinement = ConvNormActivation(
            fusion_channels,
            fusion_channels,
            kernel_size=3,
            groups=fusion_channels,
        )

        # 保存最近一次前向传播的尺度权重，便于后续解释模型偏好哪个阶段。
        self.last_scale_weights: torch.Tensor | None = None

    @staticmethod
    def _resize_to(
        feature: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        '''把特征对齐到目标空间尺寸；下采样用平均池化，上采样用双线性插值。'''
        current_height, current_width = feature.shape[-2:]
        target_height, target_width = target_size
        if (current_height, current_width) == target_size:
            return feature
        if current_height >= target_height and current_width >= target_width:
            return F.adaptive_avg_pool2d(feature, target_size)
        return F.interpolate(
            feature,
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )

    def forward(
        self,
        stage3: torch.Tensor,
        stage5: torch.Tensor,
        stage7: torch.Tensor,
    ) -> torch.Tensor:
        raw_features = (stage3, stage5, stage7)
        target_size = stage5.shape[-2:]

        # 投影并对齐后，三个张量均为 [B, fusion_channels, 14, 14]。
        aligned_features = [
            self._resize_to(projection(feature), target_size)
            for projection, feature in zip(self.projections, raw_features)
        ]

        # 每个阶段先全局池化，再拼接生成与当前输入图像相关的尺度权重。
        stage_descriptors = [
            F.adaptive_avg_pool2d(feature, 1).flatten(1)
            for feature in aligned_features
        ]
        gate_logits = self.scale_gate(torch.cat(stage_descriptors, dim=1))
        scale_weights = torch.softmax(gate_logits, dim=1)
        self.last_scale_weights = scale_weights.detach()

        # softmax 保证三个阶段权重之和为 1，避免无约束相加导致数值膨胀。
        fused = sum(
            feature * scale_weights[:, index].view(-1, 1, 1, 1)
            for index, feature in enumerate(aligned_features)
        )
        return fused + self.fusion_refinement(fused)


@BACKBONES.register('efficientnet_v2_s_multistage_gated_fusion')
class EfficientNetV2SMultiStageGatedFusionBackbone(nn.Module):
    '''路线 3：融合 EfficientNetV2-S 第 3、5、7 阶段的多层级特征。'''

    def __init__(
        self,
        pretrained: bool = True,
        fusion_channels: int = 256,
        gate_hidden_channels: int = 128,
    ):
        super().__init__()
        efficientnet = _build_efficientnet_v2_s(pretrained)

        # torchvision 的 features[3]、[5]、[7] 分别对应 28×28、14×14、7×7。
        self.features = efficientnet.features
        self.fusion = MultiStageGatedFusion(
            stage_channels=(64, 160, 1280),
            fusion_channels=fusion_channels,
            gate_hidden_channels=gate_hidden_channels,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_features = fusion_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected_features: List[torch.Tensor] = []

        # 逐段执行官方主干，并只保存路线所需的第 3、5、7 阶段输出。
        for stage_index, stage in enumerate(self.features):
            x = stage(x)
            if stage_index in (3, 5, 7):
                selected_features.append(x)

        if len(selected_features) != 3:
            raise RuntimeError(
                'EfficientNetV2-S 中间特征提取失败：预期得到 stage3、stage5、stage7。'
            )

        x = self.fusion(*selected_features)
        return self.pool(x).flatten(1)
