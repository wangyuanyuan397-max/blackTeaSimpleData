'''EfficientNetV2-S 最终分类头与精炼模块消融实验。'''

import torch
import torch.nn as nn
from torchvision import models

from ...utils.registry import BACKBONES


def _build_efficientnet_v2_s(pretrained: bool) -> nn.Module:
    '''创建 EfficientNetV2-S，并兼容不同 torchvision 版本的权重参数。'''
    if hasattr(models, 'EfficientNet_V2_S_Weights'):
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        return models.efficientnet_v2_s(weights=weights)
    return models.efficientnet_v2_s(pretrained=pretrained)


class AblationMLPHead(nn.Module):
    '''B 实验：GAP 后使用 SiLU MLP 分类头，排除“只是参数更多”的影响。'''

    def __init__(self, in_dim: int = 1280, hidden_dim: int = 512, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvRefine(nn.Module):
    '''C 实验：普通卷积精炼模块，不包含 gate，用来和 Gated-refine 对照。'''

    def __init__(self, in_channels: int = 1280, hidden_channels: int = 256):
        super().__init__()

        # 先用 1×1 卷积降维，降低后续空间卷积的计算量。
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        # 使用 depthwise 3×3 卷积，只做局部空间纹理精炼，不做门控筛选。
        self.local_conv = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        # 再用 1×1 卷积升回 1280 通道，保证能和原始特征做残差相加。
        self.expand = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.reduce(x)
        out = self.local_conv(out)
        out = self.expand(out)
        return self.act(out + identity)


class SERefine(nn.Module):
    '''D 实验：最终分类前再做一次 SE-style 通道重标定。'''

    def __init__(self, in_channels: int = 1280, reduction: int = 16):
        super().__init__()
        hidden_channels = max(in_channels // reduction, 32)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc(scale)
        return x * scale


@BACKBONES.register('efficientnet_v2_s_ablation')
class EfficientNetV2SAblationBackbone(nn.Module):
    '''
    固定 EfficientNetV2-S features，只消融最终 head/refine 模块。

    variant 可选：
    - linear：A，X → GAP → Dropout → Linear
    - mlp：B，X → GAP → Dropout → MLP
    - conv_refine：C，X → ConvRefine → GAP → Dropout → Linear
    - se_refine：D，X → SERefine → GAP → Dropout → Linear

    这里作为 backbone 注册，但 forward 直接返回 logits，因此在外层 ImageClassifier
    中需要搭配 identity head 使用，避免再额外接一层 Linear。
    '''

    def __init__(
        self,
        num_classes: int = 4,
        variant: str = 'linear',
        pretrained: bool = True,
        final_dropout: float = 0.2,
        mlp_hidden_dim: int = 512,
        refine_hidden_channels: int = 256,
        se_reduction: int = 16,
    ):
        super().__init__()

        efficientnet = _build_efficientnet_v2_s(pretrained)

        # 只保留 EfficientNetV2-S 的卷积特征提取部分，不使用 torchvision 自带 classifier。
        self.features = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=final_dropout) if final_dropout > 0 else nn.Identity()
        self.variant = variant

        if variant == 'linear':
            self.refine = nn.Identity()
            self.classifier = nn.Linear(1280, num_classes)
        elif variant == 'mlp':
            self.refine = nn.Identity()
            self.classifier = AblationMLPHead(
                in_dim=1280,
                hidden_dim=mlp_hidden_dim,
                num_classes=num_classes,
            )
        elif variant == 'conv_refine':
            self.refine = ConvRefine(
                in_channels=1280,
                hidden_channels=refine_hidden_channels,
            )
            self.classifier = nn.Linear(1280, num_classes)
        elif variant == 'se_refine':
            self.refine = SERefine(
                in_channels=1280,
                reduction=se_reduction,
            )
            self.classifier = nn.Linear(1280, num_classes)
        else:
            raise ValueError(
                f'Unknown EfficientNetV2-S ablation variant: {variant}. '
                'Expected one of: linear, mlp, conv_refine, se_refine.'
            )

        # 外层 identity head 会读取这个属性；这里返回 logits，所以维度就是类别数。
        self.out_features = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.refine(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)
