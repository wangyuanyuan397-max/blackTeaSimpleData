'''EfficientNetV2-S 的同层多尺度增强与跨层 feature fusion neck。'''

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ...utils.registry import BACKBONES


def _build_efficientnet_v2_s(pretrained: bool) -> nn.Module:
    '''创建 EfficientNetV2-S，并兼容不同 torchvision 版本。'''
    if hasattr(models, 'EfficientNet_V2_S_Weights'):
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        return models.efficientnet_v2_s(weights=weights)
    return models.efficientnet_v2_s(pretrained=pretrained)


def _normalize_stage_indices(
    values: Iterable[int] | None,
    field_name: str,
) -> tuple[int, ...]:
    '''规范化 stage 编号，并拒绝重复值和 1～6 以外的编号。'''
    normalized = tuple(int(value) for value in (values or ()))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f'{field_name} 不能包含重复 stage：{normalized}')
    invalid = [value for value in normalized if value < 1 or value > 6]
    if invalid:
        raise ValueError(f'{field_name} 只允许 1～6，收到：{invalid}')
    return normalized


class MultiScaleFusionBase(nn.Module):
    '''多尺度分支模块的 concat/add/gated 公共融合逻辑。'''

    def __init__(self, channels: int, branch_count: int, fusion: str):
        super().__init__()
        self.channels = channels
        self.branch_count = branch_count
        self.fusion = str(fusion).lower()
        if self.fusion == 'concat':
            self.concat_fusion = nn.Sequential(
                nn.Conv2d(channels * branch_count, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
        elif self.fusion == 'add':
            self.add_norm = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
        elif self.fusion == 'gated':
            hidden_channels = max(channels // 16, 16)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, hidden_channels),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_channels, branch_count),
            )
            self.gated_norm = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
        else:
            raise ValueError('multiscale_fusion 必须是 concat、add 或 gated。')

    def fuse(self, x: torch.Tensor, branches: Sequence[torch.Tensor]) -> torch.Tensor:
        '''融合各尺度分支，并与输入执行残差相加。'''
        if len(branches) != self.branch_count:
            raise RuntimeError(
                f'多尺度分支数量应为 {self.branch_count}，实际为 {len(branches)}。'
            )
        if self.fusion == 'concat':
            fused = self.concat_fusion(torch.cat(list(branches), dim=1))
        elif self.fusion == 'add':
            fused = self.add_norm(torch.stack(list(branches), dim=0).sum(dim=0))
        else:
            weights = torch.softmax(self.gate(x), dim=1)
            stacked = torch.stack(list(branches), dim=1)
            fused = (stacked * weights[:, :, None, None, None]).sum(dim=1)
            fused = self.gated_norm(fused)
        return x + fused


class MultiScaleKernelModule(MultiScaleFusionBase):
    '''MSK：1×1 与 3×3、5×5、7×7 depthwise 多卷积核分支。'''

    def __init__(self, channels: int, fusion: str = 'concat'):
        super().__init__(channels, branch_count=4, fusion=fusion)
        self.branch_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.branch_3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.branch_5 = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels, bias=False
        )
        self.branch_7 = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(
            x,
            [self.branch_1(x), self.branch_3(x), self.branch_5(x), self.branch_7(x)],
        )


class MultiScaleDilationModule(MultiScaleFusionBase):
    '''MSD：1×1 与 dilation=1/2/3 的 3×3 depthwise 分支。'''

    def __init__(self, channels: int, fusion: str = 'concat'):
        super().__init__(channels, branch_count=4, fusion=fusion)
        self.branch_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dilated_branches = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                )
                for dilation in (1, 2, 3)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.branch_1(x)]
        branches.extend(branch(x) for branch in self.dilated_branches)
        return self.fuse(x, branches)


class MultiScalePoolingModule(MultiScaleFusionBase):
    '''MSP：局部 1×1 分支与 1×1、2×2、4×4 池化金字塔。'''

    def __init__(self, channels: int, fusion: str = 'concat'):
        super().__init__(channels, branch_count=4, fusion=fusion)
        self.local_branch = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pool_sizes = (1, 2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[-2:]
        branches = [self.local_branch(x)]
        for pool_size in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
            branches.append(
                F.interpolate(
                    pooled,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                )
            )
        return self.fuse(x, branches)


class MultiScaleHybridModule(MultiScaleFusionBase):
    '''MSH：局部、膨胀卷积与全局上下文分支的混合模块。'''

    def __init__(self, channels: int, fusion: str = 'concat'):
        super().__init__(channels, branch_count=5, fusion=fusion)
        self.branch_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dilated_branches = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                )
                for dilation in (1, 2, 3)
            ]
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.branch_1(x)]
        branches.extend(branch(x) for branch in self.dilated_branches)
        branches.append(self.global_context(x).expand_as(x))
        return self.fuse(x, branches)


def _build_multiscale_module(
    module_type: str,
    channels: int,
    fusion: str,
) -> nn.Module:
    '''根据 YAML 中的简称构建多尺度模块。'''
    module_classes = {
        'msk': MultiScaleKernelModule,
        'msd': MultiScaleDilationModule,
        'msp': MultiScalePoolingModule,
        'msh': MultiScaleHybridModule,
    }
    if module_type not in module_classes:
        raise ValueError(f'未知 multiscale_type：{module_type!r}')
    return module_classes[module_type](channels=channels, fusion=fusion)


class FeatureFusionNeck(nn.Module):
    '''将多个 stage 特征投影、下采样到 X6 尺寸并执行指定融合。'''

    def __init__(
        self,
        feature_indices: Sequence[int],
        stage_channels: Sequence[int],
        out_channels: int = 128,
        fusion: str = 'concat',
    ):
        super().__init__()
        self.feature_indices = tuple(feature_indices)
        self.out_channels = int(out_channels)
        self.fusion = str(fusion).lower()
        self.projections = nn.ModuleDict(
            {
                str(index): nn.Sequential(
                    nn.Conv2d(
                        stage_channels[index - 1],
                        self.out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.out_channels),
                    nn.SiLU(inplace=True),
                )
                for index in self.feature_indices
            }
        )
        feature_count = len(self.feature_indices)
        if self.fusion == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Conv2d(
                    self.out_channels * feature_count,
                    self.out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.SiLU(inplace=True),
            )
        elif self.fusion == 'add':
            self.fusion_layer = nn.Sequential(
                nn.BatchNorm2d(self.out_channels),
                nn.SiLU(inplace=True),
            )
        elif self.fusion == 'weighted':
            self.feature_logits = nn.Parameter(torch.zeros(feature_count))
            self.fusion_layer = nn.Sequential(
                nn.BatchNorm2d(self.out_channels),
                nn.SiLU(inplace=True),
            )
        elif self.fusion == 'attention':
            hidden_channels = max(self.out_channels // 2, 32)
            self.attention_gate = nn.Sequential(
                nn.Linear(self.out_channels * feature_count, hidden_channels),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_channels, feature_count),
            )
            self.fusion_layer = nn.Sequential(
                nn.BatchNorm2d(self.out_channels),
                nn.SiLU(inplace=True),
            )
        else:
            raise ValueError('neck_fusion 必须是 concat、add、weighted 或 attention。')

    def forward(self, features: dict[int, torch.Tensor]) -> torch.Tensor:
        target_size = features[6].shape[-2:]
        projected = []
        for index in self.feature_indices:
            feature = self.projections[str(index)](features[index])
            if feature.shape[-2:] != target_size:
                feature = F.adaptive_avg_pool2d(feature, output_size=target_size)
            projected.append(feature)

        if self.fusion == 'concat':
            fused = self.fusion_layer(torch.cat(projected, dim=1))
        elif self.fusion == 'add':
            fused = self.fusion_layer(torch.stack(projected, dim=0).sum(dim=0))
        elif self.fusion == 'weighted':
            weights = torch.softmax(self.feature_logits, dim=0)
            stacked = torch.stack(projected, dim=1)
            fused = (stacked * weights[None, :, None, None, None]).sum(dim=1)
            fused = self.fusion_layer(fused)
        else:
            descriptors = [feature.mean(dim=(2, 3)) for feature in projected]
            weights = torch.softmax(
                self.attention_gate(torch.cat(descriptors, dim=1)),
                dim=1,
            )
            stacked = torch.stack(projected, dim=1)
            fused = (stacked * weights[:, :, None, None, None]).sum(dim=1)
            fused = self.fusion_layer(fused)
        return fused


@BACKBONES.register('efficientnet_v2_s_multiscale_fusion')
class EfficientNetV2SMultiScaleFusionBackbone(nn.Module):
    '''统一支持 baseline、同层多尺度、跨层 neck 及二者组合。'''

    STAGE_CHANNELS: Sequence[int] = (24, 48, 64, 128, 160, 1280)

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        final_dropout: float = 0.2,
        multiscale_type: str = 'none',
        multiscale_positions: Iterable[int] | None = None,
        multiscale_fusion: str = 'concat',
        neck_features: Iterable[int] | None = None,
        neck_fusion: str = 'none',
        neck_channels: int = 128,
    ):
        super().__init__()
        multiscale_type = str(multiscale_type).lower()
        multiscale_positions = _normalize_stage_indices(
            multiscale_positions,
            'multiscale_positions',
        )
        neck_features = _normalize_stage_indices(neck_features, 'neck_features')
        neck_fusion = str(neck_fusion).lower()
        if multiscale_type == 'none' and multiscale_positions:
            raise ValueError('multiscale_type=none 时不能配置 multiscale_positions。')
        if multiscale_type != 'none' and not multiscale_positions:
            raise ValueError('启用多尺度模块时必须配置 multiscale_positions。')
        if neck_fusion == 'none' and neck_features:
            raise ValueError('neck_fusion=none 时不能配置 neck_features。')
        if neck_fusion != 'none' and not neck_features:
            raise ValueError('启用 feature fusion neck 时必须配置 neck_features。')
        if neck_features and 6 not in neck_features:
            raise ValueError('neck_features 必须包含 Stage 6，作为目标空间尺寸。')

        efficientnet = _build_efficientnet_v2_s(pretrained)
        features = efficientnet.features
        if len(features) != 8:
            raise RuntimeError(
                f'EfficientNetV2-S features 数量应为 8，实际为 {len(features)}。'
            )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(features[0], features[1]),
                features[2],
                features[3],
                features[4],
                features[5],
                nn.Sequential(features[6], features[7]),
            ]
        )
        self.multiscale_positions = multiscale_positions
        self.neck_features = neck_features
        classifier_channels = int(neck_channels) if neck_features else 1280

        # 分类器先于可变模块创建，保证同种输出维度的实验在相同 seed 下初始化一致。
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(final_dropout) if final_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(classifier_channels, num_classes)

        self.multiscale_modules = nn.ModuleDict()
        if multiscale_type != 'none':
            for position in multiscale_positions:
                self.multiscale_modules[str(position)] = _build_multiscale_module(
                    module_type=multiscale_type,
                    channels=self.STAGE_CHANNELS[position - 1],
                    fusion=multiscale_fusion,
                )

        self.neck = None
        if neck_features:
            self.neck = FeatureFusionNeck(
                feature_indices=neck_features,
                stage_channels=self.STAGE_CHANNELS,
                out_channels=neck_channels,
                fusion=neck_fusion,
            )
        self.out_features = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stage_outputs: dict[int, torch.Tensor] = {}
        for position, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if str(position) in self.multiscale_modules:
                x = self.multiscale_modules[str(position)](x)
            if self.neck is not None and position in self.neck_features:
                stage_outputs[position] = x

        if self.neck is not None:
            x = self.neck(stage_outputs)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)
