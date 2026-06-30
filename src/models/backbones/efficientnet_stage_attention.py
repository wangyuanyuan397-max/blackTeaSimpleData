'''EfficientNetV2-S 的 post-stage attention 位置消融模型。'''

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torchvision import models

from ...utils.registry import BACKBONES


def _build_efficientnet_v2_s(pretrained: bool) -> nn.Module:
    '''创建 EfficientNetV2-S，并兼容新旧 torchvision 权重接口。'''
    if hasattr(models, 'EfficientNet_V2_S_Weights'):
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        return models.efficientnet_v2_s(weights=weights)
    return models.efficientnet_v2_s(pretrained=pretrained)


class ECALayer(nn.Module):
    '''Efficient Channel Attention：用一维卷积建模局部跨通道关系。'''

    def __init__(self, channels: int, gamma: float = 2.0, bias: float = 1.0):
        super().__init__()
        # 按 ECA 论文中的自适应规则，为不同通道数选择奇数卷积核。
        kernel_size = int(abs((math.log2(channels) + bias) / gamma))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel_size = max(kernel_size, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(weights)


class CBAMLayer(nn.Module):
    '''CBAM：依次执行通道注意力和空间注意力。'''

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial_conv = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            bias=False,
        )
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_descriptor = self.channel_mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        max_descriptor = self.channel_mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        x = x * self.channel_sigmoid(avg_descriptor + max_descriptor)
        spatial_descriptor = torch.cat(
            [torch.mean(x, dim=1, keepdim=True), torch.amax(x, dim=1, keepdim=True)],
            dim=1,
        )
        return x * self.spatial_sigmoid(self.spatial_conv(spatial_descriptor))


class CoordinateAttentionLayer(nn.Module):
    '''Coordinate Attention：分别沿高度和宽度编码方向位置信息。'''

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.act = nn.Hardswish(inplace=True)
        self.expand_h = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.expand_w = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        pooled_h = torch.mean(x, dim=3, keepdim=True)
        pooled_w = torch.mean(x, dim=2, keepdim=True).transpose(2, 3)
        encoded = torch.cat([pooled_h, pooled_w], dim=2)
        encoded = self.act(self.bn(self.reduce(encoded)))
        encoded_h, encoded_w = torch.split(encoded, [height, width], dim=2)
        encoded_w = encoded_w.transpose(2, 3)
        attention_h = torch.sigmoid(self.expand_h(encoded_h))
        attention_w = torch.sigmoid(self.expand_w(encoded_w))
        return x * attention_h * attention_w


def _normalize_positions(positions: Iterable[int] | None) -> tuple[int, ...]:
    '''把 YAML 中的位置列表规范化，并拒绝越界或重复位置。'''
    normalized = tuple(int(position) for position in (positions or ()))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f'attention_positions 不能包含重复位置：{normalized}')
    invalid = [position for position in normalized if position < 1 or position > 6]
    if invalid:
        raise ValueError(f'attention_positions 只允许 1 到 6，收到：{invalid}')
    return normalized


def _build_attention(
    attention_type: str,
    channels: int,
    cbam_reduction: int,
    cbam_spatial_kernel_size: int,
    ca_reduction: int,
) -> nn.Module:
    '''根据统一字符串创建某个 stage 后的 attention 模块。'''
    if attention_type == 'eca':
        return ECALayer(channels)
    if attention_type == 'cbam':
        return CBAMLayer(channels, cbam_reduction, cbam_spatial_kernel_size)
    if attention_type == 'ca':
        return CoordinateAttentionLayer(channels, ca_reduction)
    raise ValueError(f'未知 attention_type：{attention_type!r}')


@BACKBONES.register('efficientnet_v2_s_stage_attention')
class EfficientNetV2SStageAttentionBackbone(nn.Module):
    '''在 EfficientNetV2-S 六个主干 stage 的输出后插入 attention。'''

    STAGE_CHANNELS: Sequence[int] = (24, 48, 64, 128, 160, 1280)

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        attention_type: str = 'none',
        attention_positions: Iterable[int] | None = None,
        final_dropout: float = 0.2,
        cbam_reduction: int = 16,
        cbam_spatial_kernel_size: int = 7,
        ca_reduction: int = 32,
    ):
        super().__init__()
        attention_type = str(attention_type).strip().lower()
        positions = _normalize_positions(attention_positions)
        if attention_type in {'none', 'baseline'} and positions:
            raise ValueError('baseline/none 不应配置 attention_positions。')
        if attention_type not in {'none', 'baseline', 'eca', 'cbam', 'ca'}:
            raise ValueError('attention_type 必须是 none、eca、cbam 或 ca。')
        if attention_type in {'eca', 'cbam', 'ca'} and not positions:
            raise ValueError(f'{attention_type} 至少需要一个插入位置。')

        efficientnet = _build_efficientnet_v2_s(pretrained)
        features = efficientnet.features
        if len(features) != 8:
            raise RuntimeError(
                f'当前 torchvision EfficientNetV2-S features 数量为 {len(features)}，'
                '预期为 8；请重新确认 stage 映射。'
            )

        # 为严格对应 Input → Stage1...Stage6 → GAP，把 stem 合入 Stage 1，
        # 把最后的 1×1 head conv 合入 Stage 6；attention 后不再隐藏其他卷积层。
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
        self.attention_type = attention_type
        self.attention_positions = positions
        self.attentions = nn.ModuleDict()

        # 先创建所有实验共享的分类器，避免 attention 数量改变随机数消耗顺序，
        # 从而保证相同随机种子下 baseline 与各位置实验的分类器初始权重一致。
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=final_dropout) if final_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(1280, num_classes)

        if attention_type in {'eca', 'cbam', 'ca'}:
            for position in positions:
                self.attentions[str(position)] = _build_attention(
                    attention_type=attention_type,
                    channels=self.STAGE_CHANNELS[position - 1],
                    cbam_reduction=cbam_reduction,
                    cbam_spatial_kernel_size=cbam_spatial_kernel_size,
                    ca_reduction=ca_reduction,
                )
        # forward 已经返回类别 logits，外层必须搭配 identity head。
        self.out_features = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for position, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if str(position) in self.attentions:
                x = self.attentions[str(position)](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)
