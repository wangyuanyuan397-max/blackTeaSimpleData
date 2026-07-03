'''EfficientNetV2-S 的冻结/部分解冻与 stage 输出截断实验。'''

from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torchvision import models

from ...utils.registry import BACKBONES


def _build_efficientnet_v2_s(pretrained: bool) -> nn.Module:
    '''创建 EfficientNetV2-S，并兼容不同 torchvision 权重接口。'''
    if hasattr(models, 'EfficientNet_V2_S_Weights'):
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        return models.efficientnet_v2_s(weights=weights)
    return models.efficientnet_v2_s(pretrained=pretrained)


@BACKBONES.register('efficientnet_v2_s_stage_probe')
class EfficientNetV2SStageProbeBackbone(nn.Module):
    '''暴露指定 stage 特征，并精确控制哪些 stage 可训练。'''

    STAGE_CHANNELS: Sequence[int] = (24, 48, 64, 128, 160, 1280)

    def __init__(
        self,
        pretrained: bool = True,
        output_stage: int = 6,
        trainable_stages: Iterable[int] | None = None,
    ):
        super().__init__()
        output_stage = int(output_stage)
        if output_stage < 1 or output_stage > 6:
            raise ValueError('output_stage 只允许 1～6。')
        trainable_stages = tuple(int(stage) for stage in (trainable_stages or ()))
        if len(set(trainable_stages)) != len(trainable_stages):
            raise ValueError('trainable_stages 不能包含重复 stage。')
        invalid = [
            stage
            for stage in trainable_stages
            if stage < 1 or stage > output_stage
        ]
        if invalid:
            raise ValueError(
                f'trainable_stages 必须位于 1～output_stage，收到：{invalid}'
            )

        efficientnet = _build_efficientnet_v2_s(pretrained)
        features = efficientnet.features
        if len(features) != 8:
            raise RuntimeError(
                f'EfficientNetV2-S features 数量应为 8，实际为 {len(features)}。'
            )
        all_stages = [
            nn.Sequential(features[0], features[1]),
            features[2],
            features[3],
            features[4],
            features[5],
            nn.Sequential(features[6], features[7]),
        ]
        # 截断实验直接丢弃更深 stage，参数量和计算量都只统计真实使用部分。
        self.stages = nn.ModuleList(all_stages[:output_stage])
        self.output_stage = output_stage
        self.trainable_stages = frozenset(trainable_stages)
        self.frozen_stages = frozenset(
            stage
            for stage in range(1, output_stage + 1)
            if stage not in self.trainable_stages
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_features = self.STAGE_CHANNELS[output_stage - 1]

        for stage_index, stage_module in enumerate(self.stages, start=1):
            requires_grad = stage_index in self.trainable_stages
            for parameter in stage_module.parameters():
                parameter.requires_grad = requires_grad

    def train(self, mode: bool = True):
        '''训练时强制冻结 stage 保持 eval，避免 BatchNorm 统计量暗中更新。'''
        super().train(mode)
        if mode:
            for stage_index in self.frozen_stages:
                self.stages[stage_index - 1].eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
