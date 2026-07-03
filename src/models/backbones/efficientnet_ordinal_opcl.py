'''EfficientNetV2-S 的 ordinal class-rank 与 OPCL 多分支实验模型。'''

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ...utils.registry import BACKBONES


def _build_efficientnet_v2_s(pretrained: bool) -> nn.Module:
    '''创建 EfficientNetV2-S，并兼容新旧 torchvision 权重接口。'''
    if hasattr(models, 'EfficientNet_V2_S_Weights'):
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        return models.efficientnet_v2_s(weights=weights)
    return models.efficientnet_v2_s(pretrained=pretrained)


class OrdinalProjectionHead(nn.Module):
    '''将 1280 维共享特征投影到归一化 prototype embedding 空间。'''

    def __init__(
        self,
        in_features: int = 1280,
        hidden_features: int = 256,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(features), p=2, dim=1)


@BACKBONES.register('efficientnet_v2_s_ordinal_opcl')
class EfficientNetV2SOrdinalOPCLBackbone(nn.Module):
    '''
    根据 YAML 开关创建分类、class-rank、projection 与 prototype 分支。

    forward 返回 (class_logits, auxiliary_dict)。分类 strategy 和推理始终只读取
    tuple 的第一项，因此 rank/prototype 仅作为训练期辅助约束。
    '''

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        final_dropout: float = 0.2,
        enable_rank: bool = False,
        enable_projection: bool = False,
        enable_prototypes: bool = False,
        projection_hidden_dim: int = 256,
        embedding_dim: int = 128,
        projection_dropout: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        if num_classes != 4:
            raise ValueError('当前 ordinal OPCL 实验固定要求 4 个有序阶段。')
        if enable_prototypes and not enable_projection:
            raise ValueError('启用 prototypes 时必须同时启用 projection head。')
        if temperature <= 0:
            raise ValueError('temperature 必须大于 0。')

        efficientnet = _build_efficientnet_v2_s(pretrained)
        self.features = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classification_dropout = (
            nn.Dropout(final_dropout) if final_dropout > 0 else nn.Identity()
        )

        # 先创建所有实验共享的分类器，保证相同 seed 下 V0～V11 初始权重一致。
        self.classifier = nn.Linear(1280, num_classes)
        self.enable_rank = bool(enable_rank)
        self.enable_projection = bool(enable_projection)
        self.enable_prototypes = bool(enable_prototypes)
        self.is_ordinal_opcl = True
        self.temperature = float(temperature)

        self.rank_head = nn.Linear(1280, 1) if self.enable_rank else None
        self.projection_head = None
        if self.enable_projection:
            self.projection_head = OrdinalProjectionHead(
                in_features=1280,
                hidden_features=projection_hidden_dim,
                embedding_dim=embedding_dim,
                dropout=projection_dropout,
            )

        if self.enable_prototypes:
            self.prototypes = nn.Parameter(torch.empty(num_classes, embedding_dim))
            nn.init.xavier_uniform_(self.prototypes)
        else:
            self.register_parameter('prototypes', None)

        # 外层使用 identity head；这里已经返回 4 类 logits 和辅助字典。
        self.out_features = num_classes

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        class_logits = self.classifier(self.classification_dropout(features))
        auxiliary: Dict[str, Any] = {'feature': features}

        if self.rank_head is not None:
            auxiliary['rank_pred'] = self.rank_head(features).squeeze(1)

        if self.projection_head is not None:
            embedding = self.projection_head(features)
            auxiliary['embedding'] = embedding
            if self.prototypes is not None:
                normalized_prototypes = F.normalize(self.prototypes, p=2, dim=1)
                auxiliary['prototypes'] = normalized_prototypes
                auxiliary['proto_logits'] = (
                    embedding @ normalized_prototypes.transpose(0, 1)
                ) / self.temperature

        return class_logits, auxiliary
