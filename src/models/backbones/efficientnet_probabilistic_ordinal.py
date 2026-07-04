'''EfficientNetV2-S 的 Beta/Kumaraswamy/Logistic-Normal 概率有序头。'''

import math
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


class BetaOrdinalHead(nn.Module):
    '''预测 Beta 参数，并用 midpoint grid 近似四个阶段区间概率。'''

    def __init__(
        self,
        in_features: int = 1280,
        num_grid: int = 200,
        min_concentration: float = 1.0,
        max_concentration: float = 80.0,
    ):
        super().__init__()
        if num_grid < 4 or num_grid % 4 != 0:
            raise ValueError('num_grid 必须是不小于 4 且能被 4 整除的整数。')
        self.fc = nn.Linear(in_features, 2)
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        grid = torch.linspace(
            0.5 / num_grid,
            1.0 - 0.5 / num_grid,
            steps=num_grid,
        )
        self.register_buffer('grid', grid)
        masks = torch.stack(
            [
                (grid < 0.25),
                ((grid >= 0.25) & (grid < 0.50)),
                ((grid >= 0.50) & (grid < 0.75)),
                (grid >= 0.75),
            ],
            dim=0,
        ).float()
        self.register_buffer('interval_masks', masks)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.fc(features)
        alpha = torch.clamp(
            F.softplus(raw[:, 0]) + self.min_concentration,
            max=self.max_concentration,
        )
        beta = torch.clamp(
            F.softplus(raw[:, 1]) + self.min_concentration,
            max=self.max_concentration,
        )
        x = self.grid.unsqueeze(0)
        alpha_column = alpha.unsqueeze(1)
        beta_column = beta.unsqueeze(1)
        log_pdf = (
            (alpha_column - 1.0) * torch.log(x)
            + (beta_column - 1.0) * torch.log1p(-x)
            - (
                torch.lgamma(alpha_column)
                + torch.lgamma(beta_column)
                - torch.lgamma(alpha_column + beta_column)
            )
        )
        # 每个样本减去同一个常数不改变最终区间质量归一化，但能避免 exp 溢出。
        pdf = torch.exp(log_pdf - log_pdf.max(dim=1, keepdim=True).values)
        stage_masses = pdf @ self.interval_masks.transpose(0, 1)
        stage_probs = stage_masses.clamp_min(1e-12)
        stage_probs = stage_probs / stage_probs.sum(dim=1, keepdim=True)
        stage_probs = stage_probs.clamp_min(1e-8)
        stage_probs = stage_probs / stage_probs.sum(dim=1, keepdim=True)
        concentration = alpha + beta
        mean = alpha / concentration
        variance = (alpha * beta) / (
            concentration.square() * (concentration + 1.0)
        )
        return {
            'stage_probs': stage_probs,
            'alpha': alpha,
            'beta': beta,
            'mean': mean,
            'var': variance,
        }


class KumaraswamyOrdinalHead(nn.Module):
    '''使用闭式 Kumaraswamy CDF 计算四个有序区间概率。'''

    def __init__(
        self,
        in_features: int = 1280,
        min_concentration: float = 1.0,
        max_concentration: float = 80.0,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        self.register_buffer(
            'internal_bounds',
            torch.tensor([0.25, 0.50, 0.75], dtype=torch.float32),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.fc(features)
        a = torch.clamp(
            F.softplus(raw[:, 0]) + self.min_concentration,
            max=self.max_concentration,
        )
        b = torch.clamp(
            F.softplus(raw[:, 1]) + self.min_concentration,
            max=self.max_concentration,
        )
        x = self.internal_bounds.unsqueeze(0)
        cdf_internal = 1.0 - torch.pow(
            1.0 - torch.pow(x, a.unsqueeze(1)),
            b.unsqueeze(1),
        )
        cdf = torch.cat(
            [
                torch.zeros(features.shape[0], 1, device=features.device),
                cdf_internal,
                torch.ones(features.shape[0], 1, device=features.device),
            ],
            dim=1,
        )
        stage_probs = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(1e-8)
        stage_probs = stage_probs / stage_probs.sum(dim=1, keepdim=True)
        return {'stage_probs': stage_probs, 'a': a, 'b': b}


class LogisticNormalOrdinalHead(nn.Module):
    '''通过 Logistic-Normal 的闭式区间 CDF 计算四类概率。'''

    def __init__(
        self,
        in_features: int = 1280,
        min_sigma: float = 0.05,
        max_sigma: float = 5.0,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.register_buffer(
            'internal_bounds',
            torch.tensor([0.25, 0.50, 0.75], dtype=torch.float32),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.fc(features)
        mu = raw[:, 0]
        sigma = torch.clamp(
            F.softplus(raw[:, 1]) + self.min_sigma,
            max=self.max_sigma,
        )
        bounds = self.internal_bounds
        logit_bounds = torch.log(bounds / (1.0 - bounds)).unsqueeze(0)
        standardized = (logit_bounds - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        cdf_internal = 0.5 * (
            1.0 + torch.erf(standardized / math.sqrt(2.0))
        )
        cdf = torch.cat(
            [
                torch.zeros(features.shape[0], 1, device=features.device),
                cdf_internal,
                torch.ones(features.shape[0], 1, device=features.device),
            ],
            dim=1,
        )
        stage_probs = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(1e-8)
        stage_probs = stage_probs / stage_probs.sum(dim=1, keepdim=True)
        return {
            'stage_probs': stage_probs,
            'mu': mu,
            'sigma': sigma,
            'mean_proxy': torch.sigmoid(mu),
        }


@BACKBONES.register('efficientnet_v2_s_probabilistic_ordinal')
class EfficientNetV2SProbabilisticOrdinalBackbone(nn.Module):
    '''统一实现纯概率分类头和 CE + 概率辅助头。'''

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        distribution: str = 'beta_cdf',
        use_ce_head: bool = False,
        final_dropout: float = 0.2,
        num_grid: int = 200,
        min_concentration: float = 1.0,
        max_concentration: float = 80.0,
        min_sigma: float = 0.05,
        max_sigma: float = 5.0,
        output_stage: int = 6,
    ):
        super().__init__()
        if num_classes != 4:
            raise ValueError('概率有序头固定要求四个阶段。')
        distribution = str(distribution).lower()
        if distribution not in {
            'beta_cdf',
            'beta_nll',
            'kumaraswamy',
            'logistic_normal',
        }:
            raise ValueError(f'未知 distribution：{distribution!r}')
        output_stage = int(output_stage)
        if output_stage < 1 or output_stage > 6:
            raise ValueError("output_stage 只允许 1～6。")
        efficientnet = _build_efficientnet_v2_s(pretrained)
        raw_features = efficientnet.features
        all_stages = [
            nn.Sequential(raw_features[0], raw_features[1]),
            raw_features[2],
            raw_features[3],
            raw_features[4],
            raw_features[5],
            nn.Sequential(raw_features[6], raw_features[7]),
        ]
        stage_channels = (24, 48, 64, 128, 160, 1280)
        self.features = nn.Sequential(*all_stages[:output_stage])
        self.output_stage = output_stage
        feature_channels = stage_channels[output_stage - 1]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.distribution = distribution
        self.use_ce_head = bool(use_ce_head)
        self.is_probabilistic_ordinal = True
        self.classification_dropout = (
            nn.Dropout(final_dropout) if final_dropout > 0 else nn.Identity()
        )
        # CE 辅助实验先创建相同分类器，保证不同概率头下初始分类权重一致。
        self.classifier = (
            nn.Linear(feature_channels, num_classes) if self.use_ce_head else None
        )
        if distribution in {'beta_cdf', 'beta_nll'}:
            self.probabilistic_head = BetaOrdinalHead(
                in_features=feature_channels,
                num_grid=num_grid,
                min_concentration=min_concentration,
                max_concentration=max_concentration,
            )
        elif distribution == 'kumaraswamy':
            self.probabilistic_head = KumaraswamyOrdinalHead(
                in_features=feature_channels,
                min_concentration=min_concentration,
                max_concentration=max_concentration,
            )
        else:
            self.probabilistic_head = LogisticNormalOrdinalHead(
                in_features=feature_channels,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
            )
        self.register_buffer(
            'stage_centers',
            torch.tensor([0.125, 0.375, 0.625, 0.875]),
        )
        self.out_features = num_classes

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        auxiliary: Dict[str, Any] = self.probabilistic_head(features)
        auxiliary['distribution'] = self.distribution
        auxiliary['feature'] = features
        if self.use_ce_head:
            primary_logits = self.classifier(
                self.classification_dropout(features)
            )
        elif self.distribution == 'beta_nll':
            # 最近中心的决策边界正好是 0.25/0.50/0.75。
            primary_logits = -torch.abs(
                auxiliary['mean'].unsqueeze(1)
                - self.stage_centers.unsqueeze(0)
            )
        else:
            primary_logits = torch.log(
                auxiliary['stage_probs'].clamp_min(1e-8)
            )
        return primary_logits, auxiliary
