'''Beta/Kumaraswamy/Logistic-Normal 概率有序实验的组合损失。'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import LOSSES


@LOSSES.register('probabilistic_ordinal')
class ProbabilisticOrdinalLoss(nn.Module):
    '''支持纯概率损失以及 CE + 概率辅助损失。'''

    def __init__(
        self,
        mode: str,
        use_ce: bool = False,
        auxiliary_weight: float = 1.0,
        beta_reg_weight: float = 0.0001,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.use_ce = bool(use_ce)
        self.auxiliary_weight = float(auxiliary_weight)
        self.beta_reg_weight = float(beta_reg_weight)
        if self.mode not in {
            'beta_cdf',
            'beta_nll',
            'kumaraswamy_cdf',
            'logistic_normal_cdf',
        }:
            raise ValueError(f'未知 probabilistic loss mode：{self.mode!r}')
        self.register_buffer(
            'stage_centers',
            torch.tensor([0.125, 0.375, 0.625, 0.875]),
        )

    @staticmethod
    def _split_outputs(outputs):
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            raise ValueError('概率有序模型必须返回 (primary_logits, auxiliary)。')
        return outputs[0], outputs[1]

    @staticmethod
    def _stage_probability_loss(
        stage_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return F.nll_loss(
            torch.log(stage_probs.clamp_min(1e-8)),
            targets,
        )

    def _beta_center_nll(
        self,
        auxiliary,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        alpha = auxiliary['alpha']
        beta = auxiliary['beta']
        centers = self.stage_centers[targets].clamp(1e-6, 1.0 - 1e-6)
        log_probability = (
            (alpha - 1.0) * torch.log(centers)
            + (beta - 1.0) * torch.log1p(-centers)
            - (
                torch.lgamma(alpha)
                + torch.lgamma(beta)
                - torch.lgamma(alpha + beta)
            )
        )
        nll = -log_probability.mean()
        regularization = self.beta_reg_weight * (alpha + beta).mean()
        return nll + regularization

    def forward(self, outputs, targets, extra_targets=None):
        primary_logits, auxiliary = self._split_outputs(outputs)
        if self.mode == 'beta_nll':
            probabilistic_loss = self._beta_center_nll(auxiliary, targets)
        else:
            probabilistic_loss = self._stage_probability_loss(
                auxiliary['stage_probs'],
                targets,
            )
        if not self.use_ce:
            return probabilistic_loss
        classification_loss = F.cross_entropy(primary_logits, targets)
        return (
            classification_loss
            + self.auxiliary_weight * probabilistic_loss
        )
