'''Ordinal class-rank、soft label、prototype 与 OPCL 的统一组合损失。'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import LOSSES


@LOSSES.register('ordinal_opcl')
class OrdinalOPCLLoss(nn.Module):
    '''通过配置开关覆盖 V0～V11 的全部损失组合。'''

    DEFAULT_SOFT_TARGETS = (
        (0.93, 0.07, 0.00, 0.00),
        (0.07, 0.86, 0.07, 0.00),
        (0.00, 0.07, 0.86, 0.07),
        (0.00, 0.00, 0.07, 0.93),
    )

    def __init__(
        self,
        num_classes: int = 4,
        classification_mode: str = 'ce',
        use_rank: bool = False,
        prototype_mode: str = 'none',
        lambda_rank: float = 1.0,
        alpha_proto: float = 0.1,
        gamma: float = 0.1,
        soft_targets=None,
    ):
        super().__init__()
        if num_classes != 4:
            raise ValueError('ordinal_opcl loss 固定要求 4 个有序阶段。')
        self.num_classes = int(num_classes)
        self.classification_mode = str(classification_mode).lower()
        self.use_rank = bool(use_rank)
        self.prototype_mode = str(prototype_mode).lower()
        self.lambda_rank = float(lambda_rank)
        self.alpha_proto = float(alpha_proto)
        self.gamma = float(gamma)
        if self.classification_mode not in {'ce', 'softlabel'}:
            raise ValueError('classification_mode 必须是 ce 或 softlabel。')
        if self.prototype_mode not in {'none', 'hard', 'ordinal_soft', 'opcl'}:
            raise ValueError(
                'prototype_mode 必须是 none、hard、ordinal_soft 或 opcl。'
            )
        target_matrix = torch.tensor(
            soft_targets or self.DEFAULT_SOFT_TARGETS,
            dtype=torch.float32,
        )
        if target_matrix.shape != (num_classes, num_classes):
            raise ValueError('soft_targets 必须是 4×4 矩阵。')
        target_matrix = target_matrix / target_matrix.sum(dim=1, keepdim=True)
        self.register_buffer('soft_target_matrix', target_matrix)
        class_indices = torch.arange(num_classes, dtype=torch.float32)
        self.register_buffer(
            'class_distance_matrix',
            torch.abs(class_indices[:, None] - class_indices[None, :]),
        )

    @staticmethod
    def _split_outputs(outputs):
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            return outputs, {}
        logits, auxiliary = outputs[0], outputs[1]
        return logits, auxiliary if isinstance(auxiliary, dict) else {}

    @staticmethod
    def _soft_cross_entropy(
        logits: torch.Tensor,
        soft_targets: torch.Tensor,
    ) -> torch.Tensor:
        return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def forward(self, outputs, targets, extra_targets=None):
        class_logits, auxiliary = self._split_outputs(outputs)
        if self.classification_mode == 'softlabel':
            classification_loss = self._soft_cross_entropy(
                class_logits,
                self.soft_target_matrix[targets],
            )
        else:
            classification_loss = F.cross_entropy(class_logits, targets)
        total_loss = classification_loss

        if self.use_rank:
            rank_prediction = auxiliary.get('rank_pred')
            if rank_prediction is None:
                raise ValueError('当前 loss 启用了 rank，但模型没有返回 rank_pred。')
            rank_target = targets.float() / float(self.num_classes - 1)
            rank_loss = F.smooth_l1_loss(
                rank_prediction.view(-1),
                rank_target.view(-1),
            )
            total_loss = total_loss + self.lambda_rank * rank_loss

        if self.prototype_mode != 'none':
            proto_logits = auxiliary.get('proto_logits')
            if proto_logits is None:
                raise ValueError('当前 loss 启用了 prototype，但模型没有返回 proto_logits。')
            if self.prototype_mode == 'hard':
                prototype_loss = F.cross_entropy(proto_logits, targets)
            else:
                prototype_loss = self._soft_cross_entropy(
                    proto_logits,
                    self.soft_target_matrix[targets],
                )
                if self.prototype_mode == 'opcl':
                    proto_probabilities = torch.softmax(proto_logits, dim=1)
                    distance_penalty = (
                        proto_probabilities * self.class_distance_matrix[targets]
                    ).sum(dim=1).mean()
                    prototype_loss = prototype_loss + self.gamma * distance_penalty
            total_loss = total_loss + self.alpha_proto * prototype_loss
        return total_loss
