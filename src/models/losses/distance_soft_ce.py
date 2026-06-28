import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import LOSSES


@LOSSES.register("distance_soft_ce")
class DistanceAwareSoftCrossEntropyLoss(nn.Module):
    """
    Cross-entropy with distance-aware soft targets.

    Supported target modes:
    - soft_targets: explicit [num_classes, num_classes] matrix
    - distance_weights: per-distance weights, normalized row-wise
    """

    def __init__(self, num_classes, soft_targets=None, distance_weights=None, reduction="mean"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.reduction = str(reduction).lower()
        self.register_buffer(
            "target_matrix",
            self._build_target_matrix(
                num_classes=self.num_classes,
                soft_targets=soft_targets,
                distance_weights=distance_weights,
            ),
        )

    @staticmethod
    def _build_target_matrix(num_classes, soft_targets=None, distance_weights=None):
        if soft_targets is not None:
            matrix = torch.tensor(soft_targets, dtype=torch.float32)
            if matrix.shape != (num_classes, num_classes):
                raise ValueError(
                    f"soft_targets must have shape [{num_classes}, {num_classes}], got {tuple(matrix.shape)}"
                )
        elif distance_weights is not None:
            weights = [float(x) for x in distance_weights]
            if len(weights) < num_classes:
                weights = weights + [0.0] * (num_classes - len(weights))

            rows = []
            for target_idx in range(num_classes):
                rows.append([weights[abs(target_idx - pred_idx)] for pred_idx in range(num_classes)])
            matrix = torch.tensor(rows, dtype=torch.float32)
        else:
            raise ValueError("distance_soft_ce requires either soft_targets or distance_weights.")

        if torch.any(matrix < 0):
            raise ValueError("soft target weights must be non-negative.")

        row_sums = matrix.sum(dim=1, keepdim=True)
        if torch.any(row_sums <= 0):
            raise ValueError("each soft target row must sum to a positive value.")

        return matrix / row_sums

    def forward(self, preds, targets):
        logits = preds[0] if isinstance(preds, tuple) else preds
        log_probs = F.log_softmax(logits, dim=1)
        soft_targets = self.target_matrix[targets]
        loss = -(soft_targets * log_probs).sum(dim=1)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
