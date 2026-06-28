import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import LOSSES
from .distance_soft_ce import DistanceAwareSoftCrossEntropyLoss


@LOSSES.register("distance_soft_ce_with_aux_regression")
class DistanceSoftCEWithAuxRegressionLoss(nn.Module):
    """Distance-aware soft-label classification loss + auxiliary continuous regression."""

    def __init__(
        self,
        num_classes,
        soft_targets=None,
        distance_weights=None,
        aux_weight=1.0,
        regression_target="sfi",
        regression_loss="smooth_l1",
    ):
        super().__init__()
        self.classification_loss = DistanceAwareSoftCrossEntropyLoss(
            num_classes=num_classes,
            soft_targets=soft_targets,
            distance_weights=distance_weights,
            reduction="mean",
        )
        self.aux_weight = float(aux_weight)
        self.regression_target = str(regression_target)
        self.regression_loss = str(regression_loss).lower()

    @staticmethod
    def _split_outputs(outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            return outputs[0], outputs[1]
        return outputs, None

    def _compute_regression_loss(self, predictions, targets):
        if self.regression_loss == "mse":
            return F.mse_loss(predictions, targets)
        return F.smooth_l1_loss(predictions, targets)

    def forward(self, outputs, targets, extra_targets=None):
        logits, regression = self._split_outputs(outputs)
        total_loss = self.classification_loss(logits, targets)

        if regression is None or extra_targets is None or self.regression_target not in extra_targets:
            return total_loss

        reg_targets = extra_targets[self.regression_target].float().to(logits.device).view(-1)
        reg_predictions = regression.float().to(logits.device).view(-1)
        reg_loss = self._compute_regression_loss(reg_predictions, reg_targets)
        return total_loss + self.aux_weight * reg_loss

    def get_aux_predictions(self, outputs):
        _, regression = self._split_outputs(outputs)
        if regression is None:
            return None
        return regression.view(-1)

    def compute_aux_metrics(self, outputs, extra_targets=None):
        if extra_targets is None or self.regression_target not in extra_targets:
            return {}

        reg_predictions = self.get_aux_predictions(outputs)
        if reg_predictions is None:
            return {}

        reg_targets = extra_targets[self.regression_target].float().to(reg_predictions.device).view(-1)
        mae = torch.abs(reg_predictions - reg_targets).mean().item()
        return {f"{self.regression_target}_mae": float(mae)}
