from .ce_loss import CrossEntropyLoss
from .ce_with_aux_regression import CEWithAuxRegressionLoss
from .distance_soft_ce import DistanceAwareSoftCrossEntropyLoss
from .distance_soft_ce_with_aux_regression import DistanceSoftCEWithAuxRegressionLoss

__all__ = [
    "CrossEntropyLoss",
    "CEWithAuxRegressionLoss",
    "DistanceAwareSoftCrossEntropyLoss",
    "DistanceSoftCEWithAuxRegressionLoss",
]
