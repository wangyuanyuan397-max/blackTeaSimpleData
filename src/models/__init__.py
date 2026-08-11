"""Register the simplified framework's models, heads, losses, and backbones."""

from .classifier import ImageClassifier
from .global_local_reliability import GlobalLocalReliabilityClassifier
from .supcon_margin_classifier import SupConMarginClassifier
from . import backbones, heads, losses

__all__ = [
    "ImageClassifier",
    "GlobalLocalReliabilityClassifier",
    "SupConMarginClassifier",
    "backbones",
    "heads",
    "losses",
]
