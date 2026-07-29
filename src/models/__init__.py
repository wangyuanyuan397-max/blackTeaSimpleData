"""Register the simplified framework's models, heads, losses, and backbones."""

from .classifier import ImageClassifier
from .supcon_margin_classifier import SupConMarginClassifier
from . import backbones, heads, losses

__all__ = ["ImageClassifier", "SupConMarginClassifier", "backbones", "heads", "losses"]
