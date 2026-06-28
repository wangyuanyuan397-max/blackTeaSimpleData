"""Register the simplified framework's models, heads, losses, and backbones."""

from .classifier import ImageClassifier
from . import backbones, heads, losses

__all__ = ["ImageClassifier", "backbones", "heads", "losses"]
