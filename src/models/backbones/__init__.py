"""Backbone registrations kept by the simplified framework."""

from .resnet import ResNet50
from .safnet import SAFNetBackbone
from .torchvision_wrapper import TorchvisionBackbone
from .efficientnet_ablation import EfficientNetV2SAblationBackbone
from .efficientnet_gated import (
    EfficientNetV2SGatedRefinementBackbone,
    EfficientNetV2SMultiStageGatedFusionBackbone,
)

try:
    from .mambaout import MambaOutTinyCEBackbone
except ModuleNotFoundError as exc:
    if exc.name != 'timm':
        raise
    MambaOutTinyCEBackbone = None

try:
    from .timm_wrapper import TimmBackbone
except ModuleNotFoundError as exc:
    if exc.name != 'timm':
        raise
    TimmBackbone = None

__all__ = [
    'EfficientNetV2SAblationBackbone',
    'EfficientNetV2SGatedRefinementBackbone',
    'EfficientNetV2SMultiStageGatedFusionBackbone',
    "MambaOutTinyCEBackbone",
    "ResNet50",
    "SAFNetBackbone",
    "TimmBackbone",
    "TorchvisionBackbone",
]
