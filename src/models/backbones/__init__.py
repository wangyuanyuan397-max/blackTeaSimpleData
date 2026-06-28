"""Backbone registrations kept by the simplified framework."""

from .resnet import ResNet50
from .safnet import SAFNetBackbone
from .torchvision_wrapper import TorchvisionBackbone

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
    "MambaOutTinyCEBackbone",
    "ResNet50",
    "SAFNetBackbone",
    "TimmBackbone",
    "TorchvisionBackbone",
]
