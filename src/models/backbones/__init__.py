"""Backbone registrations kept by the simplified framework."""

from .resnet import ResNet50
from .safnet import SAFNetBackbone
from .torchvision_wrapper import TorchvisionBackbone
from .efficientnet_ablation import EfficientNetV2SAblationBackbone
from .efficientnet_stage_attention import EfficientNetV2SStageAttentionBackbone
from .efficientnet_multiscale_fusion import EfficientNetV2SMultiScaleFusionBackbone
from .efficientnet_ordinal_opcl import EfficientNetV2SOrdinalOPCLBackbone
from .efficientnet_stage_probe import EfficientNetV2SStageProbeBackbone
from .efficientnet_probabilistic_ordinal import (
    EfficientNetV2SProbabilisticOrdinalBackbone,
)
from .efficientnet_final_refinement import EfficientNetV2SFinalRefinementBackbone
from .efficientnet_region_attention import EfficientNetV2SRegionAttentionBackbone
from .efficientnet_global_local_multiscale import (
    EfficientNetV2SGlobalLocalMultiScaleBackbone,
)
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
    'EfficientNetV2SStageAttentionBackbone',
    'EfficientNetV2SMultiScaleFusionBackbone',
    'EfficientNetV2SOrdinalOPCLBackbone',
    'EfficientNetV2SStageProbeBackbone',
    'EfficientNetV2SProbabilisticOrdinalBackbone',
    'EfficientNetV2SFinalRefinementBackbone',
    'EfficientNetV2SGlobalLocalMultiScaleBackbone',
    'EfficientNetV2SGatedRefinementBackbone',
    'EfficientNetV2SMultiStageGatedFusionBackbone',
    "MambaOutTinyCEBackbone",
    "ResNet50",
    "SAFNetBackbone",
    "TimmBackbone",
    "TorchvisionBackbone",
]
