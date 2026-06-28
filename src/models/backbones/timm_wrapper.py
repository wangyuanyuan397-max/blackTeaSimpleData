"""Minimal timm backbone wrapper."""

import timm
import torch
import torch.nn as nn

from ...utils.registry import BACKBONES


@BACKBONES.register("timm")
class TimmBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, **kwargs):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        self.out_features = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
