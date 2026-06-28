"""Core ResNet backbone."""

import torch
import torch.nn as nn
from torchvision import models

from ...utils.registry import BACKBONES


@BACKBONES.register("resnet50")
class ResNet50(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = models.resnet50(weights=weights)
        self.out_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
