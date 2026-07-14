"""Minimal timm backbone wrapper."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import BACKBONES


@BACKBONES.register("timm")
class TimmBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, input_size: int = 224, **kwargs):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        self.out_features = self._infer_out_features(input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._extract_feature_vector(x)

    def _extract_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features") and hasattr(self.model, "forward_head"):
            features = self.model.forward_features(x)
            try:
                features = self.model.forward_head(features, pre_logits=True)
            except TypeError:
                features = self.model.forward_head(features)
        else:
            features = self.model(x)
        return self._to_feature_vector(features)

    def _to_feature_vector(self, features) -> torch.Tensor:
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.ndim == 2:
            return features
        if features.ndim == 4:
            num_features = getattr(self.model, "num_features", None)
            if num_features is not None and features.shape[-1] == num_features:
                return features.mean(dim=(1, 2))
            return torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
        if features.ndim == 3:
            return features.mean(dim=1)
        return torch.flatten(features, 1)

    def _infer_out_features(self, input_size: int) -> int:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, input_size, input_size)
                features = self._extract_feature_vector(dummy)
        finally:
            self.model.train(was_training)
        if features.ndim != 2 or features.shape[1] <= 0:
            raise ValueError(
                f"timm backbone must return a non-empty 2D feature tensor, got {tuple(features.shape)}"
            )
        return int(features.shape[1])
