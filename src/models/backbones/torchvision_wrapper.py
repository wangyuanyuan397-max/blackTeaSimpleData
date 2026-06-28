"""Minimal torchvision backbone wrapper."""

import torch
import torch.nn as nn
import torchvision.models as models

from ...utils.registry import BACKBONES


@BACKBONES.register("torchvision")
class TorchvisionBackbone(nn.Module):
    """Create a torchvision model and expose its pre-classifier features."""

    def __init__(self, model_name: str, pretrained: bool = True, **kwargs):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        try:
            self.model = models.get_model(model_name, weights=weights, **kwargs)
        except (TypeError, ValueError):
            self.model = models.get_model(model_name, pretrained=pretrained, **kwargs)
        self.out_features = self._remove_classifier()

    def _remove_classifier(self) -> int:
        if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
            out_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
            return out_features

        if hasattr(self.model, "classifier"):
            classifier = self.model.classifier
            if isinstance(classifier, nn.Linear):
                out_features = classifier.in_features
                self.model.classifier = nn.Identity()
                return out_features
            if isinstance(classifier, nn.Sequential):
                for index in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[index], nn.Linear):
                        out_features = classifier[index].in_features
                        classifier[index] = nn.Identity()
                        return out_features

        if hasattr(self.model, "heads"):
            heads = self.model.heads
            if hasattr(heads, "head") and isinstance(heads.head, nn.Linear):
                out_features = heads.head.in_features
                heads.head = nn.Identity()
                return out_features

        if hasattr(self.model, "head") and isinstance(self.model.head, nn.Linear):
            out_features = self.model.head.in_features
            self.model.head = nn.Identity()
            return out_features

        raise ValueError(f"Cannot locate classifier head for {type(self.model).__name__}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if getattr(output, "logits", None) is not None:
            output = output.logits
        if output.ndim > 2:
            output = torch.nn.functional.adaptive_avg_pool2d(output, 1).flatten(1)
        return output
