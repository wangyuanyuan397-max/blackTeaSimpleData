"""SupCon + cosine-margin classifier for brute-force tea-stage experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.registry import BACKBONES, MODELS


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return parsed


class ProjectionHead(nn.Module):
    """把 backbone 特征压到较低维度，专门供 SupCon 或 margin 分类使用。"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int = 128,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = _positive_int(hidden_features or in_features, "hidden_features")
        out = _positive_int(out_features, "out_features")
        self.out_features = out
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(drop_rate)) if float(drop_rate) > 0 else nn.Identity(),
            nn.Linear(hidden, out),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CosineMarginHead(nn.Module):
    """用归一化特征和类别权重的余弦相似度做分类。"""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = _positive_int(in_features, "in_features")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.scale = float(scale)
        self.margin = float(margin)
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {scale!r}.")
        if self.margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin!r}.")
        self.weight = nn.Parameter(torch.empty(self.num_classes, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def cosine_logits(self, features: torch.Tensor) -> torch.Tensor:
        # 只比较方向，降低颜色亮度或整体强弱对分类边界的影响。
        return F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.cosine_logits(features) * self.scale


@MODELS.register("supcon_margin_classifier")
class SupConMarginClassifier(nn.Module):
    """MixNet-S 等 backbone + projector + cosine-margin head 的实验模型。"""

    def __init__(
        self,
        backbone: dict[str, Any] | nn.Module,
        head: dict[str, Any],
        projector: dict[str, Any] | None = None,
        classifier_feature: str = "projected",
        return_embeddings: bool = False,
    ) -> None:
        super().__init__()
        # ComponentBuilder 会按通用 ImageClassifier 配置注入 return_embeddings；
        # SupCon/Margin 的训练辅助张量由模型内部按训练态返回，这里只做兼容接收。
        self.return_embeddings = bool(return_embeddings)
        self.backbone = self._build_backbone(backbone)
        backbone_features = int(getattr(self.backbone, "out_features"))

        projector_cfg = dict(projector or {})
        projector_cfg.pop("type", None)
        self.projector = ProjectionHead(backbone_features, **projector_cfg)

        self.classifier_feature = str(classifier_feature).lower()
        if self.classifier_feature not in {"projected", "raw"}:
            raise ValueError("classifier_feature must be 'projected' or 'raw'.")

        head_cfg = dict(head)
        head_cfg.pop("type", None)
        # HeadConfig 会自动带 drop_rate；margin head 不用 dropout，这里显式过滤。
        head_cfg.pop("drop_rate", None)
        num_classes = head_cfg.pop("num_classes", None)
        if num_classes is None:
            raise ValueError("supcon_margin_classifier requires head.num_classes.")
        classifier_in = (
            self.projector.out_features
            if self.classifier_feature == "projected"
            else backbone_features
        )
        self.head = CosineMarginHead(
            in_features=classifier_in,
            num_classes=int(num_classes),
            **head_cfg,
        )
        self.out_features = int(classifier_in)

    @staticmethod
    def _build_backbone(backbone: dict[str, Any] | nn.Module) -> nn.Module:
        if not isinstance(backbone, dict):
            if not hasattr(backbone, "out_features"):
                raise ValueError("backbone module must expose out_features.")
            return backbone

        cfg = dict(backbone)
        backbone_type = cfg.pop("type")
        backbone_cls = BACKBONES.get(backbone_type)
        if backbone_cls is None:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        built = backbone_cls(**cfg)
        if not hasattr(built, "out_features"):
            raise ValueError(f"Backbone {backbone_type} must expose out_features.")
        return built

    def _forward_single_view(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_features = self.backbone(images)
        projected_features = self.projector(raw_features)
        classifier_features = (
            projected_features
            if self.classifier_feature == "projected"
            else raw_features
        )
        cosine_logits = self.head.cosine_logits(classifier_features)
        logits = cosine_logits * self.head.scale
        return logits, cosine_logits, raw_features, projected_features

    def forward(self, images: torch.Tensor):
        if images.ndim == 5:
            batch_size, num_views, channels, height, width = images.shape
            # 训练双视图输入：[B, 2, C, H, W] -> [B*2, C, H, W] 共享 backbone 前向。
            flat_images = images.reshape(batch_size * num_views, channels, height, width)
            logits, cosine_logits, raw_features, projected_features = self._forward_single_view(flat_images)
            logits_for_metrics = logits.view(batch_size, num_views, -1).mean(dim=1)

            if self.training and torch.is_grad_enabled():
                # loss 需要两视图的投影特征和未加 margin 的余弦 logits。
                auxiliary = {
                    "view_cosine_logits": cosine_logits,
                    "view_projected_features": projected_features,
                    "view_raw_features": raw_features,
                    "num_views": torch.tensor(num_views, device=logits.device),
                    "margin": torch.tensor(self.head.margin, device=logits.device, dtype=logits.dtype),
                    "scale": torch.tensor(self.head.scale, device=logits.device, dtype=logits.dtype),
                }
                return logits_for_metrics, auxiliary
            return logits_for_metrics

        if images.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] or [B,V,C,H,W], got {tuple(images.shape)}.")
        logits, _, _, _ = self._forward_single_view(images)
        return logits
