"""Global-local reliability fusion classifier with a shared image backbone."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn

from ..utils.registry import BACKBONES, HEADS, MODELS


@MODELS.register("glrf_classifier")
class GlobalLocalReliabilityClassifier(nn.Module):
    """Fuse one full-image feature with multiple crop features.

    The backbone and classifier are constructed before any optional fusion
    modules. With the same random seed, the ``global_only`` variant therefore
    starts from the same pretrained backbone and classifier initialization as
    the ordinary ``classifier`` baseline.
    """

    def __init__(
        self,
        backbone: dict[str, Any] | nn.Module,
        head: dict[str, Any] | nn.Module,
        projection_dim: int = 128,
        local_fusion: str = "reliability",
        global_local_fusion: str = "gate",
        initial_global_weight: float = 0.8,
        initial_local_weight: float = 0.2,
        num_local_views: int = 2,
        global_image_size: int = 408,
        local_image_size: int = 224,
        return_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if return_embeddings:
            raise ValueError("GLRF currently returns final classification logits only.")
        self.local_fusion = str(local_fusion).lower()
        self.global_local_fusion = str(global_local_fusion).lower()
        self.projection_dim = int(projection_dim)
        self.num_local_views = int(num_local_views)
        self.global_image_size = int(global_image_size)
        self.local_image_size = int(local_image_size)
        if self.local_fusion not in {"mean", "reliability"}:
            raise ValueError("local_fusion must be 'mean' or 'reliability'.")
        if self.global_local_fusion not in {"global_only", "mean", "gate"}:
            raise ValueError(
                "global_local_fusion must be 'global_only', 'mean', or 'gate'."
            )
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")
        if self.num_local_views < 0:
            raise ValueError("num_local_views must be non-negative.")
        if self.global_local_fusion != "global_only" and self.num_local_views == 0:
            raise ValueError("A local fusion variant requires num_local_views > 0.")

        if isinstance(backbone, dict):
            backbone_config = copy.deepcopy(backbone)
            backbone_type = backbone_config.pop("type")
            self.backbone = BACKBONES.get(backbone_type)(**backbone_config)
        else:
            self.backbone = backbone
        if not hasattr(self.backbone, "out_features"):
            raise ValueError("GLRF backbone must expose out_features.")
        feature_dim = int(self.backbone.out_features)

        # Keep classifier construction aligned with the ordinary baseline.
        if isinstance(head, dict):
            head_config = copy.deepcopy(head)
            head_type = head_config.pop("type")
            head_config["in_features"] = feature_dim
            self.head = HEADS.get(head_type)(**head_config)
        else:
            self.head = head

        self.global_proj: nn.Module | None = None
        self.local_proj: nn.Module | None = None
        self.local_gate: nn.Module | None = None
        self.gl_gate: nn.Module | None = None

        needs_projection = (
            self.local_fusion == "reliability"
            or self.global_local_fusion == "gate"
        ) and self.global_local_fusion != "global_only"
        if needs_projection:
            self.global_proj = nn.Sequential(
                nn.Linear(feature_dim, self.projection_dim),
                nn.LayerNorm(self.projection_dim),
            )
            self.local_proj = nn.Sequential(
                nn.Linear(feature_dim, self.projection_dim),
                nn.LayerNorm(self.projection_dim),
            )

        if self.local_fusion == "reliability" and self.global_local_fusion != "global_only":
            self.local_gate = nn.Sequential(
                nn.Linear(self.projection_dim * 3, self.projection_dim),
                nn.SiLU(),
                nn.Linear(self.projection_dim, 1),
            )
            nn.init.zeros_(self.local_gate[-1].weight)
            nn.init.zeros_(self.local_gate[-1].bias)

        if self.global_local_fusion == "gate":
            if initial_global_weight <= 0 or initial_local_weight <= 0:
                raise ValueError("Initial global/local weights must both be positive.")
            self.gl_gate = nn.Sequential(
                nn.Linear(self.projection_dim * 4, self.projection_dim),
                nn.SiLU(),
                nn.Linear(self.projection_dim, 2),
            )
            nn.init.zeros_(self.gl_gate[-1].weight)
            logit_ratio = math.log(initial_global_weight / initial_local_weight)
            with torch.no_grad():
                self.gl_gate[-1].bias.copy_(
                    torch.tensor([logit_ratio, 0.0], dtype=self.gl_gate[-1].bias.dtype)
                )

        self._last_fusion_diagnostics: dict[str, torch.Tensor | None] = {}

    @staticmethod
    def _require_inputs(inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(inputs, dict):
            raise TypeError("GLRF input must be a mapping with 'global' and 'locals' tensors.")
        global_images = inputs.get("global")
        local_images = inputs.get("locals")
        if not torch.is_tensor(global_images) or not torch.is_tensor(local_images):
            raise TypeError("GLRF 'global' and 'locals' values must be tensors.")
        if global_images.ndim != 4:
            raise ValueError(f"Expected global tensor [B,C,H,W], got {tuple(global_images.shape)}")
        if local_images.ndim != 5:
            raise ValueError(f"Expected locals tensor [B,K,C,H,W], got {tuple(local_images.shape)}")
        if global_images.size(0) != local_images.size(0):
            raise ValueError("Global and local tensors must have the same batch size.")
        return global_images, local_images

    def _store_diagnostics(
        self,
        *,
        local_weights: torch.Tensor,
        global_local_weights: torch.Tensor,
        global_logits: torch.Tensor,
        local_logits: torch.Tensor | None,
    ) -> None:
        self._last_fusion_diagnostics = {
            "local_weights": local_weights.detach(),
            "global_local_weights": global_local_weights.detach(),
            "global_logits": global_logits.detach(),
            "local_logits": None if local_logits is None else local_logits.detach(),
        }

    def get_fusion_diagnostics(self) -> dict[str, torch.Tensor | None]:
        return dict(self._last_fusion_diagnostics)

    def make_profile_input(
        self,
        batch_size: int,
        device: torch.device,
        image_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del image_size
        return {
            "global": torch.zeros(
                batch_size,
                3,
                self.global_image_size,
                self.global_image_size,
                device=device,
            ),
            "locals": torch.zeros(
                batch_size,
                self.num_local_views,
                3,
                self.local_image_size,
                self.local_image_size,
                device=device,
            ),
        }

    def profile_input_description(self) -> str:
        if self.global_local_fusion == "global_only":
            return f"one {self.global_image_size}x{self.global_image_size} global view"
        return (
            f"one {self.global_image_size}x{self.global_image_size} global view + "
            f"{self.num_local_views} {self.local_image_size}x{self.local_image_size} local views"
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        global_images, local_images = self._require_inputs(inputs)
        batch_size = global_images.size(0)
        global_features = self.backbone(global_images)

        if self.global_local_fusion == "global_only":
            logits = self.head(global_features)
            local_weights = global_features.new_empty((batch_size, 0))
            fusion_weights = global_features.new_tensor([1.0, 0.0]).expand(batch_size, -1)
            self._store_diagnostics(
                local_weights=local_weights,
                global_local_weights=fusion_weights,
                global_logits=logits,
                local_logits=None,
            )
            return logits

        if local_images.size(1) == 0:
            raise ValueError("This GLRF variant received no local views.")
        batch_size, local_count, channels, height, width = local_images.shape
        flat_locals = local_images.reshape(batch_size * local_count, channels, height, width)
        local_features = self.backbone(flat_locals).reshape(batch_size, local_count, -1)

        projected_global = None
        projected_locals = None
        if self.global_proj is not None and self.local_proj is not None:
            projected_global = self.global_proj(global_features)
            projected_locals = self.local_proj(
                local_features.reshape(batch_size * local_count, -1)
            ).reshape(batch_size, local_count, -1)

        if self.local_fusion == "reliability":
            assert projected_global is not None and projected_locals is not None
            assert self.local_gate is not None
            expanded_global = projected_global.unsqueeze(1).expand(-1, local_count, -1)
            local_gate_input = torch.cat(
                [
                    projected_locals,
                    expanded_global,
                    torch.abs(projected_locals - expanded_global),
                ],
                dim=-1,
            )
            local_scores = self.local_gate(local_gate_input).squeeze(-1)
            local_weights = torch.softmax(local_scores, dim=1)
        else:
            local_weights = global_features.new_full(
                (batch_size, local_count),
                1.0 / float(local_count),
            )
        fused_local = torch.sum(local_weights.unsqueeze(-1) * local_features, dim=1)

        if self.global_local_fusion == "gate":
            assert projected_global is not None and self.local_proj is not None
            assert self.gl_gate is not None
            projected_local = self.local_proj(fused_local)
            gl_gate_input = torch.cat(
                [
                    projected_global,
                    projected_local,
                    torch.abs(projected_global - projected_local),
                    projected_global * projected_local,
                ],
                dim=-1,
            )
            fusion_weights = torch.softmax(self.gl_gate(gl_gate_input), dim=-1)
        else:
            fusion_weights = global_features.new_full((batch_size, 2), 0.5)

        fused_features = (
            fusion_weights[:, :1] * global_features
            + fusion_weights[:, 1:] * fused_local
        )
        logits = self.head(fused_features)

        # In eval mode these branch logits enable disagreement analysis. They
        # are deliberately excluded from the loss and not computed in training.
        if self.training:
            global_logits = logits
            local_logits = None
        else:
            global_logits = self.head(global_features)
            local_logits = self.head(fused_local)
        self._store_diagnostics(
            local_weights=local_weights,
            global_local_weights=fusion_weights,
            global_logits=global_logits,
            local_logits=local_logits,
        )
        return logits
