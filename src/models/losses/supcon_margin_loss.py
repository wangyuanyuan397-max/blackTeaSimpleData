"""Loss for SupCon + cosine-margin classification experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import LOSSES


@LOSSES.register("supcon_margin")
class SupConMarginLoss(nn.Module):
    """训练态使用 margin CE + SupCon，验证/测试态退化为普通 CE。"""

    def __init__(
        self,
        lambda_supcon: float = 1.0,
        temperature: float = 0.1,
        lambda_ce: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda_supcon = float(lambda_supcon)
        self.temperature = float(temperature)
        self.lambda_ce = float(lambda_ce)
        self.eps = float(eps)
        if self.lambda_supcon < 0:
            raise ValueError("lambda_supcon must be non-negative.")
        if self.lambda_ce < 0:
            raise ValueError("lambda_ce must be non-negative.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")

    @staticmethod
    def _split_outputs(outputs: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            return outputs, {}
        logits = outputs[0]
        auxiliary = outputs[1] if isinstance(outputs[1], dict) else {}
        return logits, auxiliary

    def _repeat_labels(self, labels: torch.Tensor, auxiliary: dict[str, torch.Tensor]) -> torch.Tensor:
        num_views_tensor = auxiliary.get("num_views")
        num_views = int(num_views_tensor.item()) if torch.is_tensor(num_views_tensor) else 1
        return labels.repeat_interleave(num_views)

    @staticmethod
    def _apply_margin(
        cosine_logits: torch.Tensor,
        labels: torch.Tensor,
        margin: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        # 只对真实类别的余弦 logit 扣 margin，相当于给正确类提高判别门槛。
        margin_logits = cosine_logits.clone()
        row_index = torch.arange(labels.numel(), device=labels.device)
        margin_logits[row_index, labels] = margin_logits[row_index, labels] - margin
        return margin_logits * scale

    def _supervised_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # 标准 SupCon：分母包含除自身外的所有样本，分子只累计同类正样本。
        features = F.normalize(features, dim=1)
        logits = features.matmul(features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        batch_size = labels.numel()
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).to(features.dtype)
        self_mask = torch.eye(batch_size, device=features.device, dtype=features.dtype)
        logits_mask = 1.0 - self_mask
        positive_mask = positive_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        positive_count = positive_mask.sum(dim=1)
        valid_anchor = positive_count > 0
        if not torch.any(valid_anchor):
            return features.new_zeros(())

        mean_log_prob_pos = (
            (positive_mask * log_prob).sum(dim=1)[valid_anchor]
            / positive_count[valid_anchor].clamp_min(self.eps)
        )
        return -mean_log_prob_pos.mean()

    def forward(self, outputs: Any, targets: torch.Tensor, extra_targets=None) -> torch.Tensor:
        logits, auxiliary = self._split_outputs(outputs)

        if not auxiliary:
            # 验证/测试阶段没有双视图辅助张量，按实际推理 logits 计算 CE。
            return F.cross_entropy(logits, targets)

        repeated_targets = self._repeat_labels(targets, auxiliary)
        cosine_logits = auxiliary.get("view_cosine_logits")
        projected_features = auxiliary.get("view_projected_features")

        if cosine_logits is None:
            raise ValueError("supcon_margin requires auxiliary['view_cosine_logits'] during training.")
        if projected_features is None and self.lambda_supcon > 0:
            raise ValueError("supcon_margin requires auxiliary['view_projected_features'] when lambda_supcon > 0.")

        margin = auxiliary.get("margin", cosine_logits.new_tensor(0.0))
        scale = auxiliary.get("scale", cosine_logits.new_tensor(1.0))
        margin_logits = self._apply_margin(cosine_logits, repeated_targets, margin, scale)

        total = cosine_logits.new_zeros(())
        if self.lambda_ce > 0:
            total = total + self.lambda_ce * F.cross_entropy(margin_logits, repeated_targets)

        if self.lambda_supcon > 0 and projected_features is not None:
            total = total + self.lambda_supcon * self._supervised_contrastive_loss(
                projected_features,
                repeated_targets,
            )
        return total

    def compute_aux_metrics(self, outputs: Any, extra_targets=None) -> dict[str, float]:
        _, auxiliary = self._split_outputs(outputs)
        if not auxiliary:
            return {}
        metrics: dict[str, float] = {}
        features = auxiliary.get("view_projected_features")
        if torch.is_tensor(features):
            metrics["supcon_projected_norm"] = float(features.detach().norm(dim=1).mean().cpu().item())
        return metrics
