"""Cross-entropy with AD-GBC geometric regularization."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import LOSSES


@LOSSES.register("adgbc_cross_entropy")
class ADGBCCrossEntropyLoss(nn.Module):
    """CE plus AD-GBC Wasserstein diversity and scale-consistency penalties."""

    def __init__(
        self,
        label_smoothing: float = 0.0,
        lambda_w_div: float = 0.05,
        beta_scale_con: float = 0.05,
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.lambda_w_div = float(lambda_w_div)
        self.beta_scale_con = float(beta_scale_con)

    @staticmethod
    def _split_outputs(outputs: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            return outputs, {}
        logits = outputs[0]
        auxiliary = outputs[1] if isinstance(outputs[1], dict) else {}
        return logits, auxiliary

    @staticmethod
    def _required_auxiliary(
        auxiliary: dict[str, torch.Tensor],
        key: str,
        weight: float,
    ) -> torch.Tensor | None:
        if weight == 0:
            return None
        value = auxiliary.get(key)
        if value is None:
            # 验证/测试阶段不计算 AD-GBC 辅助项，只保留 CE，避免额外特征统计开销。
            if not torch.is_grad_enabled():
                return None
            raise ValueError(
                f"adgbc_cross_entropy requires auxiliary[{key!r}] when its weight is non-zero."
            )
        return value

    def forward(self, outputs: Any, targets: torch.Tensor, extra_targets=None) -> torch.Tensor:
        logits, auxiliary = self._split_outputs(outputs)
        total = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

        # 训练阶段在 CE 外叠加 Wasserstein 多样性和尺度一致性两个几何约束。
        loss_w = self._required_auxiliary(
            auxiliary,
            "adgbc_loss_w_div",
            self.lambda_w_div,
        )
        if loss_w is not None:
            total = total + self.lambda_w_div * loss_w

        loss_scale = self._required_auxiliary(
            auxiliary,
            "adgbc_loss_scale_con",
            self.beta_scale_con,
        )
        if loss_scale is not None:
            total = total + self.beta_scale_con * loss_scale
        return total

    def compute_aux_metrics(self, outputs: Any, extra_targets=None) -> dict[str, float]:
        _, auxiliary = self._split_outputs(outputs)
        metrics: dict[str, float] = {}
        for key, value in auxiliary.items():
            if not torch.is_tensor(value) or value.numel() != 1:
                continue
            metrics[key] = float(value.detach().cpu().item())
        return metrics
