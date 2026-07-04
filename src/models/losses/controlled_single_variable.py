"""单变量控制实验所需的有序、对比和逐样本加权损失。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import LOSSES


def _require_rank_logits(outputs) -> torch.Tensor:
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    if logits.ndim != 2:
        raise ValueError(f"有序 logits 应为二维张量，实际形状为 {tuple(logits.shape)}。")
    return logits


@LOSSES.register("coral_ordinal")
class CoralOrdinalLoss(nn.Module):
    """对目标 [y>0, y>1, ..., y>K-2] 使用 BCEWithLogitsLoss。"""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        logits = _require_rank_logits(outputs)
        expected_width = self.num_classes - 1
        if logits.shape[1] != expected_width:
            raise ValueError(
                f"CORAL 需要 {expected_width} 个边界 logits，实际为 {logits.shape[1]}。"
            )
        boundaries = torch.arange(expected_width, device=targets.device)
        levels = (targets.unsqueeze(1) > boundaries.unsqueeze(0)).to(logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, levels)


@LOSSES.register("corn_ordinal")
class CornOrdinalLoss(nn.Module):
    """逐边界只在 y>=k 的有效子集上学习条件概率 P(y>k|y>=k)。"""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        logits = _require_rank_logits(outputs)
        expected_width = self.num_classes - 1
        if logits.shape[1] != expected_width:
            raise ValueError(
                f"CORN 需要 {expected_width} 个条件 logits，实际为 {logits.shape[1]}。"
            )
        losses = []
        for boundary_index in range(expected_width):
            valid = targets >= boundary_index
            if not torch.any(valid):
                continue
            conditional_targets = (targets[valid] > boundary_index).to(logits.dtype)
            losses.append(
                F.binary_cross_entropy_with_logits(
                    logits[valid, boundary_index],
                    conditional_targets,
                )
            )
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


@LOSSES.register("ce_adjacent_contrastive")
class CEAdjacentContrastiveLoss(nn.Module):
    """标准 CE 加同类聚合、相邻异类间隔约束，不处理非相邻异类。"""

    def __init__(
        self,
        margin: float = 0.5,
        contrastive_weight: float = 0.05,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError("margin 必须大于 0。")
        if contrastive_weight < 0:
            raise ValueError("contrastive_weight 不能小于 0。")
        self.margin = float(margin)
        self.contrastive_weight = float(contrastive_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            raise ValueError("相邻对比损失要求模型返回 (logits, Stage5 特征)。")
        logits, features = outputs[0], outputs[1]
        classification_loss = F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
        )
        if features.shape[0] < 2 or self.contrastive_weight == 0:
            return classification_loss

        normalized_features = F.normalize(features, p=2, dim=1)
        distances = torch.cdist(normalized_features, normalized_features, p=2)
        upper_triangle = torch.triu(
            torch.ones_like(distances, dtype=torch.bool),
            diagonal=1,
        )
        label_distance = torch.abs(targets.unsqueeze(1) - targets.unsqueeze(0))
        same_class = upper_triangle & (label_distance == 0)
        adjacent_class = upper_triangle & (label_distance == 1)

        pair_losses = []
        if torch.any(same_class):
            pair_losses.append(distances[same_class].square())
        if torch.any(adjacent_class):
            pair_losses.append(
                F.relu(self.margin - distances[adjacent_class]).square()
            )
        if not pair_losses:
            contrastive_loss = features.sum() * 0.0
        else:
            contrastive_loss = torch.cat(pair_losses).mean()
        return classification_loss + self.contrastive_weight * contrastive_loss


@LOSSES.register("weighted_cross_entropy")
class WeightedCrossEntropyLoss(nn.Module):
    """按路径注入的逐样本权重计算 CE；没有权重时严格退化为普通 CE。"""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    def forward(self, outputs, targets: torch.Tensor, extra_targets=None) -> torch.Tensor:
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        loss_each = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        sample_weights = None
        if isinstance(extra_targets, dict):
            sample_weights = extra_targets.get("sample_weights")
        if sample_weights is None:
            sample_weights = torch.ones_like(loss_each)
        sample_weights = sample_weights.to(device=loss_each.device, dtype=loss_each.dtype)
        return (loss_each * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
