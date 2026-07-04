"""用于受控实验的 CORAL 与 CORN 有序分类头。"""

import torch
import torch.nn as nn

from ...utils.registry import HEADS


@HEADS.register("coral_head")
class CoralHead(nn.Module):
    """用一个共享特征投影和 K-1 个偏置预测累计有序边界。"""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        drop_rate: float = 0.0,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CORAL 至少需要两个类别。")
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.shared_projection = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.num_classes - 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        score = self.shared_projection(self.dropout(features))
        return score + self.bias.unsqueeze(0)

    def get_label(self, logits: torch.Tensor) -> torch.Tensor:
        """把三个累计边界概率超过阈值的数量解码为 0～3 类。"""
        return (torch.sigmoid(logits) > self.threshold).sum(dim=1).long()


@HEADS.register("corn_head")
class CornHead(nn.Module):
    """输出 K-1 个条件有序 logits，并按累计条件概率解码。"""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        drop_rate: float = 0.0,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CORN 至少需要两个类别。")
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(in_features, self.num_classes - 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))

    def get_label(self, logits: torch.Tensor) -> torch.Tensor:
        """将条件概率连乘成 P(y>k)，再计算通过的有序边界数。"""
        conditional_probabilities = torch.sigmoid(logits)
        rank_probabilities = torch.cumprod(conditional_probabilities, dim=1)
        return (rank_probabilities > self.threshold).sum(dim=1).long()
