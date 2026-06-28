"""
??: src/models/losses/ce_loss.py
??: ???????
????: ????????? batch ??????
????: ????????????????
"""

import torch.nn as nn
from ...utils.registry import LOSSES

@LOSSES.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    """
    [损失函数] 标准交叉熵损失 (Cross Entropy Loss)。
    
    数学原理：
    CrossEntropy(p, q) = - sum(p(x) * log(q(x)))
    其中 p 是真实分布 (通常是 One-hot)，q 是预测分布 (Softmax 输出)。
    
    用途：
    多分类问题的默认选择。衡量预测概率分布与真实标签分布之间的差异。
    
    封装：
    这里对 PyTorch 原生的 nn.CrossEntropyLoss 进行了简单封装，使其适配 Registry 机制。
    """
    def __init__(self, label_smoothing=0.0):
        """
        初始化交叉熵损失。
        
        参数:
            label_smoothing (float): 标签平滑系数 (0.0-1.0)。
                默认 0.0 (不使用)。
                
                [原理解释]
                如果 label_smoothing = 0.1，类别数 K=4：
                真实标签 One-hot: [0, 1, 0, 0]
                平滑后标签:       [0.025, 0.925, 0.025, 0.025]
                公式: y_new = (1 - epsilon) * y_old + epsilon / K
                
                [为什么需要?]
                防止模型对预测结果过于自信 (Over-confident)。
                如果没有平滑，模型会试图让正确类别的 Logit 趋向于正无穷，这可能导致过拟合。
                平滑后，模型只需要让预测概率达到 0.925 即可，不需要无限大，从而提高了泛化能力。
        """
        super().__init__()
        # nn.CrossEntropyLoss 内部已经包含了 Softmax 操作，所以输入应该是 Logits (未归一化的分数)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, preds, targets):
        """
        参数:
            preds (Tensor): 模型输出的 Logits，形状 (Batch, Num_Classes)。
            targets (Tensor): 真实标签索引，形状 (Batch)。取值范围 [0, Num_Classes-1]。
            
        返回:
            Tensor: 标量 Loss (Batch 内平均值)。
        """
        return self.criterion(preds, targets)
