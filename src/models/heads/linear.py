"""
??: src/models/heads/linear.py
??: ??/???????
????: ????????????????????????
????: ?????????????????
"""

import torch.nn as nn
from ...utils.registry import HEADS

@HEADS.register("identity")
class IdentityHead(nn.Module):
    """Pass-through head for backbones that already return logits."""

    def __init__(self, in_features=None, num_classes=None, drop_rate=0.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

    def forward(self, x):
        return x


@HEADS.register("linear")
class LinearHead(nn.Module):
    """
    [分类头] 简单的线性分类头 (Linear Classification Head)。
    
    结构：
    Dropout -> Linear (FC Layer)
    
    适用场景：
    大多数基础分类任务。当 Backbone (如 ResNet50, ViT) 提取的特征已经非常具有辨别力时，一个简单的线性层就足以将特征映射到类别概率空间。
    """
    def __init__(self, in_features, num_classes, drop_rate=0.0):
        """
        初始化线性头。
        
        参数:
            in_features (int): 输入特征维度 (来自 Backbone 的输出)。
                               例如 ResNet50 是 2048，ViT-Base 是 768。
            num_classes (int): 类别数量 (输出 logits 维度)。
                               例如红茶发酵程度分类任务是 4 类。
            drop_rate (float): Dropout 概率 (0.0-1.0)。
                               Dropout 是一种正则化手段，在训练时随机将一部分神经元的输出置为 0，防止过拟合。
        """
        super().__init__()
        # 这是最适合入门改模块的 head：足够简单，不容易把问题藏起来。
        
        # 为什么用 Identity 而不是在 forward 里写 if/else？
        # 使用 nn.Identity() (直通层) 可以保持 forward 代码的简洁和一致性。
        # 无论是否有 dropout，forward 流程都是 self.dropout(x) -> self.fc(x)。
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        
        # 全连接层 (Fully Connected Layer)
        # 数学运算: y = xA^T + b
        # 将 in_features 维的特征向量投影到 num_classes 维的类别空间
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        参数:
            x (Tensor): 特征向量，形状 (Batch, In_Features)
            
        返回:
            Tensor: Logits，形状 (Batch, Num_Classes)
        """
        # 输入形状：[B, in_features]
        x = self.dropout(x)
        # 输出形状：[B, num_classes]。这里输出的是 logits，不是 softmax 概率。
        return self.fc(x)

@HEADS.register("mlp")
class MLPHead(nn.Module):
    """
    [分类头] 多层感知机分类头 (MLP Head)。
    
    结构：
    Linear -> ReLU -> Dropout -> Linear
    
    适用场景：
    1. 当 Backbone 提取的特征与最终类别之间的关系比较复杂，线性不可分时。
    2. 自监督学习 (Self-Supervised Learning) 如 SimCLR, MoCo 中常用于 Projector。
    3. 需要更强的拟合能力时。
    """
    def __init__(self, in_features, hidden_features, num_classes, drop_rate=0.0):
        """
        参数:
            in_features (int): 输入维度。
            hidden_features (int): 中间隐藏层的维度。通常设为 in_features 的 1x 到 4x。
            num_classes (int): 输出类别数。
            drop_rate (float): Dropout 概率。
        """
        super().__init__()
        # 第一层映射：从特征空间 -> 隐藏空间
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # 激活函数：引入非线性
        # ReLU (Rectified Linear Unit) 是最常用的激活函数: f(x) = max(0, x)
        self.act = nn.ReLU() 
        
        # 正则化
        self.drop = nn.Dropout(drop_rate)
        
        # 第二层映射：从隐藏空间 -> 类别空间
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


@HEADS.register("regression")
class RegressionHead(nn.Module):
    """Simple shared-feature regression head for auxiliary continuous targets."""

    def __init__(self, in_features, out_features=1, drop_rate=0.0, activation=None):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, out_features)

        activation_name = None if activation is None else str(activation).lower()
        if activation_name in (None, "", "identity", "none"):
            self.activation = nn.Identity()
        elif activation_name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_name == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported regression head activation: {activation}")

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return self.activation(x)
