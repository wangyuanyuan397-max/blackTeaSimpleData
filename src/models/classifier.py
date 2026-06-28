"""
??: src/models/classifier.py
??: ??????????
????: ?? backbone ? head???????? forward ???
????: ?????????????????
"""

import torch.nn as nn
from ..utils.registry import MODELS, BACKBONES, HEADS

@MODELS.register("classifier")
class ImageClassifier(nn.Module):
    """
    [架构设计] 通用的图像分类器组装类。
    
    设计模式：组合模式 (Composite Pattern)
    它本身不实现具体的卷积或注意力机制，而是作为一个容器，将 "Backbone" (骨干网络) 和 "Head" (分类头) 组合在一起。
    这种设计使得我们可以随意组合不同的 Backbone (如 ResNet, Swin) 和不同的 Head (如 Linear, MLP)，而无需修改代码。
    
    结构流程:
        Input (Image) -> [Backbone] -> Features (Vector/Map) -> [Neck (可选)] -> [Head] -> Logits (Class Scores)
        
    参数:
        backbone (dict or nn.Module): 骨干网络的配置字典或实例。负责从图像提取特征。
        head (dict or nn.Module): 分类头的配置字典或实例。负责将特征映射到类别概率。
        neck (dict or nn.Module, optional): 可选的中间层（如 FPN, GAP），用于特征增强或维度调整。
    """
    def __init__(self, backbone, head, neck=None, return_embeddings=False, aux_head=None):
        super().__init__() # 初始化父类 nn.Module，这是 PyTorch 模型必须的步骤
        
        # 当损失函数既想要 logits，又想要 head 前的特征时，
        # 可以把这个开关设为 True，让 forward 返回 (logits, features)。
        self.return_embeddings = return_embeddings
        self.aux_head = None
        
        # --- 1. 构建 Backbone (特征提取器) ---
        # 逻辑：如果传入的是配置字典，则使用 Registry 动态实例化；如果是已经实例化好的对象，则直接使用。
        # backbone 的职责是把图像张量 [B, 3, H, W] 变成更高层的特征表示。
        if isinstance(backbone, dict):
            # dict.pop("type") 会取出 "type" 对应的值，并从字典中删除该键
            # 例如 backbone={"type": "resnet50", "pretrained": True}
            # 取出后 backbone_type="resnet50", backbone={"pretrained": True}
            backbone_type = backbone.pop("type") 
            
            # BACKBONES.get(name) 返回对应的类 (如 ResNet50)
            # (**backbone) 将剩余参数 (如 pretrained=True) 作为关键字参数传入构造函数
            self.backbone = BACKBONES.get(backbone_type)(**backbone)
        else:
            # 如果已经是 nn.Module 实例，直接赋值
            self.backbone = backbone
            
        # --- 2. 构建 Neck (颈部/中间层) ---
        # 预留位置，用于后续扩展 (如 Global Average Pooling, Feature Pyramid Network)
        # Neck 通常用于将 Backbone 的多尺度特征融合，或者将 4D 特征图转换为 2D 特征向量
        self.neck = None
        if neck:
            # TODO: 实现 Neck Registry
            pass
            
        # --- 3. 构建 Head (分类头) ---
        # head 的职责是把 backbone 特征映射到类别空间。
        if isinstance(head, dict):
            head_type = head.pop("type") # 取出类型名称，如 "linear"
            
            # [自动推断] Head 的输入维度 (in_features)
            # 这是一个非常实用的设计：
            # 我们约定所有的 Backbone 必须有一个 .out_features 属性，记录其输出特征的维度。
            # 这样用户在 YAML 配置 Head 时就不需要手动填写 input_dim，减少因维度不匹配导致的报错。
            # 只要 backbone 暴露了 out_features，这里就不必在 YAML 手填 in_features。
            if hasattr(self.backbone, "out_features"):
                head["in_features"] = self.backbone.out_features
            else:
                # 如果 Backbone 没有定义 out_features，则必须在 Head 配置中显式指定 in_features
                if "in_features" not in head:
                    raise ValueError("Backbone does not have 'out_features' attribute, please specify 'in_features' in head config.")
            
            # 实例化 Head
            self.head = HEADS.get(head_type)(**head)
        else:
            self.head = head

        if aux_head is not None:
            if self.return_embeddings:
                raise ValueError("aux_head and return_embeddings cannot be enabled together in ImageClassifier.")

            if isinstance(aux_head, dict):
                aux_head_type = aux_head.pop("type")
                if hasattr(self.backbone, "out_features"):
                    aux_head["in_features"] = self.backbone.out_features
                elif "in_features" not in aux_head:
                    raise ValueError(
                        "Backbone does not have 'out_features' attribute, please specify 'in_features' in aux_head config."
                    )
                self.aux_head = HEADS.get(aux_head_type)(**aux_head)
            else:
                self.aux_head = aux_head

    def forward(self, x):
        """
        前向传播流程 (Forward Pass)
        
        参数:
            x (Tensor): 输入图像张量。
                        形状: (Batch_Size, Channels, Height, Width)
                        例如: (32, 3, 224, 224)
            
        返回:
            x (Tensor): 模型的原始输出 (Logits)。
                        形状: (Batch_Size, Num_Classes)
                        注意：这里输出的是未经过 Softmax 的原始分数。
                        在训练时，CrossEntropyLoss 会内部进行 Softmax；
                        在推理时，如果需要概率，需要手动加 Softmax。
        """
        # 1. 通过骨干网络提取特征
        # 输入: (B, 3, H, W) -> 输出: (B, Feature_Dim) 或 (B, C, H', W')
        # 先经过 backbone 提取特征。
        x = self.backbone(x) 
        
        # 2. (可选) 通过 Neck 处理特征
        if self.neck:
            x = self.neck(x)
            
        features = x # 保存特征/嵌入供 CLOC Loss 使用
            
        # 3. 通过分类头得到预测结果
        # 输入: (B, Feature_Dim) -> 输出: (B, Num_Classes)
        # 再经过 head 输出最终 logits：[B, D] -> [B, Num_Classes]
        x = self.head(x)

        if self.aux_head is not None:
            aux_output = self.aux_head(features)
            return x, aux_output
        
        # 如果开启了 return_embeddings 标志，则返回 (logits, embeddings) 元组
        return_embeddings_flag = getattr(self, "return_embeddings", False)
        if return_embeddings_flag:
            return x, features
        return x
