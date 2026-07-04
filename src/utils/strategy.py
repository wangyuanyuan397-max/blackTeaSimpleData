"""
模型处理策略模式 (Model Strategy Pattern)

设计目标：
    解决不同模型类型需要不同处理逻辑的问题，消除脆弱的 hasattr 判断。

背景：
    当前 Trainer 使用 hasattr(model.head, "get_label") 来判断模型类型，
    这种方式隐式、脆弱且难以扩展。当 scheduler 运行多个不同类型的模型时，
    容易出现兼容性问题。

解决方案：
    使用策略模式，将模型特定的处理逻辑封装到独立的策略类中，
    通过配置文件显式声明使用哪种策略。

使用示例：
    # 配置文件
    model:
      type: "classifier"
      strategy: "classification"  # 显式声明策略
      backbone:
        type: "resnet50"
      head:
        type: "linear"
    
    # Trainer 使用
    strategy = build_strategy(config, model)
    predictions = strategy.get_predictions(outputs)
"""

from typing import Protocol, Tuple
import torch
import torch.nn as nn


class ModelStrategy(Protocol):
    """
    模型处理策略协议
    
    定义了所有策略必须实现的接口。
    使用 Protocol 而非 ABC，提供更灵活的鸭子类型检查。
    """
    
    def get_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        从模型输出获取预测标签
        
        参数:
            outputs: 模型输出，可能是 logits, (logits, embeddings) 等
            
        返回:
            predictions: 预测的类别标签 [Batch]
        """
        ...
    
    def calculate_metrics(
        self, 
        outputs: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[int, int]:
        """
        计算评估指标
        
        参数:
            outputs: 模型输出
            labels: 真实标签
            
        返回:
            (correct_count, total_count): 正确预测数和总样本数
        """
        ...


class ClassificationStrategy:
    """
    标准分类策略
    
    适用于：
        - 标准多分类任务
        - 输出为 logits [Batch, NumClasses]
        - 使用 argmax 获取预测
    
    示例模型：
        - ResNet + LinearHead
        - ViT + LinearHead
        - 任何标准分类模型
    """
    
    def __init__(self, model: nn.Module):
        """
        参数:
            model: PyTorch 模型实例（用于未来扩展）
        """
        # 分类策略只关心一件事：把 logits 解释成类别预测。
        self.model = model
    
    def get_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        使用 argmax 获取预测
        
        参数:
            outputs: Logits [Batch, NumClasses]
            
        返回:
            predictions: 预测类别 [Batch]
        """
        # 处理元组输出（某些模型返回 (logits, features)）
        # 某些模型会返回 (logits, features)，这里只取 logits 做分类。
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # argmax: 获取概率最大的类别
        _, predictions = torch.max(outputs, dim=1)
        return predictions
    
    def calculate_metrics(
        self, 
        outputs: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[int, int]:
        """
        计算分类准确率
        
        返回:
            (correct, total): 正确数和总数
        """
        # 对标准分类任务，正确数 = argmax(logits) 与 labels 的逐元素比较。
        predictions = self.get_predictions(outputs)
        correct = predictions.eq(labels).sum().item()
        total = labels.size(0)
        return correct, total


class OrdinalRegressionStrategy:
    """
    序数回归策略
    
    适用于：
        - 序数回归任务（类别有顺序关系）
        - 使用多个二分类器
        - Head 有自定义的 get_label() 方法
    
    示例模型：
        - Swin + OrdinalSwinHead
        - 任何实现了 get_label() 的 Head
    
    技术细节：
        序数回归将 K 类问题转换为 K-1 个二分类问题：
        - 类别 0, 1, 2, 3
        - 二分类: [>0?, >1?, >2?]
        - 预测 [1, 1, 0] → 类别 2
    """
    
    def __init__(self, model: nn.Module):
        """
        参数:
            model: 必须有 head.get_label() 方法的模型
        """
        self.model = model
        
        # 验证模型是否支持序数回归
        if not hasattr(model, "head"):
            raise ValueError(
                "OrdinalRegressionStrategy requires model to have 'head' attribute"
            )
        
        if not hasattr(model.head, "get_label"):
            raise ValueError(
                f"Model head {type(model.head).__name__} does not implement get_label() method. "
                f"Required for ordinal regression strategy."
            )
    
    def get_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        使用 Head 的 get_label() 方法获取预测
        
        参数:
            outputs: 模型输出（通常是二分类 logits）
            
        返回:
            predictions: 序数类别 [Batch]
        """
        # 处理元组输出
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 调用 Head 的自定义解码方法
        predictions = self.model.head.get_label(outputs)
        return predictions
    
    def calculate_metrics(
        self, 
        outputs: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[int, int]:
        """
        计算序数回归准确率
        
        返回:
            (correct, total): 正确数和总数
        """
        predictions = self.get_predictions(outputs)
        correct = predictions.eq(labels).sum().item()
        total = labels.size(0)
        return correct, total


class ContrastiveLearningStrategy:
    """
    对比学习策略（示例，未来扩展）
    
    适用于：
        - 对比学习任务
        - 输出为 (embeddings, logits)
        - 需要特殊的相似度计算
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def get_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        对比学习的预测逻辑
        
        注：这是一个示例实现，实际需要根据具体任务定制
        """
        if isinstance(outputs, tuple):
            embeddings, logits = outputs
            # 使用 logits 进行分类
            _, predictions = torch.max(logits, dim=1)
            return predictions
        else:
            raise ValueError("Contrastive learning expects (embeddings, logits) output")
    
    def calculate_metrics(
        self, 
        outputs: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[int, int]:
        """
        计算对比学习指标
        """
        predictions = self.get_predictions(outputs)
        correct = predictions.eq(labels).sum().item()
        total = labels.size(0)
        return correct, total


# ============================================================================
# 策略工厂函数
# ============================================================================

def build_strategy(config: dict, model: nn.Module) -> ModelStrategy:
    """
    根据配置构建策略
    
    参数:
        config: 模型配置字典
        model: PyTorch 模型实例
        
    返回:
        strategy: 模型处理策略实例
        
    示例:
        config = {
            "type": "classifier",
            "strategy": "classification",
            ...
        }
        strategy = build_strategy(config, model)
    """
    strategy_type = config.get("strategy", "classification")  # 默认为标准分类
    
    # 策略映射
    STRATEGY_MAP = {
        "classification": ClassificationStrategy,
        "ordinal_regression": OrdinalRegressionStrategy,
        "contrastive": ContrastiveLearningStrategy,
    }
    
    if strategy_type not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available strategies: {list(STRATEGY_MAP.keys())}"
        )
    
    strategy_class = STRATEGY_MAP[strategy_type]
    
    try:
        strategy = strategy_class(model)
        return strategy
    except Exception as e:
        raise RuntimeError(
            f"Failed to build strategy '{strategy_type}': {e}"
        ) from e


# ============================================================================
# 配置验证
# ============================================================================

def validate_strategy_config(config: dict) -> None:
    """
    验证策略配置是否合法
    
    参数:
        config: 模型配置字典
        
    抛出:
        ValueError: 配置不合法时
        
    示例:
        validate_strategy_config(config)
    """
    strategy_type = config.get("strategy", "classification")
    head_type = config.get("head", {}).get("type", "")
    
    # 规则1: 序数回归策略需要序数回归 Head
    if strategy_type == "ordinal_regression":
        # 允许的序数回归 head 类型
        ordinal_head_types = [
            "ordinal",           # 原有的 ordinal heads (如 linearOrdinalSwinHead)
            "spacecutter_head",  # SpaceCutter 累积链接函数
            "coral_head",        # CORAL 秩一致性序数回归
            "corn_head",         # CORN 条件有序回归
        ]
        
        # 检查 head_type 是否符合任一允许的类型
        is_ordinal_head = any(
            allowed_type in head_type.lower() 
            for allowed_type in ordinal_head_types
        )
        
        if not is_ordinal_head:
            raise ValueError(
                f"Strategy 'ordinal_regression' requires an ordinal head, "
                f"but got head type: '{head_type}'. "
                f"Allowed ordinal heads: {ordinal_head_types}"
            )
    
    # 规则2: 标准分类策略不应使用序数回归 Head
    if strategy_type == "classification":
        # 不允许与分类策略一起使用的 head 类型
        ordinal_only_heads = ["ordinal", "spacecutter_head", "coral_head", "corn_head"]
        
        is_ordinal_only = any(
            disallowed in head_type.lower() 
            for disallowed in ordinal_only_heads
        )
        
        if is_ordinal_only:
            raise ValueError(
                f"Strategy 'classification' should not use ordinal head '{head_type}'. "
                f"Please change strategy to 'ordinal_regression' or use a standard head."
            )
