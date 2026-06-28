"""
自定义异常体系 (Custom Exception Hierarchy)

设计目标：
    提供清晰的异常分类，便于：
    - 精确捕获特定错误
    - 区分可恢复/不可恢复错误
    - 附加上下文信息
    - 更好的错误处理和日志记录

背景：
    当前问题：
    - 使用通用 Exception
    - 难以区分错误类型
    - 缺少上下文信息
    - 无法判断是否可恢复

解决方案：
    定义分层的异常类，每个异常携带上下文信息。

使用示例：
    # 抛出异常
    raise ModelBuildError(
        "Failed to load pretrained weights",
        model_type="resnet50",
        checkpoint_path="/path/to/model.pth",
        recoverable=False
    )
    
    # 捕获异常
    try:
        model = build_model(config)
    except ModelBuildError as e:
        logger.error("model_build_failed", error=str(e), **e.context)
        if not e.recoverable:
            raise
"""

from typing import Optional, Dict, Any


# ============================================================================
# 基础异常类
# ============================================================================

class BlackTeaError(Exception):
    """
    所有自定义异常的基类
    
    属性:
        message: 错误消息
        context: 上下文信息（字典）
        recoverable: 是否可恢复
        original_exception: 原始异常（如果有）
    """
    
    def __init__(
        self,
        message: str,
        recoverable: bool = False,
        original_exception: Optional[Exception] = None,
        **context
    ):
        """
        初始化异常
        
        参数:
            message: 错误消息
            recoverable: 是否可恢复
            original_exception: 原始异常
            **context: 上下文键值对
        """
        super().__init__(message)
        self.message = message
        self.recoverable = recoverable
        self.original_exception = original_exception
        self.context = context
    
    def __str__(self) -> str:
        """字符串表示"""
        parts = [self.message]
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"[{context_str}]")
        
        if self.original_exception:
            parts.append(f"(caused by: {type(self.original_exception).__name__})")
        
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于日志记录）"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "recoverable": self.recoverable,
            "context": self.context,
            "original_error": (
                type(self.original_exception).__name__
                if self.original_exception
                else None
            ),
        }


# ============================================================================
# 配置相关异常
# ============================================================================

class ConfigError(BlackTeaError):
    """配置错误基类"""
    pass


class ConfigValidationError(ConfigError):
    """配置验证失败"""
    
    def __init__(self, message: str, config_path: Optional[str] = None, **context):
        super().__init__(
            message,
            recoverable=False,  # 配置错误不可恢复
            config_path=config_path,
            **context
        )


class ConfigNotFoundError(ConfigError):
    """配置文件未找到"""
    
    def __init__(self, config_path: str):
        super().__init__(
            f"Configuration file not found: {config_path}",
            recoverable=False,
            config_path=config_path
        )


# ============================================================================
# 数据相关异常
# ============================================================================

class DataError(BlackTeaError):
    """数据错误基类"""
    pass


class DataLoadError(DataError):
    """数据加载失败"""
    
    def __init__(self, message: str, data_path: Optional[str] = None, **context):
        super().__init__(
            message,
            recoverable=False,
            data_path=data_path,
            **context
        )


class DatasetNotFoundError(DataError):
    """数据集未找到"""
    
    def __init__(self, dataset_path: str):
        super().__init__(
            f"Dataset not found: {dataset_path}",
            recoverable=False,
            dataset_path=dataset_path
        )


class DataTransformError(DataError):
    """数据转换失败"""
    
    def __init__(self, message: str, transform_type: Optional[str] = None, **context):
        super().__init__(
            message,
            recoverable=True,  # 单个样本转换失败可能可恢复
            transform_type=transform_type,
            **context
        )


# ============================================================================
# 模型相关异常
# ============================================================================

class ModelError(BlackTeaError):
    """模型错误基类"""
    pass


class ModelBuildError(ModelError):
    """模型构建失败"""
    
    def __init__(
        self,
        message: str,
        model_type: Optional[str] = None,
        **context
    ):
        super().__init__(
            message,
            recoverable=False,
            model_type=model_type,
            **context
        )


class ModelLoadError(ModelError):
    """模型加载失败"""
    
    def __init__(
        self,
        message: str,
        checkpoint_path: Optional[str] = None,
        **context
    ):
        super().__init__(
            message,
            recoverable=False,
            checkpoint_path=checkpoint_path,
            **context
        )


class StrategyError(ModelError):
    """策略相关错误"""
    
    def __init__(
        self,
        message: str,
        strategy_type: Optional[str] = None,
        **context
    ):
        super().__init__(
            message,
            recoverable=False,
            strategy_type=strategy_type,
            **context
        )


# ============================================================================
# 训练相关异常
# ============================================================================

class TrainingError(BlackTeaError):
    """训练错误基类"""
    pass


class OptimizerError(TrainingError):
    """优化器错误"""
    
    def __init__(
        self,
        message: str,
        optimizer_type: Optional[str] = None,
        **context
    ):
        super().__init__(
            message,
            recoverable=False,
            optimizer_type=optimizer_type,
            **context
        )


class LossError(TrainingError):
    """损失函数错误"""
    
    def __init__(
        self,
        message: str,
        loss_type: Optional[str] = None,
        **context
    ):
        super().__init__(
            message,
            recoverable=False,
            loss_type=loss_type,
            **context
        )


class CUDAError(TrainingError):
    """CUDA 相关错误"""
    
    def __init__(self, message: str, **context):
        super().__init__(
            message,
            recoverable=True,  # 可能可以回退到 CPU
            **context
        )


class TrainingInterruptedError(TrainingError):
    """训练被中断"""
    
    def __init__(self, message: str, epoch: Optional[int] = None, **context):
        super().__init__(
            message,
            recoverable=True,  # 可以从 checkpoint 恢复
            epoch=epoch,
            **context
        )


class EvaluationError(TrainingError):
    """评估错误"""
    
    def __init__(self, message: str, **context):
        super().__init__(
            message,
            recoverable=False,
            **context
        )


# ============================================================================
# 资源相关异常
# ============================================================================

class ResourceError(BlackTeaError):
    """资源错误基类"""
    pass


class OutOfMemoryError(ResourceError):
    """内存不足"""
    
    def __init__(
        self,
        message: str,
        required_mb: Optional[int] = None,
        available_mb: Optional[int] = None,
        **context
    ):
        super().__init__(
            message,
            recoverable=True,  # 可以减小 batch size
            required_mb=required_mb,
            available_mb=available_mb,
            **context
        )


class GPUNotAvailableError(ResourceError):
    """GPU 不可用"""
    
    def __init__(self, message: str = "CUDA not available"):
        super().__init__(
            message,
            recoverable=True,  # 可以回退到 CPU
        )


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("异常体系示例")
    print("=" * 60)
    
    # 示例 1: ModelBuildError
    try:
        raise ModelBuildError(
            "Failed to load pretrained weights",
            model_type="resnet50",
            checkpoint_path="/path/to/model.pth"
        )
    except ModelBuildError as e:
        print(f"\n1. ModelBuildError:")
        print(f"   Message: {e.message}")
        print(f"   Recoverable: {e.recoverable}")
        print(f"   Context: {e.context}")
        print(f"   Dict: {e.to_dict()}")
    
    # 示例 2: DataLoadError with original exception
    try:
        try:
            # 模拟原始错误
            raise FileNotFoundError("data.csv not found")
        except FileNotFoundError as original:
            raise DataLoadError(
                "Failed to load training data",
                data_path="./data/train",
                original_exception=original
            )
    except DataLoadError as e:
        print(f"\n2. DataLoadError:")
        print(f"   String: {e}")
        print(f"   Original: {e.original_exception}")
    
    # 示例 3: OutOfMemoryError (recoverable)
    try:
        raise OutOfMemoryError(
            "CUDA out of memory",
            required_mb=8192,
            available_mb=4096,
            batch_size=32
        )
    except OutOfMemoryError as e:
        print(f"\n3. OutOfMemoryError:")
        print(f"   Recoverable: {e.recoverable}")
        print(f"   Suggestion: Reduce batch_size from {e.context['batch_size']}")
    
    # 示例 4: 异常层次
    print(f"\n4. 异常层次:")
    print(f"   ModelBuildError is BlackTeaError: {issubclass(ModelBuildError, BlackTeaError)}")
    print(f"   ModelBuildError is ModelError: {issubclass(ModelBuildError, ModelError)}")
    
    print("\n✓ 异常体系示例完成")
