"""
类型安全的配置管理系统 (Type-Safe Configuration System)

设计目标：
    从 dict 升级为 Pydantic 模型，提供：
    - 类型安全（IDE 自动补全）
    - 运行时验证（自动检查配置合法性）
    - 更好的错误提示
    - 支持多种配置源（YAML, 环境变量, 命令行）

背景：
    当前使用 yaml.safe_load() 返回 dict，存在问题：
    - 无类型检查，cfg["optimizer"]["lr"] 可能键不存在
    - 无法在 IDE 中自动补全
    - 错误只在运行时发现

解决方案：
    使用 Pydantic 定义配置模型，自动验证和类型转换。

使用示例：
    # 旧方式
    cfg = yaml.safe_load(f)
    lr = cfg["optimizer"]["lr"]  # 可能 KeyError
    
    # 新方式
    cfg = TrainingConfig.from_yaml("config.yaml")
    lr = cfg.optimizer.lr  # 类型安全，IDE 自动补全
"""

from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml
import os

from .config_compiler import compile_run_config, is_thin_run_config, load_yaml_raw


# ============================================================================
# 基础配置模型
# ============================================================================

class BackboneConfig(BaseModel):
    """
    骨干网络配置
    
    示例:
        # ResNet
        type: "resnet50"
        pretrained: True
        
        # Timm (ViT, Swin)
        type: "timm"
        model_name: "vit_base_patch16_224"
        pretrained: True
    """
    type: str = Field(..., description="骨干网络类型")
    pretrained: bool = Field(True, description="是否使用预训练权重")
    
    # Timm 特定参数
    model_name: Optional[str] = Field(None, description="Timm 模型名称")
    
    # ResNet 特定参数
    layers: Optional[List[int]] = Field(None, description="ResNet 层数配置")
    
    model_config = {"extra": "allow"}  # Pydantic v2 语法


class HeadConfig(BaseModel):
    """
    分类头配置
    
    示例:
        type: "linear"
        num_classes: 4  # 自动注入
        drop_rate: 0.1
    """
    type: str = Field(..., description="Head 类型")
    num_classes: Optional[int] = Field(None, description="类别数（自动注入）")
    drop_rate: float = Field(0.0, description="Dropout 比例", ge=0.0, le=1.0)
    
    model_config = {"extra": "allow"}


class ModelConfig(BaseModel):
    """
    模型配置
    
    示例:
        type: "classifier"
        strategy: "classification"
        backbone: {...}
        head: {...}
    """
    type: str = Field(..., description="模型类型")
    strategy: str = Field("classification", description="处理策略")
    return_embeddings: bool = Field(
        False,
        description="是否返回 (logits, embeddings) 供带特征正则的损失函数使用",
    )
    backbone: BackboneConfig = Field(..., description="骨干网络配置")
    head: HeadConfig = Field(..., description="分类头配置")
    
    model_config = {"extra": "allow"}
    
    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v):
        """验证策略类型"""
        allowed = ["classification", "ordinal_regression", "contrastive"]
        if v not in allowed:
            raise ValueError(f"Invalid strategy: {v}. Allowed: {allowed}")
        return v


class OptimizerConfig(BaseModel):
    """
    优化器配置
    
    示例:
        type: "adamw"
        lr: 0.0001
        weight_decay: 0.05
    """
    type: str = Field(..., description="优化器类型")
    lr: float = Field(..., description="学习率", gt=0.0)
    weight_decay: float = Field(0.0, description="权重衰减", ge=0.0)
    betas: Optional[List[float]] = Field(None, description="Adam beta 参数")
    momentum: Optional[float] = Field(None, description="SGD 动量", ge=0.0, le=1.0)
    
    model_config = {"extra": "allow"}


class SchedulerConfig(BaseModel):
    """
    学习率调度器配置
    
    示例:
        type: "cosine"
        warmup_epochs: 5
        min_lr: 0.000001
    """
    type: str = Field(..., description="调度器类型")
    warmup_epochs: int = Field(0, description="预热轮数", ge=0)
    min_lr: Optional[float] = Field(None, description="最小学习率", gt=0.0)
    
    model_config = {"extra": "allow"}


class DataConfig(BaseModel):
    """
    数据配置
    
    示例:
        root: "./data/tea_fermentation"
        image_size: 224
        num_workers: 4
    """
    root: Optional[str] = Field(None, description="数据根目录（可在运行时注入）")
    image_size: Optional[int] = Field(224, description="图像大小", gt=0)
    num_workers: Optional[int] = Field(4, description="数据加载线程数", ge=0)
    
    # 兼容旧配置
    dataset_type: Optional[str] = Field(None, description="数据集类型")
    train_transform: Optional[Dict[str, Any]] = Field(None, description="训练数据增强")
    eval_transform: Optional[Dict[str, Any]] = Field(None, description="验证数据增强")
    val_transform: Optional[str] = Field(None, description="验证数据增强（旧字段）")
    
    model_config = {"extra": "allow"}
    
    @field_validator("root")
    @classmethod
    def validate_root(cls, v):
        """验证数据目录存在"""
        # 跳过验证如果路径以 ../ 开头（相对路径）
        if v.startswith("../") or v.startswith("./"):
            return v
        if not os.path.exists(v):
            # 警告但不报错（允许相对路径）
            import warnings
            warnings.warn(f"Data root directory may not exist: {v}")
        return v


class TrainConfig(BaseModel):
    """
    训练配置
    
    示例:
        epochs: 100
        batch_size: 32
        device: "cuda"
    """
    epochs: int = Field(..., description="训练轮数", gt=0)
    batch_size: int = Field(..., description="批次大小", gt=0)
    device: Optional[str] = Field("cuda", description="计算设备")
    
    # 可选参数
    seed: Optional[int] = Field(None, description="随机种子", ge=0)
    gradient_clip: Optional[float] = Field(None, description="梯度裁剪", gt=0.0)
    mixed_precision: Optional[bool] = Field(False, description="混合精度训练")
    enable_error_analysis: Optional[bool] = Field(True, description="训练结束后是否运行 error_analysis")
    error_analysis_save_images: Optional[bool] = Field(True, description="error_analysis 是否保存误分类图片到 images/")
    
    model_config = {"extra": "allow"}
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        """验证设备"""
        if v is None:
            return "cuda"
        import torch
        if v == "cuda" and not torch.cuda.is_available():
            import warnings
            warnings.warn("CUDA not available, falling back to CPU")
            return "cpu"
        return v


class WandBConfig(BaseModel):
    """
    WandB 配置
    
    示例:
        enabled: True
        project: "tea_fermentation"
        entity: "my_team"
    """
    enabled: bool = Field(False, description="是否启用 WandB")
    project: Optional[str] = Field(None, description="项目名称")
    entity: Optional[str] = Field(None, description="团队名称")
    run_name: Optional[str] = Field(None, description="运行名称")
    
    @model_validator(mode='after')
    def validate_wandb(self):
        """如果启用 WandB，必须提供 project"""
        if self.enabled and not self.project:
            raise ValueError("WandB enabled but project not specified")
        return self


# ============================================================================
# 顶层配置模型
# ============================================================================

class TrainingConfig(BaseModel):
    """
    完整训练配置
    
    这是顶层配置模型，包含所有子配置。
    
    使用示例:
        # 从 YAML 加载
        config = TrainingConfig.from_yaml("config.yaml")
        
        # 访问配置
        lr = config.optimizer.lr
        epochs = config.train.epochs
        
        # 验证
        config.validate()
    """
    # 基础信息
    run_name: Optional[str] = Field(None, description="运行名称")
    description: Optional[str] = Field(None, description="运行描述")
    
    # 核心配置
    model: ModelConfig = Field(..., description="模型配置")
    data: DataConfig = Field(..., description="数据配置")
    train: TrainConfig = Field(..., description="训练配置")
    optimizer: OptimizerConfig = Field(..., description="优化器配置")
    
    # 可选配置
    scheduler: Optional[SchedulerConfig] = Field(None, description="学习率调度器")
    loss: Optional[Dict[str, Any]] = Field(None, description="损失函数配置")
    wandb: Optional[WandBConfig] = Field(None, description="WandB 配置")
    
    # 其他配置（向后兼容）
    use_wandb: bool = Field(False, description="是否使用 WandB（旧字段）")
    enable_google_drive_upload: bool = Field(False, description="训练完成后是否自动上传结果到Google Drive")
    
    model_config = {"extra": "allow"}  # Pydantic v2

    @staticmethod
    def load_yaml_data(yaml_path: str) -> dict:
        """
        读取 YAML 配置。

        - 普通完整 YAML：直接返回原始内容
        - 薄实例 YAML：先动态编译，再返回完整配置 dict
        """
        data = load_yaml_raw(yaml_path)
        if is_thin_run_config(data):
            return compile_run_config(yaml_path)
        return data
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """
        从 YAML 文件加载配置
        
        参数:
            yaml_path: YAML 文件路径
            
        返回:
            TrainingConfig: 验证后的配置对象
            
        示例:
            config = TrainingConfig.from_yaml("configs/resnet/resnet50.yaml")
        """
        data = cls.load_yaml_data(yaml_path)
        
        # Pydantic 自动验证
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        """
        从字典加载配置
        
        参数:
            data: 配置字典
            
        返回:
            TrainingConfig: 验证后的配置对象
        """
        return cls(**data)
    
    def to_dict(self) -> dict:
        """
        转换为字典（用于向后兼容）
        
        返回:
            dict: 配置字典
        """
        return self.model_dump()  # Pydantic v2
    
    def to_yaml(self, output_path: str) -> None:
        """
        保存为 YAML 文件
        
        参数:
            output_path: 输出文件路径
        """
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, allow_unicode=True, default_flow_style=False)
    
    def validate_all(self) -> None:
        """
        执行额外的跨字段验证
        
        抛出:
            ValueError: 配置不合法时
        """
        # 验证策略与 Head 匹配
        from ..utils.strategy import validate_strategy_config
        validate_strategy_config(self.model.model_dump())  # Pydantic v2
        
        # 验证 WandB 配置
        if self.use_wandb and not self.wandb:
            raise ValueError("use_wandb is True but wandb config is missing")


# ============================================================================
# 配置构建器（向后兼容）
# ============================================================================

def load_config(yaml_path: str, validate: bool = True) -> TrainingConfig:
    """
    加载配置文件（便捷函数）
    
    参数:
        yaml_path: YAML 文件路径
        validate: 是否执行额外验证
        
    返回:
        TrainingConfig: 配置对象
        
    示例:
        config = load_config("configs/resnet/resnet50.yaml")
    """
    config = TrainingConfig.from_yaml(yaml_path)
    
    if validate:
        config.validate_all()
    
    return config


def load_config_as_dict(yaml_path: str) -> dict:
    """
    加载配置为字典（向后兼容旧代码）
    
    参数:
        yaml_path: YAML 文件路径
        
    返回:
        dict: 配置字典
    """
    return TrainingConfig.load_yaml_data(yaml_path)


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    # 示例：从 YAML 加载配置
    config = TrainingConfig.from_yaml("configs/resnet/resnet50.yaml")
    
    # 类型安全访问
    print(f"Learning rate: {config.optimizer.lr}")
    print(f"Epochs: {config.train.epochs}")
    print(f"Strategy: {config.model.strategy}")
    
    # 验证
    config.validate_all()
    
    print("✓ Configuration loaded and validated successfully!")
