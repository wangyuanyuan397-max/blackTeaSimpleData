"""
配置管理模块

提供类型安全的配置加载和验证功能。
"""

from .training_config import (
    TrainingConfig,
    ModelConfig,
    DataConfig,
    TrainConfig,
    OptimizerConfig,
    SchedulerConfig,
    WandBConfig,
    load_config,
    load_config_as_dict,
)
from .config_compiler import (
    ConfigCompilerError,
    CompiledRunBundle,
    build_compilation_bundle,
    compile_run_config,
    is_thin_run_config,
    materialize_run_config,
)

__all__ = [
    "TrainingConfig",
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "WandBConfig",
    "load_config",
    "load_config_as_dict",
    "ConfigCompilerError",
    "CompiledRunBundle",
    "build_compilation_bundle",
    "compile_run_config",
    "is_thin_run_config",
    "materialize_run_config",
]
