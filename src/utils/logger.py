"""
结构化日志系统 (Structured Logging System)

设计目标：
    替代散落的 print() 和基础 logging，提供：
    - 结构化输出（JSON/Console）
    - 上下文传播（自动附加 run_id, epoch 等）
    - 统一格式
    - 更好的可读性和可搜索性

背景：
    当前问题：
    - print() 和 logging.info() 混用
    - 日志格式不统一
    - 难以追踪上下文（哪个 epoch？哪个 run？）
    - 无法结构化查询

解决方案：
    使用 structlog 提供统一的、结构化的日志接口。

使用示例：
    # 基础用法
    logger = get_logger("trainer")
    logger.info("training_started", epochs=100, lr=0.001)
    
    # 上下文绑定
    logger = logger.bind(run_id="exp_001", epoch=1)
    logger.info("epoch_completed", acc=0.95)
    
    # 输出（Console 模式）:
    [2024-01-31 16:10:00] [INFO] training_started epochs=100 lr=0.001
    [2024-01-31 16:15:00] [INFO] epoch_completed run_id=exp_001 epoch=1 acc=0.95
    
    # 输出（JSON 模式）:
    {"event": "training_started", "level": "info", "timestamp": "...", "epochs": 100, "lr": 0.001}
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from structlog.types import Processor


# ============================================================================
# 日志处理器配置
# ============================================================================

def add_app_name(logger, method_name, event_dict):
    """Add app name to all log records"""
    from .constants import APP_NAME
    event_dict["app"] = APP_NAME
    return event_dict


def add_log_level_color(logger, method_name, event_dict):
    """
    为日志级别添加颜色（仅 Console 模式）
    
    INFO: 绿色
    WARNING: 黄色
    ERROR: 红色
    """
    level = event_dict.get("level", "").upper()
    
    colors = {
        "DEBUG": "\033[36m",    # 青色
        "INFO": "\033[32m",     # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",    # 红色
        "CRITICAL": "\033[35m", # 紫色
    }
    
    reset = "\033[0m"
    
    if level in colors:
        event_dict["level"] = f"{colors[level]}{level}{reset}"
    
    return event_dict


# ============================================================================
# 日志配置
# ============================================================================

def configure_logging(
    log_level: str = "INFO",
    log_format: str = "console",  # "console" or "json"
    log_file: Optional[Path] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    配置全局日志系统
    
    参数:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_format: 日志格式 ("console" 或 "json")
        log_file: 日志文件路径（可选）
        context: 全局上下文（如 run_id）
        
    示例:
        # Console 模式（开发）
        configure_logging(log_level="DEBUG", log_format="console")
        
        # JSON 模式（生产）
        configure_logging(
            log_level="INFO",
            log_format="json",
            log_file=Path("runs/exp_001/train.log"),
            context={"run_id": "exp_001"}
        )
    """
    # 设置标准库 logging 级别
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # 构建处理器链
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,  # 合并上下文变量
        structlog.stdlib.add_log_level,            # 添加日志级别
        structlog.stdlib.add_logger_name,          # 添加 logger 名称
        structlog.processors.TimeStamper(fmt="iso"),  # 添加时间戳
        add_app_name,                              # 添加应用名称
    ]
    
    # 根据格式选择渲染器
    if log_format == "json":
        # JSON 格式（生产环境）
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Console 格式（开发环境）
        processors.extend([
            add_log_level_color,  # 添加颜色
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        ])
    
    # 配置 structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # 文件始终使用 JSON 格式
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    # 绑定全局上下文
    if context:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**context)


# ============================================================================
# Logger 获取
# ============================================================================

def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """
    获取 logger 实例
    
    参数:
        name: Logger 名称（通常使用模块名）
        
    返回:
        BoundLogger: structlog logger 实例
        
    示例:
        logger = get_logger("trainer")
        logger.info("model_built", model_type="resnet50")
    """
    return structlog.get_logger(name)


# ============================================================================
# 上下文管理器
# ============================================================================

class LogContext:
    """
    日志上下文管理器
    
    自动绑定和解绑上下文变量。
    
    示例:
        with LogContext(epoch=1, phase="train"):
            logger.info("batch_processed", loss=0.5)
            # 自动附加 epoch=1, phase="train"
    """
    
    def __init__(self, **context):
        """
        初始化上下文
        
        参数:
            **context: 要绑定的上下文键值对
        """
        self.context = context
        self.tokens = []
    
    def __enter__(self):
        """进入上下文"""
        for key, value in self.context.items():
            token = structlog.contextvars.bind_contextvars(**{key: value})
            self.tokens.append((key, token))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        # 清除绑定的上下文
        for key, _ in self.tokens:
            structlog.contextvars.unbind_contextvars(key)


# ============================================================================
# 便捷函数
# ============================================================================

def log_training_start(logger, config: Dict[str, Any]) -> None:
    """
    记录训练开始
    
    参数:
        logger: Logger 实例
        config: 训练配置
    """
    logger.info(
        "training_started",
        model=config.get("model", {}).get("type"),
        epochs=config.get("train", {}).get("epochs"),
        batch_size=config.get("train", {}).get("batch_size"),
        lr=config.get("optimizer", {}).get("lr"),
    )


def log_epoch_metrics(
    logger,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
) -> None:
    """
    记录 epoch 指标
    
    参数:
        logger: Logger 实例
        epoch: 当前 epoch
        train_loss: 训练损失
        train_acc: 训练准确率
        val_loss: 验证损失
        val_acc: 验证准确率
    """
    logger.info(
        "epoch_completed",
        epoch=epoch,
        train_loss=round(train_loss, 4),
        train_acc=round(train_acc, 4),
        val_loss=round(val_loss, 4),
        val_acc=round(val_acc, 4),
    )


def log_model_saved(logger, path: Path, metric: str, value: float) -> None:
    """
    记录模型保存
    
    参数:
        logger: Logger 实例
        path: 模型路径
        metric: 指标名称
        value: 指标值
    """
    logger.info(
        "model_saved",
        path=str(path),
        metric=metric,
        value=round(value, 4),
    )


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    # 示例 1: Console 模式（开发）
    print("=" * 60)
    print("示例 1: Console 模式")
    print("=" * 60)
    
    configure_logging(log_level="INFO", log_format="console")
    logger = get_logger("example")
    
    logger.info("application_started", version="1.0.0")
    logger.debug("debug_message", detail="This won't show at INFO level")
    logger.warning("low_memory", available_gb=2.5)
    logger.error("training_failed", reason="CUDA out of memory")
    
    # 示例 2: 上下文绑定
    print("\n" + "=" * 60)
    print("示例 2: 上下文绑定")
    print("=" * 60)
    
    with LogContext(run_id="exp_001", epoch=1):
        logger.info("epoch_started")
        logger.info("batch_processed", batch=10, loss=0.5)
    
    # 上下文外，不再有 run_id 和 epoch
    logger.info("training_completed")
    
    # 示例 3: JSON 模式（生产）
    print("\n" + "=" * 60)
    print("示例 3: JSON 模式")
    print("=" * 60)
    
    configure_logging(
        log_level="INFO",
        log_format="json",
        context={"run_id": "exp_002"}
    )
    logger = get_logger("production")
    
    logger.info("model_loaded", model_type="resnet50", params=25_000_000)
    logger.info("training_started", epochs=100, lr=0.001)
    
    print("\n✓ 日志系统示例完成")
