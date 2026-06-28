"""Component builder utilities for training/evaluation pipelines.

This module centralizes model/optimizer/scheduler/dataloader construction so
Trainer code can focus on orchestration instead of model-specific wiring.
"""

import os
import inspect
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..schemas import TrainingConfig
from ..utils import (
    MODELS,
    LOSSES,
    DATASETS,
    TRANSFORMS,
    ModelStrategy,
    build_strategy,
    validate_strategy_config,
    ModelBuildError,
    DataLoadError,
    OptimizerError,
)
class ComponentBuilder:
    """
    Component builder responsibilities:
    - Build model/optimizer/dataloaders from config.
    - Provide a unified construction interface.
    - Stay reusable across different training/evaluation entry points.
    """

    def __init__(self, config: TrainingConfig, device: torch.device, logger: Optional[Any] = None):
        """
        初化构建器

        参数:
            config: 讻配置(Pydantic 模型)
            device: 计算设
            logger: Optional logger instance.
        """
        self.config = config
        self.device = device
        self.logger = logger

        # 缓存已构建的组件
        self._model: Optional[nn.Module] = None
        self._strategy: Optional[ModelStrategy] = None
        self._num_classes: Optional[int] = None

    @staticmethod
    def resolve_loader_batch_sizes(train_cfg: Dict[str, Any]) -> Tuple[int, int, int]:
        """Resolve train/val/test batch sizes with stable fallback rules."""
        batch_size = int(train_cfg.get("batch_size", 32))
        eval_batch_size = train_cfg.get("eval_batch_size", None)
        default_eval_bs = max(1, batch_size // 2)

        val_batch_size = int(
            train_cfg.get(
                "val_batch_size",
                eval_batch_size if eval_batch_size is not None else default_eval_bs,
            )
        )
        test_batch_size = int(
            train_cfg.get(
                "test_batch_size",
                eval_batch_size if eval_batch_size is not None else default_eval_bs,
            )
        )
        return batch_size, val_batch_size, test_batch_size

    def build_model(self) -> Tuple[nn.Module, ModelStrategy]:
        """
        Build model and strategy.
        流程:
            1. 验证策略配置
            2. 注入 num_classes
            3. Instantiate model class
            5. Move model to target device
        返回:
            (model, strategy): built model and strategy
        抛出:
            ModelBuildError: 模型构建失败
        """
        if self._model is not None and self._strategy is not None:
            return self._model, self._strategy

        try:
            model_cfg = self.config.model.model_dump()

            # 1. 验证策略配置
            # 先校验“模型结构”和“策略类型”是否匹配。
            validate_strategy_config(model_cfg)

            # 2. 注入 num_classes
            # 把数据集推断出来的类别数注入 head。
            # 因此 dataloader 必须先于 model 构建。
            if "head" in model_cfg and self._num_classes is not None:
                model_cfg["head"]["num_classes"] = self._num_classes

            # 3. ʵģ
            # 从注册表里取出模型类并实例化。
            model_class = MODELS.get(model_cfg["type"])
            if model_class is None:
                raise ModelBuildError(
                    f"Model type not found in registry: {model_cfg['type']}", model_type=model_cfg["type"]
                )

            # ˶ type/strategyݹ None ֵ
            # 过滤掉只用于装配的字段，保留真正的构造参数。
            model_kwargs = self._filter_build_kwargs(model_cfg, exclude_keys=["type", "strategy"])
            model = model_class(**model_kwargs)

            # 4. 构建策略
            # 策略层负责“如何从输出取预测”和“如何计算正确数”。
            strategy = build_strategy(model_cfg, model)

            # 5. ƶ豸
            # 最后再把模型移动到目标设备。
            model.to(self.device)

            if self.logger:
                self.logger.info(
                    "model_built",
                    model_type=model_cfg["type"],
                    strategy=model_cfg.get("strategy", "classification"),
                    device=str(self.device),
                )

            # 缓存
            self._model = model
            self._strategy = strategy

            return model, strategy

        except Exception as e:
            if isinstance(e, ModelBuildError):
                raise
            raise ModelBuildError(
                f"Failed to build model: {e}", model_type=self.config.model.type, original_exception=e
            )

    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Build optimizer.
        Args:
            model: model instance.
        Returns:
            optimizer: optimizer instance.
        Raises:
            OptimizerError: raised when optimizer construction fails.
        """
        try:
            # 处理 Pydantic 对象到dict
            if hasattr(self.config.optimizer, "model_dump"):
                opt_cfg = self.config.optimizer.model_dump()
            elif isinstance(self.config.optimizer, dict):
                opt_cfg = self.config.optimizer.copy()
            else:
                opt_cfg = vars(self.config.optimizer)

            opt_type = opt_cfg.pop("type")
            lr = opt_cfg.get("lr")  # 保存 lr 用于日志

            # 筛需要优化的参数
            params = [p for p in model.parameters() if p.requires_grad]

            # Ż͹˲
            if opt_type.lower() == "adamw":
                # AdamW ֧ momentum / nesterov None
                valid_params = {k: v for k, v in opt_cfg.items() if k not in ["momentum", "nesterov"] and v is not None}
                optimizer = optim.AdamW(params, **valid_params)
            elif opt_type.lower() == "adam":
                # Adam ֧ momentum / nesterov None
                valid_params = {k: v for k, v in opt_cfg.items() if k not in ["momentum", "nesterov"] and v is not None}
                optimizer = optim.Adam(params, **valid_params)
            elif opt_type.lower() == "sgd":
                # SGD  None
                valid_params = {k: v for k, v in opt_cfg.items() if v is not None}
                optimizer = optim.SGD(params, **valid_params)
            else:
                raise OptimizerError(f"Unknown optimizer type: {opt_type}", optimizer_type=opt_type)

            if self.logger:
                self.logger.info("optimizer_built", optimizer_type=opt_type, lr=lr, num_params=len(params))

            return optimizer

        except Exception as e:
            if isinstance(e, OptimizerError):
                raise
            raise OptimizerError(
                f"Failed to build optimizer: {e}",
                optimizer_type=getattr(self.config.optimizer, "type", "unknown"),
                original_exception=e,
            )

    def build_scheduler(self, optimizer: optim.Optimizer, total_epochs: Optional[int] = None) -> Optional[Any]:
        """
        构建学习率调度器

        参数:
            optimizer: optimizer instance; total_epochs: total training epochs (for cosine schedule)

        返回:
            scheduler: scheduler instance if configured
        """
        if self.config.scheduler is None:
            return None

        try:
            sched_cfg = self.config.scheduler.model_dump()
            sched_type = sched_cfg.pop("type")

            if total_epochs is None:
                total_epochs = self.config.train.epochs

            #
            if sched_type.lower() == "cosine":
                from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

                warmup_epochs = sched_cfg.get("warmup_epochs", 0)
                eta_min = sched_cfg.get("min_lr", 0)
                T_max = max(1, total_epochs - warmup_epochs)
                cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
                if warmup_epochs > 0:
                    # LinearLR: linearly warm up from start_factor * base_lr to base_lr
                    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
                    scheduler = SequentialLR(
                        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
                    )
                else:
                    scheduler = cosine_scheduler
            elif sched_type.lower() == "step":
                from torch.optim.lr_scheduler import StepLR

                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=sched_cfg.get("step_size", 30), gamma=sched_cfg.get("gamma", 0.1)
                )
            else:
                if self.logger:
                    self.logger.warning("unknown_scheduler", scheduler_type=sched_type)
                return None

            if self.logger:
                self.logger.info("scheduler_built", scheduler_type=sched_type)

            return scheduler

        except Exception as e:
            if self.logger:
                self.logger.error("scheduler_build_failed", error=str(e))
            return None

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build train/val/test dataloaders.
        返回:
            (train_loader, val_loader, test_loader): dataloader tuple
        抛出:
            DataLoadError: 数据加载失败
        """
        try:
            try:
                from ..data.loader import build_dataloader
            except ImportError as exc:
                raise DataLoadError(
                    "No dataset adapter is installed. Add src/data/loader.py for the new dataset."
                ) from exc
            # 处理 Pydantic 对象
            # 先把配置转成 dict，便于统一处理。
            if hasattr(self.config.data, "model_dump"):
                data_cfg = self.config.data.model_dump()
            else:
                data_cfg = self.config.data

            if hasattr(self.config.train, "model_dump"):
                train_cfg = self.config.train.model_dump()
            else:
                train_cfg = self.config.train

            # 兼旧配罠式：root vs train_path/val_path/test_path
            # 同时兼容两种数据描述方式：
            # 1. root/train, root/val, root/test
            # 2. 显式给 train_csv / val_csv / test_csv 或 train_path / val_path / test_path
            if "root" in data_cfg:
                # Legacy format: use root + train/val/test subdirectories
                root = data_cfg["root"]
                train_path = os.path.join(root, "train")
                val_path = os.path.join(root, "val")
                test_path = os.path.join(root, "test")
            else:
                # 新格式：直接指定跾
                train_path = data_cfg.get("train_path")
                val_path = data_cfg.get("val_path")
                test_path = data_cfg.get("test_path")

            # 构建 transforms
            # 兼旧格式：train_transform vs train_transforms
            # 构建 train / val / test 三套 transform。
            train_transform_cfg = data_cfg.get("train_transforms") or data_cfg.get("train_transform")
            val_transform_cfg = (
                data_cfg.get("val_transforms") or data_cfg.get("eval_transform") or data_cfg.get("val_transform")
            )
            test_transform_cfg = data_cfg.get("test_transforms") or data_cfg.get("test_transform") or val_transform_cfg

            train_transforms = self._build_transform_from_config(train_transform_cfg)
            val_transforms = self._build_transform_from_config(val_transform_cfg)
            test_transforms = self._build_transform_from_config(test_transform_cfg)

            # 构建 datasets
            # 红茶主线一般走 csv_dataset：CSV 明确给出图片路径和标签。
            dataset_type = data_cfg.get("type") or data_cfg.get("dataset_type", "image_folder")

            # 🔧 FIX: 提取class_to_idx配置（用于盖默认排序）
            class_to_idx = data_cfg.get("class_to_idx", None)

            dataset_cls = DATASETS.get(dataset_type)
            if dataset_cls is None:
                raise DataLoadError(
                    f"Dataset type not found in registry: {dataset_type}", data_path=data_cfg.get("root")
                )

            if dataset_type == "csv_dataset":
                # CSV 模式下，dataset 一般返回：
                # image: Tensor
                # label: int
                # path : str
                # CSV mode: consume explicit CSV split files.
                # Backward-compatible keys:
                # - train_csv / val_csv / test_csv
                # - train_path / val_path / test_path
                train_csv = data_cfg.get("train_csv") or train_path
                val_csv = data_cfg.get("val_csv") or val_path
                test_csv = data_cfg.get("test_csv") or test_path

                if not train_csv or not val_csv or not test_csv:
                    raise DataLoadError(
                        "csv_dataset requires train/val/test CSV paths.", data_path=str(data_cfg.get("root", ""))
                    )

                root_dir = data_cfg.get("root_dir", None)
                image_col = data_cfg.get("image_col", "image_path")
                label_col = data_cfg.get("label_col", "label")
                class_col = data_cfg.get("class_col", None)
                extra_target_cols = data_cfg.get("extra_target_cols", None)

                train_dataset = dataset_cls(
                    csv_file=train_csv,
                    root_dir=root_dir,
                    transform=train_transforms,
                    image_col=image_col,
                    label_col=label_col,
                    class_col=class_col,
                    extra_target_cols=extra_target_cols,
                )
                val_dataset = dataset_cls(
                    csv_file=val_csv,
                    root_dir=root_dir,
                    transform=val_transforms,
                    image_col=image_col,
                    label_col=label_col,
                    class_col=class_col,
                    extra_target_cols=extra_target_cols,
                )
                test_dataset = dataset_cls(
                    csv_file=test_csv,
                    root_dir=root_dir,
                    transform=test_transforms,
                    image_col=image_col,
                    label_col=label_col,
                    class_col=class_col,
                    extra_target_cols=extra_target_cols,
                )
            else:
                train_dataset = dataset_cls(root=train_path, transform=train_transforms, class_to_idx=class_to_idx)

                val_dataset = dataset_cls(root=val_path, transform=val_transforms, class_to_idx=class_to_idx)

                test_dataset = dataset_cls(root=test_path, transform=test_transforms, class_to_idx=class_to_idx)

            # 保存 num_classes
            # 从训练集推断类别数，供 model/head/loss 复用。
            self._num_classes = len(train_dataset.classes)

            # 构建 dataloaders
            # 默认 collate 后：
            # 普通图像是 [B, C, H, W]
            # 滑窗/五裁剪图像会是 [B, N, C, H, W]
            batch_size, val_batch_size, test_batch_size = self.resolve_loader_batch_sizes(train_cfg)
            # Keep explicit 0 from train_cfg; fallback only when key is missing/None.
            num_workers = train_cfg.get("num_workers", None)
            if num_workers is None:
                num_workers = data_cfg.get("num_workers", 0)

            train_loader = build_dataloader(
                train_dataset,
                batch_size=batch_size,
                is_train=True,
                num_workers=num_workers,
                enable_weighted_sampler=train_cfg.get("weighted_sampler", False),
            )

            val_loader = build_dataloader(
                val_dataset, batch_size=val_batch_size, is_train=False, num_workers=num_workers
            )

            test_loader = build_dataloader(
                test_dataset, batch_size=test_batch_size, is_train=False, num_workers=num_workers
            )

            if self.logger:
                self.logger.info(
                    "dataloaders_built",
                    num_classes=self._num_classes,
                    train_samples=len(train_dataset),
                    val_samples=len(val_dataset),
                    test_samples=len(test_dataset),
                )

            return train_loader, val_loader, test_loader

        except Exception as e:
            if isinstance(e, DataLoadError):
                raise
            # ȡ train_pathͬ
            train_path = None
            if hasattr(self.config.data, "train_path"):
                train_path = self.config.data.train_path
            elif hasattr(self.config.data, "root"):
                train_path = self.config.data.root
            elif isinstance(self.config.data, dict):
                train_path = self.config.data.get("train_path") or self.config.data.get("root")

            raise DataLoadError(f"Failed to build dataloaders: {e}", data_path=train_path, original_exception=e)

    def build_loss(self) -> nn.Module:
        """
        构建损失函数

        返回:
            loss_fn: 损失函数实例
        """
        # FIX:
        # 优先读取顶层 loss 配置（新格式），若不存在则回退到 model.loss（旧格式）。
        # 这样可以保证历史 YAML 不需要迁移也能继续运行。
        # 优先读取顶层 loss 配置；如果没有，再兼容旧版 model.loss 写法。
        if hasattr(self.config, "loss") and self.config.loss is not None:
            loss_cfg = self.config.loss.model_dump() if hasattr(self.config.loss, "model_dump") else self.config.loss
            loss_type = loss_cfg.get("type", "cross_entropy")
        else:
            # Fallback: 兼容旧配置结构
            model_cfg = (
                self.config.model.model_dump() if hasattr(self.config.model, "model_dump") else self.config.model
            )
            loss_cfg = model_cfg.get("loss", {"type": "cross_entropy"})
            loss_type = loss_cfg.get("type", "cross_entropy")

        loss_class = LOSSES.get(loss_type)
        if loss_class is None:
            # 使用默损失
            loss_fn = nn.CrossEntropyLoss()
            if self.logger:
                self.logger.warning("loss_not_found_using_default", loss_type=loss_type)
        else:
            # 关键兼容逻辑:
            # 一些损失（如 CrossEntropyWithCLOCLoss）要求 n_classes/num_classes，
            # 但用户通常不会在 YAML 手动写这个值。
            # 这里自动把数据集推断得到的类别数注入到 loss 参数中，避免训练时因缺参报错。
            # 同时我们基于构造函数签名做参数过滤，确保不会把无关字段传给 loss，避免破坏其他损失。
            # 某些 loss 需要 num_classes / n_classes，但用户通常不会手写。
            # 这里自动从数据集推断值注入进去。
            init_params = inspect.signature(loss_class.__init__).parameters
            if "n_classes" in init_params and "n_classes" not in loss_cfg and self._num_classes is not None:
                loss_cfg["n_classes"] = self._num_classes
            if "num_classes" in init_params and "num_classes" not in loss_cfg and self._num_classes is not None:
                loss_cfg["num_classes"] = self._num_classes

            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values())
            loss_kwargs = {}
            for k, v in loss_cfg.items():
                if k == "type" or v is None:
                    continue
                if k in init_params or has_kwargs:
                    loss_kwargs[k] = v

            loss_fn = loss_class(**loss_kwargs)

        if self.logger:
            self.logger.info("loss_built", loss_type=loss_type)

        return loss_fn

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _build_transform_from_config(self, transform_config: Any) -> Any:
        """
        从配罞建transform

        攌两格式:
        1. 旧格式：{"type": "default_train", "image_size": 224}
        2. 新格式：[{"type": "resize", "size": 224}, {"type": "to_tensor"}]
        """
        from torchvision import transforms as T

        if transform_config is None:
            return None

        # 旧格式：单个 dict
        if isinstance(transform_config, dict):
            t_type = transform_config.get("type")
            if t_type:
                transform = TRANSFORMS.get(t_type)
                if transform:
                    # 过滤掉type 字
                    kwargs = {k: v for k, v in transform_config.items() if k != "type"}
                    return transform(**kwargs) if kwargs else transform()
            return None

        # 新格式：list of dicts
        if isinstance(transform_config, list):
            return self._build_transforms(transform_config)

        return None

    def _build_transforms(self, transform_configs: list) -> Any:
        """构建数据增强 transforms"""
        from torchvision import transforms as T

        transform_list = []
        for cfg in transform_configs:
            if isinstance(cfg, dict):
                cfg_copy = cfg.copy()  # 避免俔原配置
                t_type = cfg_copy.pop("type")
                transform = TRANSFORMS.get(t_type)
                if transform:
                    transform_list.append(transform(**cfg_copy))
            else:
                # ַʽ
                transform = TRANSFORMS.get(cfg)
                if transform:
                    transform_list.append(transform())

        return T.Compose(transform_list) if transform_list else None

    def _filter_build_kwargs(self, config: Dict[str, Any], exclude_keys: list) -> Dict[str, Any]:
        """
        Filter model construction kwargs.
        Remove excluded top-level keys and recursively drop None values.
        """

        def filter_dict(d, is_top_level=False):
            if isinstance(d, dict):
                return {
                    k: filter_dict(v, is_top_level=False)
                    for k, v in d.items()
                    if (not is_top_level or k not in exclude_keys) and v is not None
                }
            return d

        return filter_dict(config, is_top_level=True)

    @property
    def num_classes(self) -> Optional[int]:
        """Return inferred number of classes from the training dataset."""
        return self._num_classes
