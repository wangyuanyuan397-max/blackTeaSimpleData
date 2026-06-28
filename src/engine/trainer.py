"""
重构后的 Trainer (Refactored Trainer)

设计目标(
    解决"不同模型需要不(Trainer"的问题(
    
核心改进(
    1. 依赖注入 - 通过构造函数注(ComponentBuilder, Evaluator, HookManager
    2. 单一职责 - Trainer 只负责训练流程协(
    3. Hook 机制 - 提供扩展点，无需修改 Trainer
    4. 100% 类型注解 - 完整的类型提(
    
使用示例:
    # 方式 1: 自动构建(向后兼容)
    trainer = Trainer(config)
    trainer.train()
    
    # 方式 2: 依赖注入(推荐)
    builder = ComponentBuilder(config, device, logger)
    evaluator = Evaluator(model, strategy, device, logger)
    hook_manager = HookManager(logger)
    
    trainer = Trainer(
        config=config,
        builder=builder,
        evaluator=evaluator,
        hook_manager=hook_manager
    )
    trainer.train()
"""

import os
import hashlib
import random
import inspect
import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from ..schemas import TrainingConfig, load_config
from ..utils import configure_logging, get_logger, ModelStrategy
from .builder import ComponentBuilder
from .evaluator import Evaluator
from .hooks import (
    HookManager,
    WandBHook,
    CheckpointHook,
    EarlyStoppingHook,
    LRSchedulerHook,
    infer_metric_mode,
)


class Trainer:
    """
    重构后的训练(
    
    职责(
        - 协调训练流程
        - 管理训练循环
        - 触发 Hook 事件
        
    设计模式(
        - 依赖注入：通过构造函数注入依(
        - 观察者模式：通过 Hook 提供扩展(
        - Builder 模式：通过 ComponentBuilder 构建组件
    """
    
    def __init__(
        self,
        config: Union[TrainingConfig, dict, str, Path],
        builder: Optional[ComponentBuilder] = None,
        evaluator: Optional[Evaluator] = None,
        hook_manager: Optional[HookManager] = None,
        logger: Optional[Any] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化Trainer
        
        参数:
            config: 训练配置(支持多种格式)
                - TrainingConfig: Pydantic 配置对象
                - dict: 配置字典
                - str/Path: 配置文件路径
            builder: 组件构建器(可选，默认自动创建)
            evaluator: 评估器(可选，默认自动创建)
            hook_manager: Hook 管理器(可选，默认自动创建(
            logger: 日志器(可选，默认自动创建(
            device: 计算设备(可选，默认自动检测)
        """
        # 1. 处理配置
        self.config = self._load_config(config)
        
        # 2. 设置随机种子(确保可复现性)
        self._set_random_seed()
        
        # 3. 设置设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 4. 设置输出目录
        self.output_dir = self._setup_output_dir()
        
        # 5. 设置日志
        self.logger = logger or self._setup_logger()
        self.logger.info("trainer_initialized", output_dir=str(self.output_dir), device=str(self.device))
        
        # 6. 保存配置
        self._save_config()
        
        # 7. 构建组件(使用依赖注入或自动创建(
        self.builder = builder or ComponentBuilder(self.config, self.device, self.logger)
        
        # 构建数据加载(
        # 先构建 dataloader，再构建 model。
        # 因为 num_classes 需要先从数据集推断出来，后续会注入 head / loss。
        self.train_loader, self.val_loader, self.test_loader = self.builder.build_dataloaders()
        
        # 构建模型和策(
        # model 负责前向，strategy 负责从输出里取预测和计算正确数。
        self.model, self.strategy = self.builder.build_model()
        
        # 保存模型结构
        self._save_model_structure()
        
        # 构建损失函数，并确保其 buffer（如 classes、ordinal_map 等）随 device 移动
        # loss 也在这里统一构建，确保与 model / num_classes 保持一致。
        self.loss_fn = self.builder.build_loss()
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn = self.loss_fn.to(self.device)

        self._init_staged_training()

        
        # 构建优化(
        self.optimizer = self.builder.build_optimizer(self.model)
        
        # 构建调度(
        self.scheduler = self.builder.build_scheduler(self.optimizer)
        
        # 7. 创建评估(
        tta_mode = getattr(self.config.train, "tta_mode", "mean") if hasattr(self.config, "train") else "mean"
        tta_topk = getattr(self.config.train, "tta_topk", 0) if hasattr(self.config, "train") else 0
        tta_compare = getattr(self.config.train, "tta_compare", False) if hasattr(self.config, "train") else False
        tta_compare_modes = getattr(self.config.train, "tta_compare_modes", None) if hasattr(self.config, "train") else None
        if isinstance(self.config.train, dict):
            tta_mode = self.config.train.get("tta_mode", tta_mode)
            tta_topk = self.config.train.get("tta_topk", tta_topk)
            tta_compare = self.config.train.get("tta_compare", tta_compare)
            tta_compare_modes = self.config.train.get("tta_compare_modes", tta_compare_modes)
        # Evaluator 负责验证/测试阶段的前向与指标统计，TTA 逻辑也放在这里。
        self.evaluator = evaluator or Evaluator(
            self.model, self.strategy, self.device, self.logger,
            tta_mode=tta_mode, tta_topk=tta_topk,
            tta_compare=tta_compare, tta_compare_modes=tta_compare_modes
        )
        
        # 8. 设置 Hook 管理(
        self.hook_manager = hook_manager or self._setup_hooks()
        
        # 9. 初始化训练状(
        self.current_epoch = 0
        self.best_val_acc = 0.0
        # 2026-03-20 Codex: 保留有序指标与综合分数历史，便于 fold4 筛选和训练复盘。
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_mae": [],
            "val_qwk": [],
            "monitor_score": [],
        }
        
        # 10. 梯度累加设置
        # 如果 config.train.accumulation_steps 存在则使用，否则默认为 1
        # 梯度累加可以让“小显存 + 大等效 batch”同时成立。
        self.accumulation_steps = 1
        if hasattr(self.config.train, "accumulation_steps"):
            self.accumulation_steps = self.config.train.accumulation_steps or 1
        elif isinstance(self.config.train, dict):
            self.accumulation_steps = self.config.train.get("accumulation_steps", 1)
        
        if self.accumulation_steps > 1:
            self.logger.info(f"gradient_accumulation_enabled", steps=self.accumulation_steps)
    
    def train(self) -> None:
        """
        训练主循(
        
        流程:
            1. 触发 on_train_begin Hook
            2. 循环训练每个 epoch
            3. 触发 on_train_end Hook
        """
        self.hook_manager.trigger("on_train_begin", trainer=self)
        
        try:
            for epoch in range(self.config.train.epochs):
                self.current_epoch = epoch
                self._maybe_transition_staged_training(epoch)
                
                # 训练一个 epoch
                # 先训练一个 epoch，再在验证集上评估。
                train_metrics = self._train_epoch(epoch)
                
                # 验证
                val_metrics = self._validate_epoch(epoch)
                
                # 合并指标
                metrics = {
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_mae": val_metrics.get("mae", 0.0),
                    "val_qwk": val_metrics.get("qwk", 0.0),
                }
                # 2026-03-20 Codex: 允许用组合指标做选模，但默认不改变原始 baseline 行为。
                monitor_score = self._compute_monitor_score(metrics)
                if monitor_score is not None:
                    metrics["monitor_score"] = monitor_score
                
                # 记录历史
                for key, value in metrics.items():
                    self.history.setdefault(key, [])
                    self.history[key].append(value)
                
                # 触发 epoch_end Hook
                self.hook_manager.trigger("on_epoch_end", trainer=self, epoch=epoch, metrics=metrics)
                
                # 更新最佳准确率
                if val_metrics["accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["accuracy"]
                
                # 检查早停 ──────────────────────────────────────────────────
                # FIX: 用 flag 变量跳出外层 epoch 循环，而不是 break 只退出内层
                should_stop_training = False
                for hook in self.hook_manager.hooks:
                    if hasattr(hook, 'should_stop') and hook.should_stop:
                        self.logger.info("early_stopping", epoch=epoch)
                        should_stop_training = True
                        break
                if should_stop_training:
                    break  # 退出外层 epoch 循环
        
        finally:
            # finally 在正常结束/早停/未捕获异常时均会执行
            # 注意：若进程被外部强制 kill (SIGKILL/TerminateProcess) 则不会执行
            self.hook_manager.trigger("on_train_end", trainer=self)
            
            # 训练结束后按配置决定是否运行错误分析
            enable_error_analysis = bool(
                self._get_train_option(
                    "enable_error_analysis",
                    self._get_train_option("error_analysis_enabled", True),
                )
            )
            if enable_error_analysis:
                self._run_error_analysis()
            else:
                self.logger.info("error_analysis_skipped", reason="disabled_by_config")
    
    def _get_train_option(self, key: str, default: Any = None) -> Any:
        """Read train.* options from both dict-style and object-style configs."""
        train_cfg = getattr(self.config, "train", None)
        if isinstance(train_cfg, dict):
            return train_cfg.get(key, default)
        if train_cfg is not None and hasattr(train_cfg, key):
            value = getattr(train_cfg, key)
            return default if value is None else value
        return default

    def _get_staged_training_config(self) -> Optional[Dict[str, Any]]:
        cfg = self._get_train_option("staged_training", None)
        return cfg if isinstance(cfg, dict) else None

    def _iter_scheduler_hooks(self):
        for hook in getattr(self.hook_manager, "hooks", []):
            if hasattr(hook, "scheduler"):
                yield hook

    def _set_optimizer_lr(self, lr: float) -> None:
        optimizer_cfg = getattr(self.config, "optimizer", None)
        if optimizer_cfg is None:
            return
        if isinstance(optimizer_cfg, dict):
            optimizer_cfg["lr"] = float(lr)
        elif hasattr(optimizer_cfg, "lr"):
            setattr(optimizer_cfg, "lr", float(lr))

    def _set_scheduler_warmup_epochs(self, warmup_epochs: int) -> None:
        scheduler_cfg = getattr(self.config, "scheduler", None)
        if scheduler_cfg is None:
            return
        if isinstance(scheduler_cfg, dict):
            scheduler_cfg["warmup_epochs"] = int(warmup_epochs)
        elif hasattr(scheduler_cfg, "warmup_epochs"):
            setattr(scheduler_cfg, "warmup_epochs", int(warmup_epochs))

    def _rebuild_optimizer_and_scheduler(
        self,
        lr: float,
        total_epochs: int,
        warmup_epochs: Optional[int] = None,
    ) -> None:
        self._set_optimizer_lr(lr)
        if warmup_epochs is not None:
            self._set_scheduler_warmup_epochs(warmup_epochs)
        self.optimizer = self.builder.build_optimizer(self.model)
        self.scheduler = self.builder.build_scheduler(self.optimizer, total_epochs=total_epochs)
        for hook in self._iter_scheduler_hooks():
            hook.scheduler = self.scheduler

    def _freeze_backbone_keep_plugins_trainable(self) -> None:
        backbone = self.model.backbone
        for param in backbone.parameters():
            param.requires_grad_(False)

        plugin_modules = getattr(backbone, "plugin_modules", None)
        if plugin_modules is not None:
            for module in plugin_modules:
                for param in module.parameters():
                    param.requires_grad_(True)

        head = getattr(self.model, "head", None)
        if head is not None:
            for param in head.parameters():
                param.requires_grad_(True)

    def _unfreeze_backbone_for_joint_finetune(self) -> None:
        backbone = self.model.backbone
        for param in backbone.parameters():
            param.requires_grad_(True)
        if hasattr(backbone, "_freeze_batch_norm_layers"):
            backbone._freeze_batch_norm_layers()

    def _init_staged_training(self) -> None:
        self.staged_training_cfg = self._get_staged_training_config()
        self.staged_training_active = False
        self.staged_training_phase = "full"

        cfg = self.staged_training_cfg
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return

        freeze_backbone_epochs = int(cfg.get("freeze_backbone_epochs", 0) or 0)
        stage1_lr = float(cfg.get("stage1_lr", 0.0) or 0.0)
        stage2_lr = float(cfg.get("stage2_lr", 0.0) or 0.0)
        if freeze_backbone_epochs <= 0:
            self.logger.warning("staged_training_disabled", reason="freeze_backbone_epochs<=0")
            return
        if stage1_lr <= 0 or stage2_lr <= 0:
            self.logger.warning("staged_training_disabled", reason="invalid_stage_lr")
            return

        backbone = getattr(self.model, "backbone", None)
        plugin_modules = getattr(backbone, "plugin_modules", None)
        if backbone is None or plugin_modules is None or len(plugin_modules) == 0:
            self.logger.warning("staged_training_disabled", reason="plugin_modules_not_found")
            return

        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        self.stage1_warmup_epochs = int(cfg.get("stage1_warmup_epochs", min(2, max(freeze_backbone_epochs - 1, 0))))
        self.stage2_warmup_epochs = int(cfg.get("stage2_warmup_epochs", 3))

        self._freeze_backbone_keep_plugins_trainable()
        self.staged_training_active = True
        self.staged_training_phase = "frozen_backbone"
        self._set_optimizer_lr(self.stage1_lr)
        self._set_scheduler_warmup_epochs(self.stage1_warmup_epochs)
        self.logger.info(
            "staged_training_initialized",
            freeze_backbone_epochs=self.freeze_backbone_epochs,
            stage1_lr=self.stage1_lr,
            stage2_lr=self.stage2_lr,
            stage1_warmup_epochs=self.stage1_warmup_epochs,
            stage2_warmup_epochs=self.stage2_warmup_epochs,
        )

    def _maybe_transition_staged_training(self, epoch: int) -> None:
        if not self.staged_training_active:
            return
        if self.staged_training_phase != "frozen_backbone":
            return
        if epoch < self.freeze_backbone_epochs:
            return

        self._unfreeze_backbone_for_joint_finetune()
        remaining_epochs = max(1, int(self.config.train.epochs) - epoch)
        self._rebuild_optimizer_and_scheduler(
            lr=self.stage2_lr,
            total_epochs=remaining_epochs,
            warmup_epochs=min(self.stage2_warmup_epochs, max(remaining_epochs - 1, 0)),
        )
        self.staged_training_phase = "joint_finetune"
        self.logger.info(
            "staged_training_transition",
            epoch=epoch,
            new_phase=self.staged_training_phase,
            stage2_lr=self.stage2_lr,
            remaining_epochs=remaining_epochs,
        )

    @staticmethod
    def _normalize_metric_name(metric: Optional[str]) -> str:
        metric_name = str(metric or "val_acc").strip().lower()
        aliases = {
            "acc": "val_acc",
            "accuracy": "val_acc",
            "val_accuracy": "val_acc",
            "loss": "val_loss",
            "val_mae": "val_mae",
            "mae": "val_mae",
            "val_qwk": "val_qwk",
            "qwk": "val_qwk",
            "score": "monitor_score",
        }
        return aliases.get(metric_name, metric_name)

    def _get_selection_weights(self) -> Dict[str, float]:
        raw = self._get_train_option("selection_weights", None)
        if not isinstance(raw, dict):
            return {}

        weights: Dict[str, float] = {}
        for key, value in raw.items():
            try:
                weights[self._normalize_metric_name(key)] = float(value)
            except Exception:
                continue
        return weights

    def _compute_monitor_score(self, metrics: Dict[str, float]) -> Optional[float]:
        # 2026-03-20 Codex: 支持通过加权方式把 val_acc / val_qwk / val_mae 合成单一监控分数。
        weights = self._get_selection_weights()
        if not weights:
            return None

        score = 0.0
        used = 0
        for metric_name, weight in weights.items():
            if metric_name not in metrics:
                continue
            sign = -1.0 if infer_metric_mode(metric_name) == "min" else 1.0
            score += sign * weight * float(metrics[metric_name])
            used += 1

        if used == 0:
            return None
        return float(score)

    @staticmethod
    def _move_extra_targets_to_device(extra_targets: Optional[Dict[str, Any]], device: torch.device):
        if extra_targets is None:
            return None
        moved = {}
        for key, value in extra_targets.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _compute_loss(
        self,
        outputs: Any,
        labels: torch.Tensor,
        extra_targets: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if extra_targets is None:
            return self.loss_fn(outputs, labels)

        forward_fn = getattr(self.loss_fn, "forward", None)
        if forward_fn is None:
            return self.loss_fn(outputs, labels)

        try:
            signature = inspect.signature(forward_fn)
        except (TypeError, ValueError):
            return self.loss_fn(outputs, labels)

        if "extra_targets" in signature.parameters:
            return self.loss_fn(outputs, labels, extra_targets=extra_targets)
        return self.loss_fn(outputs, labels)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一(epoch
        
        参数:
            epoch: 当前 epoch 编号
            
        返回:
            metrics: 训练指标
        """
        # Keep loss warmup/phase in sync with epoch for losses that implement set_epoch.
        if hasattr(self.loss_fn, "set_epoch"):
            self.loss_fn.set_epoch(epoch)

        # 训练阶段使用 train() 模式，启用 dropout / BN 的训练行为。
        self.model.train()
        self.hook_manager.trigger("on_epoch_start", trainer=self, epoch=epoch)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.train.epochs}")
        
        # 优化器梯度清零 (移动到循环外，配合梯度累加)
        self.optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(pbar):
            # Handle both 2-value (images, labels) and 3-value (images, labels, paths) batch returns
            # 数据集可能返回 (image, label, path) 或 (image, label) 两种格式。
            extra_targets = None
            if len(batch_data) == 4:
                images, labels, _, extra_targets = batch_data
            elif len(batch_data) == 3:
                images, labels, _ = batch_data
            elif len(batch_data) == 2:
                images, labels = batch_data
            else:
                raise ValueError(f"Unexpected batch format: got {len(batch_data)} elements")
            # 触发 batch_start Hook
            self.hook_manager.trigger("on_batch_start", trainer=self, batch_idx=batch_idx)
            
            # 训练步骤 (传入 batch_idx 和 total_batches 以处理梯度累加)
            loss, correct, total = self._train_step(
                images,
                labels,
                batch_idx,
                len(self.train_loader),
                extra_targets=extra_targets,
            )
            
            total_loss += loss * total
            total_correct += correct
            total_samples += total
            
            # 更新进度(
            pbar.set_postfix({
                'loss': f'{total_loss/total_samples:.4f}',
                'acc': f'{total_correct/total_samples:.4f}'
            })
            
            # 触发 batch_end Hook
            self.hook_manager.trigger("on_batch_end", trainer=self, batch_idx=batch_idx, loss=loss)
        
        metrics = {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples
        }
        
        return metrics
    
    def _train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_idx: int = 0,
        total_batches: int = 1,
        extra_targets: Optional[Dict[str, Any]] = None,
    ) -> tuple[float, int, int]:
        """
        训练单个 batch (支持梯度累加)
        
        参数:
            images: 图像张量
            labels: 标签张量
            batch_idx: 当前 batch 索引
            total_batches: 总 batch 数
            
        返回:
            (loss, correct, total): 损失、正确数、总数
        """
        # 移动到设(
        # 训练阶段输入通常是 [B, C, H, W]。
        images = images.to(self.device)
        labels = labels.to(self.device)
        extra_targets = self._move_extra_targets_to_device(extra_targets, self.device)
        
        # 前向传播
        # outputs 对普通分类模型通常是 [B, Num_Classes]；
        # 对某些带特征返回的模型则可能是 (logits, features)。
        outputs = self.model(images)
        
        # Pass full outputs to loss_fn so CE+CLOC / CE+AOL can use embeddings.
        loss = self._compute_loss(outputs, labels, extra_targets=extra_targets)
        
        # 缩放损失 (为了梯度累加)
        # 如果 accumulation_steps=1，则 loss / 1 不变
        # 做梯度累加时，要先把 loss 按累加步数缩放。
        scaled_loss = loss / self.accumulation_steps
        
        # 反向传播
        scaled_loss.backward()
        
        # 判断是否需要更新权重
        # 条件：累加步数达到 OR 最后一个 batch
        is_update_step = ((batch_idx + 1) % self.accumulation_steps == 0) or ((batch_idx + 1) == total_batches)
        
        if is_update_step:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # 计算指标
        correct, total = self.strategy.calculate_metrics(outputs, labels)
        
        return loss.item(), correct, total
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        验证一(epoch
        
        参数:
            epoch: 当前 epoch 编号
            
        返回:
            metrics: 验证指标
        """
        self.hook_manager.trigger("on_validation_start", trainer=self)
        
        # 验证阶段统一交给 Evaluator，这样训练器不需要关心 TTA 等细节。
        metrics = self.evaluator.evaluate(
            self.val_loader,
            self.loss_fn,
            desc=f"Validating Epoch {epoch+1}"
        )
        
        self.hook_manager.trigger("on_validation_end", trainer=self, metrics=metrics)
        
        return metrics
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _load_config(self, config: Union[TrainingConfig, dict, str, Path]) -> TrainingConfig:
        """加载配置"""
        if isinstance(config, TrainingConfig):
            return config
        elif isinstance(config, dict):
            # 尝试转换(Pydantic 对象
            try:
                return TrainingConfig(**config)
            except:
                # 如果失败，包装为简单对(
                class DictConfig:
                    def __init__(self, d):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                setattr(self, k, DictConfig(v))
                            else:
                                setattr(self, k, v)
                    def get(self, key, default=None):
                        return getattr(self, key, default)
                return DictConfig(config)
        elif isinstance(config, (str, Path)):
            return load_config(config, validate=False)
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
    
    def _set_random_seed(self) -> None:
        """设置随机种子以确保可复现性"""
        seed = None
        
        # 尝试从配置中获取 random_seed
        if hasattr(self.config, 'random_seed'):
            seed = self.config.random_seed
        elif isinstance(self.config, dict):
            seed = self.config.get('random_seed')
        
        # 如果配置中没有设置，则不设置种子（保持随机性）
        if seed is None:
            return
        
        # 设置所有随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 确保 CUDA 操作的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"[RandomSeed] 已设置所有随机种子为: {seed}")
    
    def _setup_output_dir(self) -> Path:
        """设置输出目录"""
        # 获取基础目录
        if hasattr(self.config, 'output_dir'):
            base_dir = self.config.output_dir
        elif isinstance(self.config, dict):
            base_dir = self.config.get("output_dir", "./runs")
        else:
            base_dir = getattr(self.config, 'output_dir', "./runs")
        
        base_dir_str = str(base_dir).strip()
        if base_dir_str and base_dir_str not in {"./runs", "runs"}:
            output_dir = Path(base_dir_str)
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir

        exp_name = "exp"
        if hasattr(self.config, 'wandb_run_name'):
            exp_name = self.config.wandb_run_name
        elif hasattr(self.config, 'run_name'):
            exp_name = self.config.run_name
        elif isinstance(self.config, dict):
            exp_name = self.config.get("wandb_run_name") or self.config.get("run_name", "exp")
        
        exp_name = str(exp_name or "exp")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(base_dir) / f"{exp_name}_{timestamp}"
        if os.name == "nt" and len(str(output_dir)) > 220:
            suffix = hashlib.md5(exp_name.encode("utf-8")).hexdigest()[:10]
            output_dir = Path(base_dir) / f"exp_{suffix}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _setup_logger(self) -> Any:
        """设置日志器"""
        log_file = self.output_dir / "train.log"
        configure_logging(
            log_level="INFO",
            log_format="console",
            log_file=log_file
        )
        return get_logger("trainer")
    
    def _save_config(self) -> None:
        """保存配置"""
        config_file = self.output_dir / "config.yaml"
        
        if hasattr(self.config, 'model_dump'):
            config_dict = self.config.model_dump()
        elif isinstance(self.config, dict):
            config_dict = self.config
        else:
            config_dict = vars(self.config)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True)
    
    def _save_model_structure(self) -> None:
        """保存模型结构"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        structure_file = self.output_dir / "model_structure.txt"
        with open(structure_file, 'w', encoding='utf-8') as f:
            f.write(str(self.model))
    
    def _setup_hooks(self) -> HookManager:
        """设置默认 Hooks"""
        hook_manager = HookManager(self.logger)
        
        # WandB Hook
        use_wandb = False
        if hasattr(self.config, 'use_wandb'):
            use_wandb = self.config.use_wandb
        elif isinstance(self.config, dict):
            use_wandb = self.config.get("use_wandb", False)
        
        if use_wandb:
            hook_manager.register(WandBHook(self.config, enabled=True))
        
        # 2026-03-20 Codex: 训练期的 checkpoint 和 early stopping 统一读 train.selection_* 配置。
        selection_metric = self._normalize_metric_name(
            self._get_train_option("selection_metric", "val_acc")
        )
        selection_mode = self._get_train_option("selection_mode", None)
        selection_mode = selection_mode or infer_metric_mode(selection_metric)
        selection_min_delta = float(self._get_train_option("selection_min_delta", 0.0) or 0.0)

        # Checkpoint Hook
        hook_manager.register(CheckpointHook(
            self.output_dir,
            metric=selection_metric,
            mode=selection_mode,
            min_delta=selection_min_delta,
        ))
        
        # Early Stopping Hook ────────────────────────────────────────────────
        # FIX: 之前 patience 配置被读取但从未注册 EarlyStoppingHook
        patience = 0
        if hasattr(self.config, 'train') and hasattr(self.config.train, 'patience'):
            patience = self.config.train.patience or 0
        elif isinstance(self.config, dict):
            patience = (self.config.get('train') or {}).get('patience', 0)
        
        if patience and patience > 0:
            hook_manager.register(EarlyStoppingHook(
                patience=patience,
                metric=selection_metric,
                mode=selection_mode,
                min_delta=selection_min_delta,
            ))
            if self.logger:
                self.logger.info(
                    "early_stopping_hook_registered",
                    patience=patience,
                    selection_metric=selection_metric,
                    selection_mode=selection_mode,
                    selection_min_delta=selection_min_delta,
                )
        
        # LR Scheduler Hook
        if self.scheduler:
            hook_manager.register(LRSchedulerHook(self.scheduler))
        
        return hook_manager
    
    def _run_error_analysis(self) -> None:
        """
        运行错误分析
        
        在训练结束后，加载最佳模型并生成错误分析报告
        """
        best_model_path = self.output_dir / "best_model.pth"
        
        if not best_model_path.exists():
            self.logger.warning("error_analysis_skipped", reason="best_model_not_found")
            return
        
        self.logger.info("error_analysis_start", model_path=str(best_model_path))
        
        try:
            # 加载最佳模型
            checkpoint = torch.load(best_model_path)
            
            # CheckpointHook 保存的是字典格式，需要提取 model_state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 如果是直接保存的 state_dict
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            
            # 导入错误分析工具
            from ..utils.analysis import analyze_and_save_errors

            if self.test_loader is None:
                self.logger.warning("error_analysis_skipped", reason="test_loader_not_found")
                return
            
            # 获取类别名称
            if hasattr(self.test_loader.dataset, 'classes'):
                class_names = self.test_loader.dataset.classes
            else:
                # 尝试从 dataset 的 dataset 属性获取（处理 Subset 情况）
                if hasattr(self.test_loader.dataset, 'dataset') and hasattr(self.test_loader.dataset.dataset, 'classes'):
                    class_names = self.test_loader.dataset.dataset.classes
                elif hasattr(self.train_loader.dataset, 'classes'):
                    class_names = self.train_loader.dataset.classes
                else:
                    self.logger.warning("error_analysis_failed", reason="class_names_not_found")
                    return
            
            # 运行错误分析
            save_error_images = bool(
                self._get_train_option("error_analysis_save_images", True)
            )
            analyze_and_save_errors(
                model=self.model,
                dataloader=self.test_loader,
                device=self.device,
                output_dir=str(self.output_dir),
                classes=class_names,  # Parameter name is 'classes' not 'class_names'
                history=self.history,
                split_name="test",
                tta_mode=getattr(self.evaluator, "tta_mode", "mean"),
                tta_topk=getattr(self.evaluator, "tta_topk", 0),
                save_error_images=save_error_images,
            )
            
            self.logger.info("error_analysis_complete", output_dir=str(self.output_dir / "error_analysis"))
            
        except Exception as e:
            self.logger.error("error_analysis_failed", error=str(e))
            import traceback
            self.logger.error("error_analysis_traceback", traceback=traceback.format_exc())


# ============================================================================
# 向后兼容：保留旧版 Trainer 接口
# ============================================================================

def create_trainer_legacy(cfg: dict) -> Trainer:
    """
    创建旧版 Trainer(向后兼容)
    
    参数:
        cfg: 配置字典
        
    返回:
        trainer: Trainer 实例
    """
    return Trainer(cfg)
