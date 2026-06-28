"""
Hook 系统 (Hook System)

设计目标：
    提供扩展点，允许在训练流程的关键节点插入自定义逻辑。
    
核心问题：
    用户痛点："ViT 加了新 loss 后 Trainer 就报错"
    - 不同模型需要不同的处理逻辑
    - 修改 Trainer 代码风险高
    - 难以复用自定义逻辑
    
解决方案：
    使用 Observer 模式，定义 Hook 接口，允许用户注册自定义 Hook。
    Trainer 在关键节点触发 Hook，而不需要修改核心代码。
    
使用示例：
    # 自定义 Hook
    class MyCustomHook(Hook):
        def on_epoch_end(self, trainer, epoch, metrics):
            print(f"Epoch {epoch} completed with acc={metrics['acc']}")
    
    # 注册 Hook
    hook_manager = HookManager()
    hook_manager.register(MyCustomHook())
    hook_manager.register(WandBHook(config))
    
    # Trainer 使用
    trainer = Trainer(config, hook_manager=hook_manager)
"""

from typing import Protocol, List, Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn


# 2026-03-20 Codex: 统一推断监控指标方向，减少每次手动写 max/min 的重复配置。
def infer_metric_mode(metric: Optional[str]) -> str:
    """Infer whether a monitored metric should be maximized or minimized."""
    name = str(metric or "val_acc").lower()
    if any(token in name for token in ("loss", "mae", "error")):
        return "min"
    return "max"


# ============================================================================
# Hook 接口
# ============================================================================

class Hook(Protocol):
    """
    Hook 接口（协议）
    
    定义了训练流程中的所有扩展点。
    用户可以实现部分或全部方法。
    """
    
    def on_train_begin(self, trainer: Any) -> None:
        """训练开始前"""
        ...
    
    def on_train_end(self, trainer: Any) -> None:
        """训练结束后"""
        ...
    
    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Epoch 开始前"""
        ...
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Epoch 结束后"""
        ...
    
    def on_batch_start(self, trainer: Any, batch_idx: int) -> None:
        """Batch 开始前"""
        ...
    
    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        """Batch 结束后"""
        ...
    
    def on_validation_start(self, trainer: Any) -> None:
        """验证开始前"""
        ...
    
    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]) -> None:
        """验证结束后"""
        ...


# ============================================================================
# Hook 管理器
# ============================================================================

class HookManager:
    """
    Hook 管理器
    
    职责：
        - 管理所有注册的 Hook
        - 在适当的时机触发 Hook
        - 处理 Hook 执行异常
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """
        初始化 Hook 管理器
        
        参数:
            logger: 日志器（可选）
        """
        self.hooks: List[Hook] = []
        self.logger = logger
    
    def register(self, hook: Hook) -> None:
        """
        注册 Hook
        
        参数:
            hook: Hook 实例
        """
        self.hooks.append(hook)
        if self.logger:
            self.logger.info(
                "hook_registered",
                hook_type=hook.__class__.__name__
            )
    
    def trigger(self, event: str, **kwargs) -> None:
        """
        触发事件
        
        参数:
            event: 事件名称（如 "on_epoch_end"）
            **kwargs: 传递给 Hook 的参数
        """
        for hook in self.hooks:
            if hasattr(hook, event):
                try:
                    getattr(hook, event)(**kwargs)
                except Exception as e:
                    if self.logger:
                        # Fix: Use exc_info for exception details, avoid parameter conflict
                        self.logger.error(
                            f"hook_execution_failed: {hook.__class__.__name__}.{event}",
                            error=str(e),
                            exc_info=True
                        )
                    # 不中断训练，继续执行其他 Hook


# ============================================================================
# 内置 Hooks
# ============================================================================

class WandBHook:
    """
    WandB 日志 Hook
    
    自动记录训练指标到 WandB
    """
    
    def __init__(self, config: Any, enabled: bool = True):
        """
        初始化 WandB Hook
        
        参数:
            config: 配置对象
            enabled: 是否启用
        """
        self.enabled = enabled and config.use_wandb
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                self.enabled = False
                self.wandb = None
    
    def on_train_begin(self, trainer: Any) -> None:
        """初始化 WandB"""
        if not self.enabled:
            return
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Fix: Check both run_name and wandb_run_name for backward compatibility
        exp_name = getattr(trainer.config, "run_name", None) or \
                   getattr(trainer.config, "wandb_run_name", "exp")
        
        # Fix: Safely access wandb config
        wandb_config = getattr(trainer.config, 'wandb', None)
        project_name = wandb_config.project if wandb_config else "blacktea"
        
        try:
            self.wandb.init(
                project=project_name,
                name=f"{exp_name}_{timestamp}",
                config=trainer.config.model_dump() if hasattr(trainer.config, 'model_dump') else trainer.config,
                dir=str(trainer.output_dir)
            )
        except Exception as e:
            if trainer.logger:
                trainer.logger.error(
                    f"wandb_init_failed: WandB initialization failed, disabling WandB logging.",
                    error=str(e),
                    exc_info=True
                )
            # 初始化失败，自动禁用后续所有 logging，防止刷屏报错
            self.enabled = False
            self.wandb = None
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """记录 epoch 指标"""
        if not self.enabled or self.wandb.run is None:
            return
        
        self.wandb.log({
            "epoch": epoch,
            **metrics
        })
        
    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        """记录 batch 指标"""
        if not self.enabled or self.wandb.run is None:
            return
            
        self.wandb.log({
            "train/batch_loss": loss
        })
    
    def on_train_end(self, trainer: Any) -> None:
        """训练结束：在测试集上评估，上传指标和混淆矩阵，然后结束 WandB"""
        if not self.enabled:
            return
        
        try:
            self._run_test_evaluation(trainer)
        except Exception as e:
            # 测试评估失败不影响 WandB 正常结束
            if hasattr(trainer, 'logger') and trainer.logger:
                trainer.logger.error("wandb_test_eval_failed", error=str(e))
        
        self.wandb.finish()
    
    def _run_test_evaluation(self, trainer: Any) -> None:
        """
        加载最佳模型，在测试集上评估，并将结果上传到 WandB。
        
        上传内容:
            - test/accuracy, test/loss 等标量指标
            - test/confusion_matrix 混淆矩阵图
            - test/per_class_accuracy 每类准确率
        """
        import json
        import torch
        import numpy as np
        
        best_model_path = trainer.output_dir / "best_model.pth"
        if not best_model_path.exists():
            return
        
        # 加载最佳模型权重
        checkpoint = torch.load(best_model_path, map_location=trainer.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            trainer.model.load_state_dict(checkpoint)
        trainer.model.eval()
        
        # 在测试集上评估
        test_loader = trainer.test_loader
        evaluator = trainer.evaluator
        
        # 基础指标
        test_metrics = evaluator.evaluate(
            test_loader,
            trainer.loss_fn,
            desc="Testing (WandB)"
        )
        
        # 获取类别名称
        class_names = None
        dataset = test_loader.dataset
        if hasattr(dataset, 'classes'):
            class_names = dataset.classes
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'classes'):
            class_names = dataset.dataset.classes
        
        # 推断 num_classes：
        # 1. 优先用已取到的 class_names 长度
        # 2. 其次读 head.num_classes（明确在配置中定义）
        # 3. 均不可用时传 None，让 compute_confusion_matrix 自动推断
        if class_names is not None:
            num_classes = len(class_names)
        else:
            try:
                num_classes = trainer.config.model.head.num_classes
            except AttributeError:
                num_classes = None

        # 计算混淆矩阵
        cm = evaluator.compute_confusion_matrix(test_loader, num_classes=num_classes)
        
        per_class_acc = {}
        for cls_idx in range(cm.shape[0]):
            total = cm[cls_idx, :].sum()
            acc = cm[cls_idx, cls_idx] / total if total > 0 else 0.0
            name = class_names[cls_idx] if class_names else str(cls_idx)
            per_class_acc[name] = float(acc)

        # Save local test summary for LOBO macro-averaging scripts.
        test_summary = {
            "accuracy": float(test_metrics.get("accuracy", 0.0)),
            "loss": float(test_metrics.get("loss", 0.0)) if "loss" in test_metrics else None,
            "class_names": class_names if class_names else [str(i) for i in range(cm.shape[0])],
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": cm.tolist(),
            "num_samples": int(cm.sum()),
        }
        with (trainer.output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_summary, f, ensure_ascii=False, indent=2)

        # ---- 上传标量指标 ----
        wandb_log = {
            "test/accuracy": test_metrics.get("accuracy", 0.0),
        }
        if "loss" in test_metrics:
            wandb_log["test/loss"] = test_metrics["loss"]
        
        # 每类准确率
        for name, acc in per_class_acc.items():
            wandb_log[f"test/class_acc/{name}"] = acc
        
        self.wandb.log(wandb_log)
        
        # ---- 上传混淆矩阵图 ----
        cm_image = self._plot_confusion_matrix(cm, class_names)
        self.wandb.log({"test/confusion_matrix": cm_image})
    
    # ── 中文字体支持（模块级缓存，只检测一次）─────────────────────────────────
    _cjk_font_name: Optional[str] = None
    _cjk_font_checked: bool = False

    # 红茶发酵阶段的英文备用标签（顺序与中文 class_to_idx 保持一致）
    _FERMENTATION_EN = {
        "发酵前":  "PreFerm",
        "轻微发酵": "Light",
        "适度发酵": "Moderate",
        "过度发酵": "Over",
    }

    def _resolve_cjk_font(self) -> Optional[str]:
        """
        检测当前环境是否有可用的 CJK 字体。
        结果缓存在类变量，整个训练过程只检测一次。
        """
        if WandBHook._cjk_font_checked:
            return WandBHook._cjk_font_name

        WandBHook._cjk_font_checked = True
        candidates = [
            # Windows
            "Microsoft YaHei", "SimHei", "SimSun", "FangSong",
            # Linux
            "WenQuanYi Micro Hei", "Noto Sans CJK SC",
            "AR PL UMing CN", "Droid Sans Fallback",
        ]
        try:
            import matplotlib.font_manager as fm
            available = {f.name for f in fm.fontManager.ttflist}
            for name in candidates:
                if name in available:
                    WandBHook._cjk_font_name = name
                    return name
        except Exception:
            pass
        return None

    def _plot_confusion_matrix(
        self,
        cm: "np.ndarray",
        class_names: Optional[List[str]] = None
    ) -> "wandb.Image":
        """
        绘制并返回混淆矩阵的 WandB Image 对象。
        使用归一化（每行占比）以便不同类别样本数不同时也易于对比。
        - 自动检测 CJK 字体，找到则用中文标签；
        - 找不到则将类别名替换为英文缩写，消除 glyph UserWarning。
        """
        import numpy as np
        import warnings
        import matplotlib
        matplotlib.use("Agg")   # 无头模式，不弹窗
        import matplotlib.pyplot as plt

        num_classes = cm.shape[0]
        raw_labels = class_names if class_names else [str(i) for i in range(num_classes)]
        labels_numeric = None
        try:
            labels_numeric = [int(x) for x in raw_labels]
        except Exception:
            labels_numeric = None

        # ── 字体 & 标签处理 ──────────────────────────────────────────────────
        cjk_font = self._resolve_cjk_font()
        if cjk_font:
            # 有 CJK 字体：直接使用中文标签
            plt.rcParams["font.family"] = cjk_font
            labels = raw_labels
        else:
            # 无 CJK 字体：尝试翻译为英文，屏蔽 glyph warning
            labels = [self._FERMENTATION_EN.get(l, l) for l in raw_labels]
            warnings.filterwarnings(
                "ignore",
                message="Glyph .* missing from font",
                category=UserWarning,
            )

        # Numeric-label fallback for timepoint tasks.
        if labels_numeric is not None and sorted(labels_numeric) == list(range(len(labels_numeric))):
            if len(labels_numeric) == 13:
                labels = [f"{i * 0.5:.1f}h" for i in labels_numeric]

        # ── 归一化（行归一化，即 recall）────────────────────────────────────
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1   # 避免除零
        cm_norm = cm_norm / row_sums

        fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Predicted",
            ylabel="True",
            title="Test Confusion Matrix (row-normalized)",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 在格子里写数字（原始计数 + 归一化比例）
        thresh = 0.5
        for i in range(num_classes):
            for j in range(num_classes):
                color = "white" if cm_norm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                        ha="center", va="center", color=color, fontsize=8)

        fig.tight_layout()

        # 还原 warning 过滤器，避免影响其他模块
        warnings.resetwarnings()

        wandb_image = self.wandb.Image(fig)
        plt.close(fig)
        return wandb_image


class CheckpointHook:
    """
    检查点保存 Hook
    
    自动保存最佳模型和定期检查点
    """
    
    def __init__(
        self,
        output_dir: Path,
        metric: str = "val_acc",
        mode: Optional[str] = None,
        save_interval: Optional[int] = None,
        min_delta: float = 0.0,
    ):
        """
        初始化检查点 Hook
        
        参数:
            output_dir: 输出目录
            metric: 监控的指标
            mode: "max" 或 "min"
            save_interval: 保存间隔（epoch）
        """
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.mode = mode or infer_metric_mode(metric)
        self.save_interval = save_interval
        self.min_delta = float(min_delta or 0.0)
        
        self.best_value = float('-inf') if self.mode == 'max' else float('inf')
        self.best_epoch = 0
        self.has_saved_once = False  # 跟踪是否至少保存过一次
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """保存检查点"""
        # 保存最佳模型
        if self.metric in metrics:
            value = metrics[self.metric]
            if self.mode == 'max':
                is_best = value > self.best_value + self.min_delta
            else:
                is_best = value < self.best_value - self.min_delta
            
            # 如果是最佳性能，或者第一次保存，都进行保存
            if is_best or not self.has_saved_once:
                self.best_value = value
                self.best_epoch = epoch
                self.has_saved_once = True
                
                checkpoint_path = self.output_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'metrics': metrics,
                    self.metric: value
                }, checkpoint_path)
        
        # 定期保存
        if self.save_interval and (epoch + 1) % self.save_interval == 0:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': metrics
            }, checkpoint_path)


class EarlyStoppingHook:
    """
    早停 Hook
    
    当指标不再提升时提前停止训练
    """
    
    def __init__(
        self,
        patience: int = 10,
        metric: str = "val_acc",
        mode: Optional[str] = None,
        min_delta: float = 0.0
    ):
        """
        初始化早停 Hook
        
        参数:
            patience: 容忍的 epoch 数
            metric: 监控的指标
            mode: "max" 或 "min"
            min_delta: 最小改进量
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode or infer_metric_mode(metric)
        self.min_delta = min_delta
        
        self.best_value = float('-inf') if self.mode == 'max' else float('inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """检查是否应该早停"""
        if self.metric not in metrics:
            return
        
        value = metrics[self.metric]
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            if hasattr(trainer, 'logger') and trainer.logger:
                trainer.logger.info(
                    "early_stopping_triggered",
                    epoch=epoch,
                    patience=self.patience,
                    best_value=self.best_value
                )


class LRSchedulerHook:
    """
    学习率调度 Hook

    在每个 epoch 后更新学习率。
    Warmup 由 builder.py 通过 SequentialLR 统一管理，
    此 Hook 每 epoch 无条件调用 scheduler.step()。
    """

    def __init__(self, scheduler: Any):
        """
        初始化学习率调度 Hook

        参数:
            scheduler: PyTorch 调度器（含 Warmup 时应为 SequentialLR）
        """
        self.scheduler = scheduler
        self.current_epoch = 0

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """更新学习率（Warmup 由 SequentialLR 内部处理，无需在此跳过）"""
        self.current_epoch = epoch

        if self.scheduler:
            self.scheduler.step()

            if hasattr(trainer, 'logger') and trainer.logger:
                current_lr = self.scheduler.get_last_lr()[0]
                trainer.logger.info(
                    "lr_updated",
                    epoch=epoch,
                    lr=current_lr
                )


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Hook 系统示例")
    print("="*60)
    
    # 创建 Hook 管理器
    hook_manager = HookManager()
    
    # 注册 Hooks
    class DummyConfig:
        use_wandb = False
    
    hook_manager.register(WandBHook(DummyConfig(), enabled=False))
    hook_manager.register(CheckpointHook(Path("./test_output")))
    hook_manager.register(EarlyStoppingHook(patience=3))
    
    print(f"✓ 注册了 {len(hook_manager.hooks)} 个 Hooks")
    
    # 模拟训练流程
    class DummyTrainer:
        def __init__(self):
            self.model = None
            self.optimizer = None
            self.output_dir = Path("./test_output")
    
    trainer = DummyTrainer()
    
    # 触发事件
    hook_manager.trigger("on_train_begin", trainer=trainer)
    print("✓ on_train_begin 触发")
    
    for epoch in range(5):
        hook_manager.trigger("on_epoch_start", trainer=trainer, epoch=epoch)
        
        metrics = {
            "train_loss": 0.5 - epoch * 0.1,
            "val_acc": 0.7 + epoch * 0.05
        }
        
        hook_manager.trigger("on_epoch_end", trainer=trainer, epoch=epoch, metrics=metrics)
        print(f"✓ Epoch {epoch} 完成, val_acc={metrics['val_acc']:.3f}")
    
    hook_manager.trigger("on_train_end", trainer=trainer)
    print("✓ on_train_end 触发")
    
    print("\n✅ Hook 系统测试完成！")
