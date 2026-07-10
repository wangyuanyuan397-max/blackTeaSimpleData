"""
Evaluator - 独立的模型评估器

职责：
    - 在验证集/测试集上评估模型
    - 计算各种评估指标
    - 生成混淆矩阵和 per-class accuracy
    - 与 Strategy 模式集成

设计模式：
    - Strategy Pattern: 使用 ModelStrategy 处理不同模型类型
    - Dependency Injection: 通过构造函数注入依赖
    
使用示例：
    ```python
    evaluator = Evaluator(model, strategy, device, logger)
    
    # 评估整个数据集
    metrics = evaluator.evaluate(val_loader, loss_fn)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    
    # 评估单个批次
    batch_metrics = evaluator.evaluate_batch(images, labels, loss_fn)
    
    # 获取预测结果
    predictions, labels = evaluator.get_predictions(test_loader)
    
    # 计算混淆矩阵
    cm = evaluator.compute_confusion_matrix(test_loader)
    ```
"""

from typing import Dict, Tuple, Optional, Any
import inspect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from ..utils.strategy import ModelStrategy, OrdinalRegressionStrategy
from ..utils.exceptions import EvaluationError


# 2026-03-20 Codex: 新增有序任务评估指标，方便后续按 MAE/QWK 做选模与早停。
def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE for ordinal class indices."""
    if y_true.size == 0:
        return 0.0
    return float(np.abs(y_true.astype(np.float64) - y_pred.astype(np.float64)).mean())


def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute quadratic weighted kappa without external dependencies."""
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    min_rating = int(min(y_true.min(), y_pred.min()))
    max_rating = int(max(y_true.max(), y_pred.max()))
    num_ratings = max_rating - min_rating + 1

    if num_ratings <= 1:
        return 1.0

    conf_mat = np.zeros((num_ratings, num_ratings), dtype=np.float64)
    for true_label, pred_label in zip(y_true, y_pred):
        conf_mat[true_label - min_rating, pred_label - min_rating] += 1.0

    hist_true = conf_mat.sum(axis=1)
    hist_pred = conf_mat.sum(axis=0)
    num_samples = conf_mat.sum()
    if num_samples <= 0:
        return 0.0

    expected = np.outer(hist_true, hist_pred) / num_samples
    weights = np.zeros((num_ratings, num_ratings), dtype=np.float64)
    denom = float((num_ratings - 1) ** 2)
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = ((i - j) ** 2) / denom

    observed_score = float((weights * conf_mat).sum() / num_samples)
    expected_score = float((weights * expected).sum() / num_samples)
    if expected_score <= 1e-12:
        return 1.0 if observed_score <= 1e-12 else 0.0
    return float(1.0 - observed_score / expected_score)


def _plus_minus_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 ±1 accuracy：预测与真实标签相差不超过 1 也视为正确。"""
    if y_true.size == 0:
        return 0.0
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    # 这里默认标签编码已经按有序分类顺序映射为 0,1,2,3。
    return float((np.abs(y_true - y_pred) <= 1).mean())


class Evaluator:
    """
    独立的模型评估器
    
    设计理念：
        - 可脱离 Trainer 独立使用
        - 支持多种评估场景
        - 自动使用策略模式计算指标
        - 返回标准化的指标字典
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategy: ModelStrategy,
        device: torch.device,
        logger: Optional[Any] = None,
        tta_mode: str = "mean",
        tta_topk: int = 0,
        tta_compare: bool = False,
        tta_compare_modes: Optional[list] = None
    ):
        """
        初始化 Evaluator
        
        参数:
            model: 要评估的模型
            strategy: 模型处理策略（用于获取预测、计算指标）
            device: 计算设备（CPU/GPU）
            logger: 日志记录器（可选）
        """
        self.model = model
        self.strategy = strategy
        self.device = device
        self.logger = logger
        # tta_mode 决定多裁剪/多滑窗结果怎么聚合：
        # mean / vote / topk
        self.tta_mode = (tta_mode or "mean").lower()
        self.tta_topk = int(tta_topk or 0)
        self.tta_compare = bool(tta_compare)
        self.tta_compare_modes = [str(m).lower() for m in (tta_compare_modes or ["vote", "topk"])]

        # Keep evaluation simple and deterministic:
        # when main mode is vote/topk, do not compute alternative TTA metrics.
        # 当主模式已经是 vote/topk 时，不再额外比较其他模式，避免评估逻辑过重。
        if self.tta_mode in {"vote", "topk"}:
            self.tta_compare = False
            self.tta_compare_modes = []

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

    @staticmethod
    def _compute_loss(loss_fn: Optional[nn.Module], outputs, labels, extra_targets=None):
        if loss_fn is None:
            return None
        if extra_targets is None:
            return loss_fn(outputs, labels)

        forward_fn = getattr(loss_fn, "forward", None)
        if forward_fn is None:
            return loss_fn(outputs, labels)

        try:
            signature = inspect.signature(forward_fn)
        except (TypeError, ValueError):
            return loss_fn(outputs, labels)

        if "extra_targets" in signature.parameters:
            return loss_fn(outputs, labels, extra_targets=extra_targets)
        return loss_fn(outputs, labels)

    def _aggregate_tta_features(self, features: torch.Tensor, batch_size: int, num_crops: int) -> torch.Tensor:
        """
        Aggregate features from multiple TTA crops by averaging.
        
        Args:
            features: [B*N, D] tensor of features
            batch_size: original batch size
            num_crops: number of crops per sample
            
        Returns:
            [B, D] tensor of aggregated features
        """
        features = features.view(batch_size, num_crops, -1)
        return features.mean(dim=1)

    def _uses_ordinal_aggregation(self) -> bool:
        """Return True when outputs should be aggregated as ordinal rank logits."""
        return isinstance(self.strategy, OrdinalRegressionStrategy)

    @staticmethod
    def _rank_probas_to_logits(rank_probas: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Convert rank probabilities P(y > k) back into numerically stable logits."""
        rank_probas = rank_probas.clamp(min=eps, max=1.0 - eps)
        return torch.log(rank_probas) - torch.log1p(-rank_probas)

    @staticmethod
    def _rank_probas_to_class_probs(rank_probas: torch.Tensor) -> torch.Tensor:
        """Convert ordinal rank probabilities to K-way class probabilities."""
        first = 1.0 - rank_probas[..., :1]
        last = rank_probas[..., -1:]
        middle = rank_probas[..., :-1] - rank_probas[..., 1:]
        class_probs = torch.cat([first, middle, last], dim=-1)
        class_probs = class_probs.clamp_min(0.0)
        return class_probs / class_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _class_probs_to_rank_probas(class_probs: torch.Tensor) -> torch.Tensor:
        """Convert K-way class probabilities back to ordinal rank probabilities."""
        tail_sums = torch.cumsum(torch.flip(class_probs, dims=[-1]), dim=-1)
        return torch.flip(tail_sums, dims=[-1])[..., 1:]

    def _aggregate_tta_ordinal_logits(
        self,
        logits: torch.Tensor,
        batch_size: int,
        num_crops: int,
        mode: str,
        topk: int,
    ) -> torch.Tensor:
        """
        Aggregate CORAL / ordinal outputs without misusing softmax over rank logits.

        The returned tensor keeps the original [B, K-1] rank-logit shape so downstream
        decoding and ordinal losses remain compatible.
        """
        rank_probs = torch.sigmoid(logits).view(batch_size, num_crops, -1)
        class_probs = self._rank_probas_to_class_probs(rank_probs)
        num_classes = class_probs.size(-1)

        if mode == "vote":
            preds = self.strategy.get_predictions(logits).view(batch_size, num_crops)
            counts = torch.zeros(batch_size, num_classes, device=logits.device, dtype=rank_probs.dtype)
            ones = torch.ones_like(preds, dtype=rank_probs.dtype)
            counts.scatter_add_(1, preds, ones)
            agg_rank_probs = self._class_probs_to_rank_probas(counts / float(num_crops))
        elif mode == "topk":
            if topk <= 0:
                agg_rank_probs = rank_probs.mean(dim=1)
            else:
                k = min(topk, num_crops)
                conf = class_probs.max(dim=2).values
                topk_idx = conf.topk(k, dim=1).indices
                gathered = rank_probs.gather(
                    1,
                    topk_idx.unsqueeze(-1).expand(-1, -1, rank_probs.size(-1)),
                )
                agg_rank_probs = gathered.mean(dim=1)
        else:
            agg_rank_probs = rank_probs.mean(dim=1)

        return self._rank_probas_to_logits(agg_rank_probs)
    
    def _aggregate_tta_logits(self, logits: torch.Tensor, batch_size: int, num_crops: int, mode: str, topk: int) -> torch.Tensor:
        if self._uses_ordinal_aggregation():
            return self._aggregate_tta_ordinal_logits(logits, batch_size, num_crops, mode, topk)
        # 先把每个 crop 的 logits 变成概率，再按 TTA 策略聚合。
        probs = torch.softmax(logits, dim=1)
        probs = probs.view(batch_size, num_crops, -1)
        num_classes = probs.size(-1)

        if mode == "vote":
            preds = probs.argmax(dim=2)
            counts = torch.zeros(batch_size, num_classes, device=probs.device, dtype=probs.dtype)
            ones = torch.ones_like(preds, dtype=probs.dtype)
            counts.scatter_add_(1, preds, ones)
            agg = counts / float(num_crops)
        elif mode == "topk":
            if topk <= 0:
                agg = probs.mean(dim=1)
            else:
                k = min(topk, num_crops)
                conf = probs.max(dim=2).values
                topk_idx = conf.topk(k, dim=1).indices
                gathered = probs.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, num_classes))
                agg = gathered.mean(dim=1)
        else:
            agg = probs.mean(dim=1)

        return torch.log(agg.clamp_min(1e-12))

    def _forward_with_optional_tta(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, Dict[str, torch.Tensor]]:
        """
        Support both:
        - normal input: [B, C, H, W]
        - N-crop TTA input: [B, N, C, H, W]  (N=5 for 5-crop, N=63 for sliding window)
        When N is large, process crops in mini-batches to avoid GPU OOM.
        
        Returns:
            - outputs: model outputs (can be logits or (logits, features) tuple)
            - batch_size: number of samples
            - extras: dict of extra outputs for TTA comparison
        """
        # 5D 输入表示 TTA：
        # [B, N, C, H, W]，其中 N 可能是 5-crop 的 5，也可能是滑窗的 63。
        if images.dim() == 5:
            bsz, ncrops, c, h, w = images.shape

            # 小 N (e.g. 5-crop): 一次性推理
            if ncrops <= 16:
                # crop 数较少时，直接展平为一个大 batch 一次性前向。
                flat = images.view(bsz * ncrops, c, h, w)
                outputs = self.model(flat)
                
                if isinstance(outputs, tuple):
                    logits, features = outputs
                    agg_logits = self._aggregate_tta_logits(logits, bsz, ncrops, self.tta_mode, self.tta_topk)
                    agg_features = self._aggregate_tta_features(features, bsz, ncrops)
                    agg = (agg_logits, agg_features)
                else:
                    logits = outputs
                    agg = self._aggregate_tta_logits(logits, bsz, ncrops, self.tta_mode, self.tta_topk)
                
                extras = {}
                if self.tta_compare:
                    for mode in self.tta_compare_modes:
                        if mode == self.tta_mode:
                            continue
                        if isinstance(outputs, tuple):
                            extras[mode] = (self._aggregate_tta_logits(logits, bsz, ncrops, mode, self.tta_topk), agg_features)
                        else:
                            extras[mode] = self._aggregate_tta_logits(logits, bsz, ncrops, mode, self.tta_topk)
                return agg, bsz, extras

            # 大 N (e.g. 63 sliding window): 逐样本分批推理
            crops_batch = 16  # 每批处理的 crops 数
            all_agg = []
            extras = {mode: [] for mode in self.tta_compare_modes if mode != self.tta_mode} if self.tta_compare else {}
            
            # crop 数很多时，按样本、按小块前向，避免显存溢出。
            for i in range(bsz):
                sample_crops = images[i]  # [N, C, H, W]
                logits_parts = []
                features_parts = []
                is_tuple_output = None
                
                for start in range(0, ncrops, crops_batch):
                    end = min(start + crops_batch, ncrops)
                    chunk = sample_crops[start:end]
                    out = self.model(chunk)
                    
                    if isinstance(out, tuple):
                        is_tuple_output = True
                        lg, ft = out
                        logits_parts.append(lg)
                        features_parts.append(ft)
                    else:
                        is_tuple_output = False
                        logits_parts.append(out)
                
                all_logits = torch.cat(logits_parts, dim=0)  # [N, C]
                
                if is_tuple_output:
                    all_features = torch.cat(features_parts, dim=0)  # [N, D]
                    agg_logits = self._aggregate_tta_logits(all_logits, 1, ncrops, self.tta_mode, self.tta_topk)
                    agg_features = self._aggregate_tta_features(all_features, 1, ncrops)
                    all_agg.append((agg_logits.squeeze(0), agg_features.squeeze(0)))
                    
                    if self.tta_compare:
                        for mode in extras.keys():
                            agg_logits_mode = self._aggregate_tta_logits(all_logits, 1, ncrops, mode, self.tta_topk)
                            extras[mode].append((agg_logits_mode.squeeze(0), agg_features.squeeze(0)))
                else:
                    agg = self._aggregate_tta_logits(all_logits, 1, ncrops, self.tta_mode, self.tta_topk)
                    all_agg.append(agg.squeeze(0))
                    
                    if self.tta_compare:
                        for mode in extras.keys():
                            extras[mode].append(self._aggregate_tta_logits(all_logits, 1, ncrops, mode, self.tta_topk).squeeze(0))
            
            if is_tuple_output:
                agg_logits_list = [item[0] for item in all_agg]
                agg_features_list = [item[1] for item in all_agg]
                result = (torch.stack(agg_logits_list, dim=0), torch.stack(agg_features_list, dim=0))
                
                if self.tta_compare:
                    extras = {k: (torch.stack([item[0] for item in v], dim=0), 
                                  torch.stack([item[1] for item in v], dim=0)) 
                              for k, v in extras.items()}
            else:
                result = torch.stack(all_agg, dim=0)
                if self.tta_compare:
                    extras = {k: torch.stack(v, dim=0) for k, v in extras.items()}
            
            return result, bsz, extras

        # 普通验证/测试输入是 [B, C, H, W]，直接前向。
        outputs = self.model(images)
        return outputs, images.size(0), {}
    
    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: Optional[nn.Module] = None,
        desc: str = "Evaluating"
    ) -> Dict[str, float]:
        """
        在数据集上评估模型
        
        参数:
            dataloader: 数据加载器
            loss_fn: 损失函数（可选）
            desc: 进度条描述
            
        返回:
            Dict[str, float]: 评估指标字典
                - accuracy: 准确率
                - loss: 平均损失（如果提供了 loss_fn）
                - correct: 正确预测数量
                - total: 总样本数量
                
        抛出:
            EvaluationError: 评估过程中出错
        """
        import gc
        try:
            # 评估阶段使用 eval() 模式，关闭 dropout，固定 BN 行为。
            self.model.eval()
            
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            # 2026-03-20 Codex: 验证阶段直接缓存全部预测，避免上层重复跑一遍才能算 MAE/QWK。
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                pbar = tqdm(dataloader, desc=desc, leave=False)
                for batch_idx, batch in enumerate(pbar):
                    # 解包批次数据
                    extra_targets = None
                    sample_paths = None
                    if len(batch) == 4:
                        images, labels, sample_paths, extra_targets = batch
                    elif len(batch) == 3:
                        images, labels, sample_paths = batch  # (images, labels, paths)
                    else:
                        images, labels = batch
                    if sample_paths is not None:
                        if extra_targets is None:
                            extra_targets = {}
                        else:
                            extra_targets = dict(extra_targets)
                        # 让验证/测试 loss 和训练 loss 一样能基于图片路径解析发酵时间。
                        extra_targets["sample_paths"] = sample_paths
                    
                    # 这里 images 既可能是 [B, C, H, W]，
                    # 也可能是 [B, N, C, H, W]（滑窗 / 五裁剪）。
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    extra_targets = self._move_extra_targets_to_device(extra_targets, self.device)
                    
                    # 前向传播（支持 5-crop TTA）
                    # 前向入口统一支持普通输入和 TTA 输入。
                    outputs, sample_count, extra_outputs = self._forward_with_optional_tta(images)
                    
                    # 计算损失
                    if loss_fn is not None:
                        # Keep evaluation loss path consistent with training path.
                        loss = self._compute_loss(loss_fn, outputs, labels, extra_targets=extra_targets)
                        total_loss += loss.item() * sample_count
                    
                    # 计算准确率
                    correct, total = self.strategy.calculate_metrics(outputs, labels)
                    total_correct += correct
                    total_samples += total
                    # strategy 决定如何从 outputs 中拿到最终预测标签。
                    predictions = self.strategy.get_predictions(outputs)
                    all_predictions.append(predictions.detach().cpu().numpy())
                    all_labels.append(labels.detach().cpu().numpy())

                    if extra_outputs:
                        if "extra_corrects" not in locals():
                            extra_corrects = {k: 0 for k in extra_outputs.keys()}
                            extra_totals = {k: 0 for k in extra_outputs.keys()}
                        for mode, out in extra_outputs.items():
                            c, t = self.strategy.calculate_metrics(out, labels)
                            extra_corrects[mode] += c
                            extra_totals[mode] += t

                    if loss_fn is not None and extra_targets is not None and hasattr(loss_fn, "compute_aux_metrics"):
                        batch_aux_metrics = loss_fn.compute_aux_metrics(outputs, extra_targets=extra_targets)
                        if batch_aux_metrics:
                            if "aux_metric_sums" not in locals():
                                aux_metric_sums = {}
                            for metric_name, metric_value in batch_aux_metrics.items():
                                aux_metric_sums[metric_name] = aux_metric_sums.get(metric_name, 0.0) + (
                                    float(metric_value) * sample_count
                                )
                    
                    # 更新进度条
                    current_acc = total_correct / total_samples if total_samples > 0 else 0
                    pbar.set_postfix({
                        'acc': f'{current_acc:.2%}',
                        'loss': f'{total_loss/total_samples:.4f}' if loss_fn else 'N/A'
                    })

                    # 显存清理
                    if batch_idx % 5 == 0:
                        del images, labels, outputs
                        if 'extra_outputs' in locals():
                            del extra_outputs
                        torch.cuda.empty_cache()
            
            # 循环结束后彻底清理
            gc.collect()
            torch.cuda.empty_cache()
            
            # 计算平均指标
            predictions_array = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([], dtype=np.int64)
            labels_array = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.int64)
            metrics = {
                'accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
                'correct': total_correct,
                'total': total_samples,
                'mae': _mean_absolute_error(labels_array, predictions_array),
                'qwk': _quadratic_weighted_kappa(labels_array, predictions_array),
                # 对有序分类补充一个更宽松的正确率指标。
                'plus_minus_one_accuracy': _plus_minus_one_accuracy(labels_array, predictions_array),
            }

            if "extra_corrects" in locals():
                for mode in extra_corrects.keys():
                    metrics[f"accuracy_{mode}"] = extra_corrects[mode] / extra_totals[mode] if extra_totals[mode] > 0 else 0.0

            if "aux_metric_sums" in locals():
                for metric_name, metric_sum in aux_metric_sums.items():
                    metrics[metric_name] = metric_sum / total_samples if total_samples > 0 else 0.0
            
            if loss_fn is not None:
                metrics['loss'] = total_loss / total_samples if total_samples > 0 else 0.0
            
            if self.logger:
                self.logger.info(
                    "evaluation_completed",
                    accuracy=metrics['accuracy'],
                    total_samples=total_samples,
                    loss=metrics.get('loss', 'N/A')
                )
            
            return metrics
            
        except Exception as e:
            if isinstance(e, EvaluationError):
                raise
            raise EvaluationError(
                f"Evaluation failed: {e}",
                original_exception=e
            )
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        评估单个批次
        
        参数:
            images: 图像张量 [B, C, H, W]
            labels: 标签张量 [B]
            loss_fn: 损失函数（可选）
            
        返回:
            Dict[str, float]: 批次评估指标
                - accuracy: 准确率
                - loss: 损失值（如果提供了 loss_fn）
                - correct: 正确预测数量
                - total: 样本数量
        """
        try:
            self.model.eval()
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                outputs, _, _ = self._forward_with_optional_tta(images)
                
                # 计算指标
                correct, total = self.strategy.calculate_metrics(outputs, labels)
                
                metrics = {
                    'accuracy': correct / total if total > 0 else 0.0,
                    'plus_minus_one_accuracy': _plus_minus_one_accuracy(
                        labels.detach().cpu().numpy(),
                        self.strategy.get_predictions(outputs).detach().cpu().numpy(),
                    ),
                    'correct': correct,
                    'total': total
                }
                
                if loss_fn is not None:
                    loss = self._compute_loss(loss_fn, outputs, labels)
                    metrics['loss'] = loss.item()
                
                return metrics
                
        except Exception as e:
            raise EvaluationError(
                f"Batch evaluation failed: {e}",
                original_exception=e
            )
    
    def get_predictions(
        self,
        dataloader: DataLoader,
        return_labels: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        获取模型在数据集上的预测结果
        
        参数:
            dataloader: 数据加载器
            return_labels: 是否返回真实标签
            
        返回:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                - predictions: 预测结果 [N]
                - labels: 真实标签 [N]（如果 return_labels=True）
        """
        try:
            self.model.eval()
            
            all_predictions = []
            all_labels = [] if return_labels else None
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Getting predictions", leave=False):
                    # 解包批次数据
                    if len(batch) == 4:
                        images, labels, _, _ = batch
                    elif len(batch) == 3:
                        images, labels, _ = batch
                    else:
                        images, labels = batch
                    
                    images = images.to(self.device)
                    
                    # 前向传播（支持 5-crop TTA）
                    outputs, _, _ = self._forward_with_optional_tta(images)
                    
                    # 获取预测
                    predictions = self.strategy.get_predictions(outputs)
                    all_predictions.append(predictions.cpu().numpy())
                    
                    if return_labels:
                        all_labels.append(labels.numpy())
            
            # 合并所有批次
            predictions = np.concatenate(all_predictions, axis=0)
            labels_array = np.concatenate(all_labels, axis=0) if return_labels else None
            
            return predictions, labels_array
            
        except Exception as e:
            raise EvaluationError(
                f"Failed to get predictions: {e}",
                original_exception=e
            )
    
    def compute_confusion_matrix(
        self,
        dataloader: DataLoader,
        num_classes: Optional[int] = None
    ) -> np.ndarray:
        """
        计算混淆矩阵
        
        参数:
            dataloader: 数据加载器
            num_classes: 类别数量（可选，自动推断）
            
        返回:
            np.ndarray: 混淆矩阵 [num_classes, num_classes]
                行表示真实标签，列表示预测标签
        """
        try:
            predictions, labels = self.get_predictions(dataloader, return_labels=True)
            
            # 推断类别数量
            if num_classes is None:
                num_classes = max(labels.max(), predictions.max()) + 1
            
            # 初始化混淆矩阵
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
            
            # 填充混淆矩阵
            for true_label, pred_label in zip(labels, predictions):
                confusion_matrix[true_label, pred_label] += 1
            
            if self.logger:
                self.logger.info(
                    "confusion_matrix_computed",
                    num_classes=num_classes,
                    total_samples=len(labels)
                )
            
            return confusion_matrix
            
        except Exception as e:
            raise EvaluationError(
                f"Failed to compute confusion matrix: {e}",
                original_exception=e
            )
    
    def compute_per_class_accuracy(
        self,
        dataloader: DataLoader
    ) -> Dict[int, float]:
        """
        计算每个类别的准确率
        
        参数:
            dataloader: 数据加载器
            
        返回:
            Dict[int, float]: 每个类别的准确率
                键为类别索引，值为准确率 [0, 1]
        """
        try:
            confusion_matrix = self.compute_confusion_matrix(dataloader)
            
            per_class_acc = {}
            for class_idx in range(confusion_matrix.shape[0]):
                # 该类别的总样本数
                total = confusion_matrix[class_idx, :].sum()
                
                if total > 0:
                    # 该类别的正确预测数
                    correct = confusion_matrix[class_idx, class_idx]
                    per_class_acc[class_idx] = correct / total
                else:
                    per_class_acc[class_idx] = 0.0
            
            if self.logger:
                self.logger.info(
                    "per_class_accuracy_computed",
                    num_classes=len(per_class_acc),
                    avg_accuracy=np.mean(list(per_class_acc.values()))
                )
            
            return per_class_acc
            
        except Exception as e:
            raise EvaluationError(
                f"Failed to compute per-class accuracy: {e}",
                original_exception=e
            )
    
    def compute_detailed_metrics(
        self,
        dataloader: DataLoader
    ) -> Dict[str, Any]:
        """
        计算详细的评估指标
        
        参数:
            dataloader: 数据加载器
            
        返回:
            Dict[str, Any]: 详细指标字典
                - accuracy: 总体准确率
                - per_class_accuracy: 每类准确率
                - confusion_matrix: 混淆矩阵
                - predictions: 预测结果
                - labels: 真实标签
        """
        try:
            # 获取预测和标签
            predictions, labels = self.get_predictions(dataloader, return_labels=True)
            
            # 计算混淆矩阵
            confusion_matrix = self.compute_confusion_matrix(dataloader)
            
            # 计算每类准确率
            per_class_acc = self.compute_per_class_accuracy(dataloader)
            
            # 计算总体准确率
            accuracy = (predictions == labels).mean()
            
            metrics = {
                'accuracy': float(accuracy),
                'plus_minus_one_accuracy': _plus_minus_one_accuracy(labels, predictions),
                'per_class_accuracy': per_class_acc,
                'confusion_matrix': confusion_matrix,
                'predictions': predictions,
                'labels': labels,
                'num_samples': len(labels),
                'num_classes': len(per_class_acc)
            }
            
            if self.logger:
                self.logger.info(
                    "detailed_metrics_computed",
                    accuracy=accuracy,
                    num_samples=len(labels),
                    num_classes=len(per_class_acc)
                )
            
            return metrics
            
        except Exception as e:
            raise EvaluationError(
                f"Failed to compute detailed metrics: {e}",
                original_exception=e
            )
