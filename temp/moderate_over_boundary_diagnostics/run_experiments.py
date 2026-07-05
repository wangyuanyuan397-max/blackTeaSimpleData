"""针对 Moderate(2) 与 Over(3) 边界的五组临时诊断实验。"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# 当前脚本位于 PROJECT_ROOT/temp/moderate_over_boundary_diagnostics。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (  # noqa: E402
    ImageFolderWithPaths,
    build_patch_eval_transform,
    build_patch_train_transform,
)
from src.models.backbones.efficientnet_probabilistic_ordinal import (  # noqa: E402
    EfficientNetV2SProbabilisticOrdinalBackbone,
)
from src.models.backbones.efficientnet_stage_probe import (  # noqa: E402
    EfficientNetV2SStageProbeBackbone,
)


# ---------------------------------------------------------------------------
# PyCharm 右键运行配置：所有临时结果都留在本文件夹，不改动仓库原训练布局。
# ---------------------------------------------------------------------------
TEMP_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "datasets_split_patches"
RESULTS_ROOT = TEMP_ROOT / "results"
PYCHARM_DEVICE = "auto"
PYCHARM_DRY_RUN = False
PYCHARM_KEEP_PTH = False

CLASS_TO_IDX = {"pre": 0, "slight": 1, "moderate": 2, "over": 3}
FOUR_CLASS_NAMES = ["pre", "slight", "moderate", "over"]
BINARY_CLASS_NAMES = ["moderate", "over"]
T23_CANDIDATES = (0.65, 0.70, 0.75, 0.80, 0.85)


@dataclass(frozen=True)
class ExperimentSpec:
    """一个实验只描述相对于公共训练设置发生变化的单一变量。"""

    name: str
    task: str
    model_type: str
    loss_type: str
    moderate_over_weight: float = 1.0
    margin: float = 0.0
    margin_weight: float = 0.0


# 如只想跑其中一部分，可直接注释列表项；右键运行无需填写命令行参数。
EXPERIMENT_LIST = (
    ExperimentSpec("stage5_ce", "four_class", "stage5", "ce"),
    ExperimentSpec(
        "binary_moderate_over_stage5_ce",
        "moderate_over_binary",
        "stage5",
        "ce",
    ),
    ExperimentSpec(
        "stage5_ce_mo_weight1.5",
        "four_class",
        "stage5",
        "mo_weighted_ce",
        moderate_over_weight=1.5,
    ),
    ExperimentSpec(
        "stage5_ce_mo_logit_margin_m0.3_lam0.1",
        "four_class",
        "stage5",
        "mo_logit_margin",
        margin=0.3,
        margin_weight=0.1,
    ),
    ExperimentSpec(
        "logistic_normal_cdf_t23_search",
        "four_class",
        "logistic_normal",
        "logistic_normal_nll",
    ),
)


@dataclass(frozen=True)
class RuntimeSettings:
    seed: int = 2026
    epochs: int = 150
    batch_size: int = 32
    val_batch_size: int = 64
    test_batch_size: int = 64
    num_workers: int = 4
    patience: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 2
    min_learning_rate: float = 1e-6
    image_size: int = 224


SETTINGS = RuntimeSettings()


class ModerateOverDataset(Dataset):
    """从固定 split 中只保留标签 2/3，并映射为 Moderate=0、Over=1。"""

    def __init__(self, base_dataset: ImageFolderWithPaths) -> None:
        self.base_dataset = base_dataset
        self.indices = [
            index
            for index, target in enumerate(base_dataset.targets)
            if int(target) in (2, 3)
        ]
        self.targets = [int(base_dataset.targets[index]) - 2 for index in self.indices]
        self.classes = list(BINARY_CLASS_NAMES)
        self.class_to_idx = {"moderate": 0, "over": 1}
        if not self.indices:
            raise ValueError("过滤后没有 Moderate/Over 样本。")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, original_label, path = self.base_dataset[self.indices[index]]
        return image, int(original_label) - 2, path


class Stage5Classifier(nn.Module):
    """EfficientNetV2-S Stage1～5 → GAP → Linear。"""

    def __init__(self, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.backbone = EfficientNetV2SStageProbeBackbone(
            pretrained=pretrained,
            output_stage=5,
            trainable_stages=(1, 2, 3, 4, 5),
        )
        self.classifier = nn.Linear(self.backbone.out_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(images))


def set_random_seed(seed: int) -> None:
    """统一 Python、NumPy、CPU 和 CUDA 随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境没有可用 GPU。")
    return torch.device(name)


def build_datasets(spec: ExperimentSpec):
    """固定使用原项目 train/val/test，不重新随机拆分。"""
    train_transform = build_patch_train_transform(SETTINGS.image_size)
    eval_transform = build_patch_eval_transform(SETTINGS.image_size)
    split_datasets = {}
    for split_name, transform in (
        ("train", train_transform),
        ("val", eval_transform),
        ("test", eval_transform),
    ):
        base_dataset = ImageFolderWithPaths(
            root=DATASET_ROOT / split_name,
            transform=transform,
            class_to_idx=CLASS_TO_IDX,
        )
        split_datasets[split_name] = (
            ModerateOverDataset(base_dataset)
            if spec.task == "moderate_over_binary"
            else base_dataset
        )
    return (
        split_datasets["train"],
        split_datasets["val"],
        split_datasets["test"],
    )


def build_loaders(spec: ExperimentSpec):
    train_dataset, val_dataset, test_dataset = build_datasets(spec)
    generator = torch.Generator().manual_seed(SETTINGS.seed)
    common = {
        "num_workers": SETTINGS.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": SETTINGS.num_workers > 0,
        "drop_last": False,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=SETTINGS.batch_size,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=SETTINGS.val_batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=SETTINGS.test_batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader


def build_model(spec: ExperimentSpec, pretrained: bool = True) -> nn.Module:
    if spec.model_type == "stage5":
        num_classes = 2 if spec.task == "moderate_over_binary" else 4
        return Stage5Classifier(num_classes=num_classes, pretrained=pretrained)
    if spec.model_type == "logistic_normal":
        return EfficientNetV2SProbabilisticOrdinalBackbone(
            num_classes=4,
            pretrained=pretrained,
            distribution="logistic_normal",
            use_ce_head=False,
            final_dropout=0.0,
            min_sigma=0.05,
            max_sigma=5.0,
            output_stage=6,
        )
    raise ValueError(f"未知 model_type：{spec.model_type}")


def primary_logits(outputs) -> torch.Tensor:
    return outputs[0] if isinstance(outputs, tuple) else outputs


def compute_loss(outputs, targets: torch.Tensor, spec: ExperimentSpec) -> torch.Tensor:
    """分别实现普通 CE、2/3 类别加权、2/3 logit margin 和概率 NLL。"""
    logits = primary_logits(outputs)
    if spec.loss_type == "ce":
        return F.cross_entropy(logits, targets)
    if spec.loss_type == "mo_weighted_ce":
        class_weights = torch.tensor(
            [1.0, 1.0, spec.moderate_over_weight, spec.moderate_over_weight],
            device=logits.device,
            dtype=logits.dtype,
        )
        return F.cross_entropy(logits, targets, weight=class_weights)
    if spec.loss_type == "mo_logit_margin":
        classification_loss = F.cross_entropy(logits, targets)
        boundary_mask = (targets == 2) | (targets == 3)
        if not torch.any(boundary_mask):
            return classification_loss
        selected_logits = logits[boundary_mask]
        selected_targets = targets[boundary_mask]
        signed_gap = torch.where(
            selected_targets == 2,
            selected_logits[:, 2] - selected_logits[:, 3],
            selected_logits[:, 3] - selected_logits[:, 2],
        )
        margin_loss = F.relu(spec.margin - signed_gap).mean()
        return classification_loss + spec.margin_weight * margin_loss
    if spec.loss_type == "logistic_normal_nll":
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            raise ValueError("Logistic-Normal 模型必须返回 (logits, auxiliary)。")
        stage_probabilities = outputs[1]["stage_probs"]
        return F.nll_loss(torch.log(stage_probabilities.clamp_min(1e-8)), targets)
    raise ValueError(f"未知 loss_type：{spec.loss_type}")


def extract_continuous_scores(outputs) -> torch.Tensor | None:
    """Logistic-Normal 使用 sigmoid(mu) 作为 [0,1] 连续发酵分数。"""
    if not isinstance(outputs, tuple) or len(outputs) < 2:
        return None
    mean_proxy = outputs[1].get("mean_proxy")
    return mean_proxy if torch.is_tensor(mean_proxy) else None


def decode_with_t23(scores: np.ndarray, threshold_23: float) -> np.ndarray:
    """固定前两条边界 0.25/0.50，仅改变 Moderate/Over 边界。"""
    predictions = np.zeros(scores.shape[0], dtype=np.int64)
    predictions[scores >= 0.25] = 1
    predictions[scores >= 0.50] = 2
    predictions[scores >= threshold_23] = 3
    return predictions


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """统一计算整体、逐类、边界和方向误分指标。"""
    num_classes = len(class_names)
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    qwk = cohen_kappa_score(
        labels,
        predictions,
        labels=list(range(num_classes)),
        weights="quadratic",
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mae": float(np.mean(np.abs(labels - predictions))),
        "qwk": float(qwk) if np.isfinite(qwk) else 0.0,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
        "class_wise": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(class_names)
        },
    }
    if num_classes == 4:
        for left, right in ((0, 1), (1, 2), (2, 3)):
            mask = (labels == left) | (labels == right)
            # 只筛真实标签；predictions 仍是原始四分类结果，不重新二选一。
            result[f"acc_{left}_{right}"] = (
                float(np.mean(predictions[mask] == labels[mask]))
                if np.any(mask)
                else None
            )
        for source, target in ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)):
            result[f"error_{source}_to_{target}"] = int(matrix[source, target])
        result["far_error_count"] = int(np.sum(np.abs(labels - predictions) >= 2))
    else:
        result["moderate_over_binary_accuracy"] = result["accuracy"]
        result["moderate_to_over_count"] = int(matrix[0, 1])
        result["over_to_moderate_count"] = int(matrix[1, 0])
    return result


def build_scheduler(optimizer: AdamW):
    """复用正式实验的 2 epoch warmup + cosine 设置。"""
    cosine_epochs = max(1, SETTINGS.epochs - SETTINGS.warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=SETTINGS.min_learning_rate,
    )
    if SETTINGS.warmup_epochs <= 0:
        return cosine
    warmup = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=SETTINGS.warmup_epochs,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[SETTINGS.warmup_epochs],
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    spec: ExperimentSpec,
    device: torch.device,
    description: str,
) -> dict[str, Any]:
    """评估时同时收集离散预测和 Logistic-Normal 连续分数。"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels_all = []
    predictions_all = []
    continuous_scores = []
    started = time.perf_counter()
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc=description, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = compute_loss(outputs, labels, spec)
            logits = primary_logits(outputs)
            predictions = logits.argmax(dim=1)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            labels_all.append(labels.detach().cpu())
            predictions_all.append(predictions.detach().cpu())
            scores = extract_continuous_scores(outputs)
            if scores is not None:
                continuous_scores.append(scores.detach().cpu())
    elapsed = time.perf_counter() - started
    labels_array = torch.cat(labels_all).numpy()
    predictions_array = torch.cat(predictions_all).numpy()
    return {
        "loss": total_loss / max(total_samples, 1),
        "labels": labels_array,
        "predictions": predictions_array,
        "continuous_scores": (
            torch.cat(continuous_scores).numpy() if continuous_scores else None
        ),
        "elapsed_seconds": elapsed,
        "samples": total_samples,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    spec: ExperimentSpec,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float]:
    """统一用验证集 Accuracy 选最佳 epoch，并将最佳权重仅保存在内存。"""
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=SETTINGS.learning_rate,
        weight_decay=SETTINGS.weight_decay,
    )
    scheduler = build_scheduler(optimizer)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_accuracy = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_started = time.perf_counter()

    for epoch in range(SETTINGS.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        progress = tqdm(
            train_loader,
            desc=f"{spec.name} {epoch + 1}/{SETTINGS.epochs}",
            leave=False,
        )
        for images, labels, _ in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = compute_loss(outputs, labels, spec)
            loss.backward()
            optimizer.step()

            logits = primary_logits(outputs)
            predictions = logits.argmax(dim=1)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int(predictions.eq(labels).sum().item())
            total_samples += batch_size
            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.4f}",
            )

        validation = evaluate_model(
            model,
            val_loader,
            spec,
            device,
            description=f"Validating {epoch + 1}",
        )
        val_accuracy = float(
            accuracy_score(validation["labels"], validation["predictions"])
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / max(total_samples, 1),
            "train_accuracy": total_correct / max(total_samples, 1),
            "val_loss": float(validation["loss"]),
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(
            f"[{spec.name}] epoch={epoch + 1:03d} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= SETTINGS.patience:
            print(f"[{spec.name}] early stopping at epoch {epoch + 1}")
            break

    if best_state is None:
        raise RuntimeError(f"{spec.name} 没有生成最佳模型状态。")
    training_seconds = time.perf_counter() - training_started
    return best_state, history, best_epoch, training_seconds


def search_t23_threshold(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    """只使用验证集选择阈值；测试集绝不参与选择。"""
    rows = []
    for threshold in T23_CANDIDATES:
        predictions = decode_with_t23(validation_scores, threshold)
        metrics = compute_metrics(
            validation_labels,
            predictions,
            FOUR_CLASS_NAMES,
        )
        rows.append(
            {
                "threshold_23": threshold,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "val_mae": metrics["mae"],
                "val_qwk": metrics["qwk"],
                "val_acc_2_3": metrics["acc_2_3"],
            }
        )
    # 先最大化验证 Accuracy，再看 Macro-F1；完全相同时优先接近默认 0.75。
    best = max(
        rows,
        key=lambda row: (
            row["val_accuracy"],
            row["val_macro_f1"],
            -abs(row["threshold_23"] - 0.75),
        ),
    )
    return float(best["threshold_23"]), rows


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_csv(run_directory: Path, metrics: dict[str, Any]) -> None:
    class_names = metrics["class_names"]
    matrix = metrics["confusion_matrix"]
    with (run_directory / "confusion_matrix.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])


def write_experiment_report(
    run_directory: Path,
    spec: ExperimentSpec,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> Path:
    cards = "".join(
        f"<div><span>{html.escape(key)}</span><strong>{value:.6f}</strong></div>"
        for key, value in metrics.items()
        if key in {
            "accuracy",
            "macro_f1",
            "mae",
            "qwk",
            "acc_0_1",
            "acc_1_2",
            "acc_2_3",
            "moderate_over_binary_accuracy",
        }
        and isinstance(value, (int, float))
    )
    class_names = metrics["class_names"]
    matrix = metrics["confusion_matrix"]
    matrix_header = "".join(f"<th>预测 {html.escape(name)}</th>" for name in class_names)
    matrix_rows = "".join(
        "<tr><th>真实 "
        + html.escape(class_name)
        + "</th>"
        + "".join(f"<td>{int(value)}</td>" for value in row)
        + "</tr>"
        for class_name, row in zip(class_names, matrix)
    )
    threshold_note = ""
    if metrics.get("selected_threshold_23") is not None:
        threshold_note = (
            "<p><strong>验证集选出的 t23：</strong>"
            f"{float(metrics['selected_threshold_23']):.2f}。"
            "测试集未参与阈值选择。</p>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(spec.name)}</title>
<style>
body{{max-width:1000px;margin:36px auto;padding:0 18px;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#172033}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.cards div{{padding:14px;border:1px solid #dfe5ef;border-radius:8px}}.cards span{{display:block;color:#667085}}.cards strong{{font-size:20px}}
table{{width:100%;border-collapse:collapse;margin-top:12px}}th,td{{border:1px solid #dfe5ef;padding:8px;text-align:center}}th{{background:#f4f6fa}}
</style></head><body>
<h1>{html.escape(spec.name)}</h1>{threshold_note}<section class="cards">{cards}</section>
<h2>测试集混淆矩阵</h2><table><thead><tr><th></th>{matrix_header}</tr></thead><tbody>{matrix_rows}</tbody></table>
<p><a href="metrics.json">指标 JSON</a> · <a href="history.csv">训练历史</a> · <a href="confusion_matrix.csv">混淆矩阵 CSV</a></p>
<p>最佳 epoch：{metrics['best_epoch']}；训练耗时：{metrics['training_time_seconds']:.2f}s；测试推理耗时：{metrics['test_inference_seconds']:.2f}s。</p>
</body></html>"""
    report_path = run_directory / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def run_experiment(
    spec: ExperimentSpec,
    device: torch.device,
    batch_directory: Path,
    keep_pth: bool,
) -> dict[str, Any]:
    """完整训练一个实验，并把所有产物限制在当前临时批次目录。"""
    print("\n" + "=" * 80)
    print(f"开始临时实验：{spec.name}")
    print("=" * 80)
    set_random_seed(SETTINGS.seed)
    run_directory = batch_directory / spec.name
    run_directory.mkdir(parents=True, exist_ok=False)
    save_json(
        run_directory / "config.json",
        {
            "experiment": asdict(spec),
            "runtime": asdict(SETTINGS),
            "dataset_root": str(DATASET_ROOT),
            "threshold_23_candidates": list(T23_CANDIDATES),
            "keep_pth": keep_pth,
        },
    )

    train_loader, val_loader, test_loader = build_loaders(spec)
    model = build_model(spec, pretrained=True).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    best_state, history, best_epoch, training_seconds = train_model(
        model,
        train_loader,
        val_loader,
        spec,
        device,
    )
    model.load_state_dict(best_state, strict=True)
    if keep_pth:
        torch.save(
            {"model_state_dict": best_state, "best_epoch": best_epoch},
            run_directory / "best_model.pth",
        )

    class_names = (
        BINARY_CLASS_NAMES
        if spec.task == "moderate_over_binary"
        else FOUR_CLASS_NAMES
    )
    validation = evaluate_model(model, val_loader, spec, device, "Final validation")
    testing = evaluate_model(model, test_loader, spec, device, "Final testing")

    if spec.model_type == "logistic_normal":
        if validation["continuous_scores"] is None or testing["continuous_scores"] is None:
            raise RuntimeError("Logistic-Normal 没有返回连续 mean_proxy。")
        selected_threshold, threshold_rows = search_t23_threshold(
            validation["labels"],
            validation["continuous_scores"],
        )
        write_rows(run_directory / "threshold_23_search.csv", threshold_rows)
        fixed_metrics = compute_metrics(
            testing["labels"],
            testing["predictions"],
            FOUR_CLASS_NAMES,
        )
        test_predictions = decode_with_t23(
            testing["continuous_scores"],
            selected_threshold,
        )
        metrics = compute_metrics(testing["labels"], test_predictions, FOUR_CLASS_NAMES)
        metrics["selected_threshold_23"] = selected_threshold
        metrics["fixed_cdf_argmax_accuracy"] = fixed_metrics["accuracy"]
        metrics["fixed_cdf_argmax_macro_f1"] = fixed_metrics["macro_f1"]
        metrics["threshold_selection_source"] = "validation_only"
    else:
        metrics = compute_metrics(
            testing["labels"],
            testing["predictions"],
            class_names,
        )

    metrics.update(
        {
            "model_name": spec.name,
            "best_epoch": best_epoch,
            "training_time_seconds": training_seconds,
            "test_inference_seconds": testing["elapsed_seconds"],
            "test_ms_per_sample": (
                testing["elapsed_seconds"] * 1000.0 / max(testing["samples"], 1)
            ),
            "parameters_total": total_parameters,
            "parameters_trainable": trainable_parameters,
            "keep_pth": keep_pth,
        }
    )
    save_json(run_directory / "metrics.json", metrics)
    write_rows(run_directory / "history.csv", history)
    save_confusion_csv(run_directory, metrics)
    report_path = write_experiment_report(run_directory, spec, metrics, history)
    result = {
        "model_name": spec.name,
        "status": "success",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mae": metrics["mae"],
        "qwk": metrics["qwk"],
        "acc_2_3": metrics.get("acc_2_3"),
        "selected_threshold_23": metrics.get("selected_threshold_23"),
        "run_directory": str(run_directory),
        "report_path": str(report_path),
    }
    del model, best_state, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_batch_report(batch_directory: Path, results: list[dict[str, Any]]) -> Path:
    write_rows(batch_directory / "summary.csv", results)
    rows = []
    for result in results:
        report_path = result.get("report_path")
        report_link = (
            Path(report_path).relative_to(batch_directory).as_posix()
            if report_path
            else ""
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result['model_name']))}</td>"
            f"<td>{html.escape(str(result['status']))}</td>"
            f"<td>{result.get('accuracy', '')}</td>"
            f"<td>{result.get('macro_f1', '')}</td>"
            f"<td>{result.get('acc_2_3', '')}</td>"
            f"<td>{result.get('selected_threshold_23', '')}</td>"
            + (f"<td><a href=\{report_link}\>报告</a></td>" if report_link else "<td></td>")
            + "</tr>"
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Moderate/Over 边界诊断</title><style>body{{max-width:1000px;margin:36px auto;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:9px;text-align:center}}th{{background:#f4f6fa}}</style></head>
<body><h1>Moderate ↔ Over 边界诊断</h1><table><thead><tr><th>实验</th><th>状态</th><th>Accuracy</th><th>Macro-F1</th><th>Acc_2_3</th><th>t23</th><th>报告</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"""
    path = batch_directory / "summary.html"
    path.write_text(document, encoding="utf-8")
    return path


def select_experiments(names: list[str] | None) -> list[ExperimentSpec]:
    by_name = {spec.name: spec for spec in EXPERIMENT_LIST}
    if not names:
        return list(EXPERIMENT_LIST)
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"未知实验名：{unknown}")
    return [by_name[name] for name in names]


def run_dry_check(specs: list[ExperimentSpec], device: torch.device) -> None:
    """不下载预训练权重、不创建结果目录，只验证数据、模型和损失链路。"""
    print(f"数据集：{DATASET_ROOT}")
    for spec in specs:
        train_dataset, val_dataset, test_dataset = build_datasets(spec)
        set_random_seed(SETTINGS.seed)
        model = build_model(spec, pretrained=False).to(device)
        model.train()
        images = torch.randn(2, 3, 64, 64, device=device)
        labels = (
            torch.tensor([0, 1], device=device)
            if spec.task == "moderate_over_binary"
            else torch.tensor([2, 3], device=device)
        )
        outputs = model(images)
        loss = compute_loss(outputs, labels, spec)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{spec.name} dry-run loss 非有限值。")
        output_shape = tuple(primary_logits(outputs).shape)
        print(
            f"{spec.name}: train/val/test="
            f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}, "
            f"output={output_shape}, loss={float(loss.detach()):.6f}, PASS"
        )
        del model, outputs, loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    synthetic_scores = np.array([0.10, 0.40, 0.60, 0.79])
    assert decode_with_t23(synthetic_scores, 0.75).tolist() == [0, 1, 2, 3]
    print("dry-run 完成：数据过滤、模型输出、损失与 t23 解码均通过。")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="临时 Moderate/Over 边界诊断实验。")
    parser.add_argument("--experiments", nargs="+", help="只运行指定实验名。")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=PYCHARM_DEVICE,
    )
    parser.add_argument("--dry-run", action="store_true", default=PYCHARM_DRY_RUN)
    parser.add_argument("--keep-pth", action="store_true", default=PYCHARM_KEEP_PTH)
    parser.add_argument("--list-experiments", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    specs = select_experiments(args.experiments)
    if args.list_experiments:
        for spec in specs:
            print(spec.name)
        return
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"固定数据集不存在：{DATASET_ROOT}")
    device = resolve_device(args.device)
    if args.dry_run:
        run_dry_check(specs, device)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_directory = RESULTS_ROOT / f"batch_{timestamp}"
    batch_directory.mkdir(parents=True, exist_ok=False)
    results = []
    for spec in specs:
        try:
            results.append(
                run_experiment(spec, device, batch_directory, bool(args.keep_pth))
            )
        except Exception as error:
            failure_directory = batch_directory / spec.name
            failure_directory.mkdir(parents=True, exist_ok=True)
            (failure_directory / "failure.txt").write_text(
                f"{type(error).__name__}: {error}",
                encoding="utf-8",
            )
            results.append(
                {
                    "model_name": spec.name,
                    "status": "failed",
                    "accuracy": None,
                    "macro_f1": None,
                    "mae": None,
                    "qwk": None,
                    "acc_2_3": None,
                    "selected_threshold_23": None,
                    "run_directory": str(failure_directory),
                    "report_path": "",
                }
            )
            print(f"{spec.name} 失败：{error}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary_path = write_batch_report(batch_directory, results)
    print(f"\n全部临时实验结束：{summary_path}")
    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
