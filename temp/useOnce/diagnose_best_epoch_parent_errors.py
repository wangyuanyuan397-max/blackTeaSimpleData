"""datasets_01234 最佳epoch混淆矩阵、父图投票和高置信错误图诊断。

直接右键运行即可。默认重新训练 fixed_efficientnet_v2_s，并在内存中保存：
最高 val_acc、最低 val_loss、最高 val_macro_f1、最高 val_qwk 四个最佳状态。
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error


# =========================
# 右键运行前主要改这里
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_CONFIG_PATH = Path("configs/fixed_split_01234_train.yaml")
MODEL_CONFIG_PATH = Path("configs/fixed_split_01234_models/fixed_efficientnet_v2_s.yaml")

# 408 版本数据集：裁剪出 408x408 后不再提前缩放成 224x224。
DATASET_ROOT = Path("datasets_01234_408")

# 关键：如果这里不改成 408，公共 YAML 里的 transform 仍会把图片 resize 回 224。
# 如果想切回旧的 224 数据集，把 DATASET_ROOT 改回 datasets_01234，并把这里改成 224。
INPUT_IMAGE_SIZE = 408

OUTPUT_ROOT = Path("temp/useOnce/best_epoch_parent_diagnostics_408_runs")

DEVICE_NAME = "auto"  # auto / cuda / cpu
EPOCHS_OVERRIDE: Optional[int] = None  # 快速试脚本可改成 2；正式诊断保持 None
NUM_WORKERS_OVERRIDE: Optional[int] = 0  # 服务器可改成 4
BATCH_SIZE_OVERRIDE: Optional[int] = None
VAL_BATCH_SIZE_OVERRIDE: Optional[int] = None
TEST_BATCH_SIZE_OVERRIDE: Optional[int] = None
RANDOM_SEED = 2026

DIAGNOSTIC_SPLIT = "val"  # val / test / train
PREDICTION_STATE_KEY = "best_val_acc"
TOP_ERROR_COUNT = 100
SAVE_BEST_STATE_PTH = False

BEST_TRACKERS = {
    "best_val_acc": {"metric": "accuracy", "mode": "max"},
    "best_val_loss": {"metric": "loss", "mode": "min"},
    "best_val_macro_f1": {"metric": "macro_f1", "mode": "max"},
    "best_val_qwk": {"metric": "qwk", "mode": "max"},
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models  # noqa: E402,F401
from src.engine import ComponentBuilder  # noqa: E402
from src.schemas import TrainingConfig  # noqa: E402


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 cuda，但当前没有可用 CUDA。")
    return torch.device(name)


def load_yaml(path: Path) -> Dict[str, Any]:
    with resolve_project_path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是字典：{path}")
    return data


def make_run_dir(model_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_project_path(OUTPUT_ROOT) / f"{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def build_config(run_dir: Path, device: torch.device) -> Tuple[str, TrainingConfig]:
    common = load_yaml(COMMON_CONFIG_PATH)
    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    model_name = str(model_cfg.get("name") or MODEL_CONFIG_PATH.stem)
    cfg = copy.deepcopy(common)
    cfg["run_name"] = model_name
    cfg["description"] = f"useOnce diagnostics; model={MODEL_CONFIG_PATH.as_posix()}"
    cfg["output_dir"] = str(run_dir)
    cfg["use_wandb"] = False
    cfg["enable_google_drive_upload"] = False
    cfg["random_seed"] = int(cfg.get("random_seed", RANDOM_SEED))
    cfg["model"] = copy.deepcopy(model_cfg["model"])
    if isinstance(model_cfg.get("loss"), dict):
        cfg["loss"] = copy.deepcopy(model_cfg["loss"])
    for section in ("train", "optimizer", "scheduler"):
        if isinstance(model_cfg.get(section), dict):
            cfg.setdefault(section, {}).update(copy.deepcopy(model_cfg[section]))
    cfg["data"]["root"] = str(resolve_project_path(DATASET_ROOT))
    cfg["data"]["class_to_idx"] = copy.deepcopy(cfg.pop("class_to_idx"))
    for transform_key in ("train_transform", "eval_transform", "test_transform"):
        transform_cfg = cfg["data"].get(transform_key)
        if isinstance(transform_cfg, dict):
            transform_cfg["image_size"] = int(INPUT_IMAGE_SIZE)
    cfg["train"]["device"] = device.type
    cfg["train"]["keep_pth_files"] = False
    if EPOCHS_OVERRIDE is not None:
        cfg["train"]["epochs"] = int(EPOCHS_OVERRIDE)
    if NUM_WORKERS_OVERRIDE is not None:
        cfg["train"]["num_workers"] = int(NUM_WORKERS_OVERRIDE)
    if BATCH_SIZE_OVERRIDE is not None:
        cfg["train"]["batch_size"] = int(BATCH_SIZE_OVERRIDE)
    if VAL_BATCH_SIZE_OVERRIDE is not None:
        cfg["train"]["val_batch_size"] = int(VAL_BATCH_SIZE_OVERRIDE)
    if TEST_BATCH_SIZE_OVERRIDE is not None:
        cfg["train"]["test_batch_size"] = int(TEST_BATCH_SIZE_OVERRIDE)
    cfg.pop("experiment_name", None)
    cfg.pop("dataset_root", None)
    cfg.pop("runs_root", None)
    return model_name, TrainingConfig(**cfg)


def extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, tuple):
        return outputs[0]
    if isinstance(outputs, dict):
        for key in ("logits", "cls_logits", "output"):
            if key in outputs:
                return outputs[key]
        raise ValueError(f"模型输出是 dict，但没有 logits 键：{outputs.keys()}")
    return outputs


def compute_loss(loss_fn: nn.Module, outputs: Any, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    try:
        return loss_fn(outputs, labels)
    except Exception:
        return loss_fn(logits, labels)


def unpack_batch(batch_data: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor, Optional[Sequence[str]]]:
    if len(batch_data) >= 3:
        return batch_data[0], batch_data[1], batch_data[2]
    if len(batch_data) == 2:
        return batch_data[0], batch_data[1], None
    raise ValueError(f"无法识别 batch 格式，元素数量={len(batch_data)}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = len(loader)
    for batch_index, batch_data in enumerate(loader):
        images, labels, _paths = unpack_batch(batch_data)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        outputs = model(images)
        logits = extract_logits(outputs)
        loss = compute_loss(loss_fn, outputs, logits, labels)
        (loss / accumulation_steps).backward()
        is_update = ((batch_index + 1) % accumulation_steps == 0) or ((batch_index + 1) == total_batches)
        if is_update:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size
    return {"loss": total_loss / max(1, total_samples), "accuracy": total_correct / max(1, total_samples)}


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], loss: Optional[float], num_classes: int) -> Dict[str, float]:
    labels = list(range(num_classes))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    records: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_data in loader:
            images, labels, paths = unpack_batch(batch_data)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            outputs = model(images)
            logits = extract_logits(outputs)
            loss = compute_loss(loss_fn, outputs, logits, labels)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            cpu_labels = labels.detach().cpu().tolist()
            cpu_preds = preds.detach().cpu().tolist()
            cpu_confs = confs.detach().cpu().tolist()
            cpu_probs = probs.detach().cpu().tolist()
            if paths is None:
                paths = [""] * batch_size
            y_true.extend(int(x) for x in cpu_labels)
            y_pred.extend(int(x) for x in cpu_preds)
            for path, true_label, pred_label, confidence, prob_vector in zip(paths, cpu_labels, cpu_preds, cpu_confs, cpu_probs):
                row = {
                    "path": str(path),
                    "true_label": int(true_label),
                    "true_class": class_names[int(true_label)],
                    "pred_label": int(pred_label),
                    "pred_class": class_names[int(pred_label)],
                    "confidence": float(confidence),
                }
                for index, class_name in enumerate(class_names):
                    row[f"prob_{class_name}"] = float(prob_vector[index])
                records.append(row)
    metrics = compute_metrics(y_true, y_pred, total_loss / max(1, total_samples), len(class_names))
    return metrics, records


def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def better(current: float, previous: Optional[float], mode: str) -> bool:
    if previous is None:
        return True
    return current > previous if mode == "max" else current < previous


def update_best(best: Dict[str, Dict[str, Any]], epoch: int, val_metrics: Dict[str, float], model: nn.Module) -> None:
    for key, spec in BEST_TRACKERS.items():
        metric = str(spec["metric"])
        value = float(val_metrics[metric])
        old = best.get(key, {}).get("metric_value")
        if better(value, old, str(spec["mode"])):
            best[key] = {
                "epoch": int(epoch),
                "metric_name": metric,
                "metric_value": value,
                "val_metrics": copy.deepcopy(val_metrics),
                "state_dict": clone_state(model),
            }


def save_history(run_dir: Path, history: Sequence[Dict[str, Any]]) -> None:
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "val_mae", "val_qwk", "seconds"]
    write_rows(run_dir / "epoch_history.csv", history, fields)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[0, 1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, [row["val_macro_f1"] for row in history])
    axes[1, 0].set_title("Validation Macro-F1")
    axes[1, 1].plot(epochs, [row["val_qwk"] for row in history])
    axes[1, 1].set_title("Validation QWK")
    for ax in axes.ravel():
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "epoch_history_curves.png")
    plt.close(fig)


def save_confusion_csv(path: Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    rows = []
    for i, true_name in enumerate(class_names):
        row = {"true\\pred": true_name}
        for j, pred_name in enumerate(class_names):
            value = matrix[i, j]
            row[pred_name] = float(value) if matrix.dtype.kind == "f" else int(value)
        rows.append(row)
    write_rows(path, rows)


def plot_confusion(path: Path, matrix: np.ndarray, class_names: Sequence[str], title: str, normalize: bool) -> None:
    shown = matrix.astype(float)
    if normalize:
        sums = shown.sum(axis=1, keepdims=True)
        shown = np.divide(shown, sums, out=np.zeros_like(shown), where=sums != 0)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    image = ax.imshow(shown, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = f"{shown[i, j]:.2f}\n({int(matrix[i, j])})" if normalize else str(int(matrix[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion_summary(matrix: np.ndarray, class_names: Sequence[str]) -> Dict[str, Any]:
    adjacent = 0
    far = 0
    pair_errors = {}
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            count = int(matrix[i, j])
            if i == j or count == 0:
                continue
            pair_errors[f"{true_name}->{pred_name}"] = count
            if abs(i - j) == 1:
                adjacent += count
            else:
                far += count
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    errors = total - correct
    return {
        "total": total,
        "correct": correct,
        "errors": errors,
        "adjacent_errors": adjacent,
        "far_errors": far,
        "adjacent_error_ratio_among_errors": adjacent / max(1, errors),
        "far_error_ratio_among_errors": far / max(1, errors),
        "pair_errors": dict(sorted(pair_errors.items(), key=lambda x: x[1], reverse=True)),
    }


def save_best_confusions(
    run_dir: Path,
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
    best: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    out = run_dir / "best_epoch_confusion_matrices"
    out.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    summary = {}
    for key, record in best.items():
        model.load_state_dict(record["state_dict"])
        metrics, records = evaluate(model, val_loader, loss_fn, device, class_names)
        y_true = [row["true_label"] for row in records]
        y_pred = [row["pred_label"] for row in records]
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        normalized = matrix.astype(float)
        sums = normalized.sum(axis=1, keepdims=True)
        normalized = np.divide(normalized, sums, out=np.zeros_like(normalized), where=sums != 0)
        save_confusion_csv(out / f"{key}_raw.csv", matrix, class_names)
        save_confusion_csv(out / f"{key}_normalized.csv", normalized, class_names)
        plot_confusion(out / f"{key}_raw.png", matrix, class_names, f"{key} epoch {record['epoch']} raw", False)
        plot_confusion(out / f"{key}_normalized.png", matrix, class_names, f"{key} epoch {record['epoch']} normalized", True)
        if SAVE_BEST_STATE_PTH:
            torch.save(record["state_dict"], out / f"{key}_epoch{record['epoch']}.pth")
        summary[key] = {
            "epoch": int(record["epoch"]),
            "selection_metric": record["metric_name"],
            "selection_metric_value": float(record["metric_value"]),
            "reevaluated_metrics": metrics,
            "confusion_summary": confusion_summary(matrix, class_names),
        }
    save_json(out / "best_epoch_confusion_summary.json", summary)
    return summary


def parse_parent_and_crop(path: str) -> Tuple[str, str]:
    stem = Path(path).stem
    match = re.match(r"(?P<parent>.+)__random\d+_(?P<crop>\d+)$", stem)
    if match:
        return match.group("parent"), match.group("crop")
    return stem, ""


def majority_vote(counts: Sequence[int]) -> int:
    return int(max(range(len(counts)), key=lambda i: (counts[i], -i)))


def save_parent_diagnostics(run_dir: Path, records: Sequence[Dict[str, Any]], class_names: Sequence[str]) -> Dict[str, Any]:
    out = run_dir / f"parent_vote_{DIAGNOSTIC_SPLIT}_{PREDICTION_STATE_KEY}"
    out.mkdir(parents=True, exist_ok=True)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        parent_id, crop_index = parse_parent_and_crop(row["path"])
        row = copy.deepcopy(row)
        row["parent_id"] = parent_id
        row["crop_index"] = crop_index
        groups[parent_id].append(row)

    parent_rows = []
    parent_true = []
    parent_pred = []
    num_classes = len(class_names)
    for parent_id, rows in sorted(groups.items()):
        true_label = Counter(int(row["true_label"]) for row in rows).most_common(1)[0][0]
        pred_counts = [0] * num_classes
        for row in rows:
            pred_counts[int(row["pred_label"])] += 1
        vote_label = majority_vote(pred_counts)
        consistency = max(pred_counts) / max(1, len(rows))
        item: Dict[str, Any] = {
            "parent_id": parent_id,
            "true_class": class_names[true_label],
            "true_label": int(true_label),
            "n_crops": len(rows),
        }
        for i, name in enumerate(class_names):
            item[f"pred_{name}"] = pred_counts[i]
        item.update({
            "majority_class": class_names[vote_label],
            "majority_label": vote_label,
            "consistency": consistency,
            "consistency_percent": f"{consistency * 100:.1f}%",
            "vote_correct": int(vote_label == true_label),
            "vote_error_distance": abs(vote_label - true_label),
            "adjacent_vote_error": int(vote_label != true_label and abs(vote_label - true_label) == 1),
            "far_vote_error": int(abs(vote_label - true_label) > 1),
        })
        parent_rows.append(item)
        parent_true.append(int(true_label))
        parent_pred.append(int(vote_label))

    fields = [
        "parent_id", "true_class", "true_label", "n_crops",
        *[f"pred_{name}" for name in class_names],
        "majority_class", "majority_label", "consistency", "consistency_percent",
        "vote_correct", "vote_error_distance", "adjacent_vote_error", "far_vote_error",
    ]
    write_rows(out / "parent_prediction_consistency.csv", parent_rows, fields)

    crop_true = [int(row["true_label"]) for row in records]
    crop_pred = [int(row["pred_label"]) for row in records]
    crop_metrics = compute_metrics(crop_true, crop_pred, None, num_classes)
    parent_metrics = compute_metrics(parent_true, parent_pred, None, num_classes)
    write_rows(out / "crop_vs_parent_vote_metrics.csv", [
        {"level": "crop", **crop_metrics},
        {"level": "parent_majority_vote", **parent_metrics},
    ])
    parent_matrix = confusion_matrix(parent_true, parent_pred, labels=list(range(num_classes)))
    save_confusion_csv(out / "parent_vote_confusion_raw.csv", parent_matrix, class_names)
    plot_confusion(out / "parent_vote_confusion_raw.png", parent_matrix, class_names, "Parent majority vote confusion", False)
    consistencies = [float(row["consistency"]) for row in parent_rows]
    summary = {
        "split": DIAGNOSTIC_SPLIT,
        "state_key": PREDICTION_STATE_KEY,
        "num_crop_records": len(records),
        "num_parent_images": len(parent_rows),
        "mean_parent_consistency": float(np.mean(consistencies)) if consistencies else 0.0,
        "median_parent_consistency": float(np.median(consistencies)) if consistencies else 0.0,
        "parents_below_50pct_consistency": sum(1 for x in consistencies if x < 0.5),
        "crop_metrics": crop_metrics,
        "parent_vote_metrics": parent_metrics,
        "parent_vote_confusion_summary": confusion_summary(parent_matrix, class_names),
    }
    save_json(out / "parent_vote_summary.json", summary)
    return summary


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("._-") or "item"


def annotate_error_image(source: Path, target: Path, lines: Sequence[str]) -> None:
    image = Image.open(source).convert("RGB").resize((224, 224))
    canvas = Image.new("RGB", (224, 296), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = 228
    for line in lines:
        draw.text((4, y), line, fill=(0, 0, 0))
        y += 16
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=95)


def save_contact_sheet(paths: Sequence[Path], target: Path, cols: int = 5) -> None:
    if not paths:
        return
    cell_w, cell_h = 224, 296
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        sheet.paste(image, ((index % cols) * cell_w, (index // cols) * cell_h))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=95)


def save_high_confidence_errors(run_dir: Path, records: Sequence[Dict[str, Any]], class_names: Sequence[str]) -> Dict[str, Any]:
    out = run_dir / f"high_confidence_errors_{DIAGNOSTIC_SPLIT}_{PREDICTION_STATE_KEY}"
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    errors = [row for row in records if int(row["true_label"]) != int(row["pred_label"])]
    errors = sorted(errors, key=lambda row: float(row["confidence"]), reverse=True)
    top = errors[:TOP_ERROR_COUNT]
    rows = []
    annotated_paths = []
    for rank, row in enumerate(top, start=1):
        source = Path(row["path"])
        parent_id, crop_index = parse_parent_and_crop(row["path"])
        image_name = (
            f"rank{rank:03d}_true{row['true_class']}_pred{row['pred_class']}_"
            f"conf{float(row['confidence']):.3f}_{safe_name(parent_id)}_crop{crop_index}.jpg"
        )
        target = image_dir / image_name
        annotate_error_image(source, target, [
            f"rank {rank:03d} conf {float(row['confidence']):.3f}",
            f"true {row['true_class']} -> pred {row['pred_class']}",
            f"parent {parent_id}",
            f"crop {crop_index}",
        ])
        annotated_paths.append(target)
        csv_row = {
            "rank": rank,
            "true_class": row["true_class"],
            "pred_class": row["pred_class"],
            "confidence": float(row["confidence"]),
            "parent_id": parent_id,
            "crop_index": crop_index,
            "source_path": str(source),
            "annotated_image": str(target),
        }
        for class_name in class_names:
            csv_row[f"prob_{class_name}"] = row[f"prob_{class_name}"]
        rows.append(csv_row)
    fields = [
        "rank", "true_class", "pred_class", "confidence", "parent_id", "crop_index",
        "source_path", "annotated_image", *[f"prob_{name}" for name in class_names],
    ]
    write_rows(out / "high_confidence_errors_top100.csv", rows, fields)
    save_contact_sheet(annotated_paths, out / "high_confidence_errors_contact_sheet.jpg")
    summary = {
        "split": DIAGNOSTIC_SPLIT,
        "state_key": PREDICTION_STATE_KEY,
        "total_errors": len(errors),
        "exported_errors": len(top),
        "output_dir": str(out),
    }
    save_json(out / "high_confidence_errors_summary.json", summary)
    return summary


def main() -> None:
    set_random_seed(RANDOM_SEED)
    device = resolve_device(DEVICE_NAME)
    model_name_for_dir = str(load_yaml(MODEL_CONFIG_PATH).get("name", MODEL_CONFIG_PATH.stem))
    run_dir = make_run_dir(model_name_for_dir)
    model_name, config = build_config(run_dir, device)

    builder = ComponentBuilder(config, device, logger=None)
    train_loader, val_loader, test_loader = builder.build_dataloaders()
    model, _strategy = builder.build_model()
    loss_fn = builder.build_loss()
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn.to(device)
    optimizer = builder.build_optimizer(model)
    scheduler = builder.build_scheduler(optimizer)

    split_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    if DIAGNOSTIC_SPLIT not in split_loaders:
        raise ValueError(f"DIAGNOSTIC_SPLIT 必须是 train/val/test，当前={DIAGNOSTIC_SPLIT}")
    class_names = list(train_loader.dataset.classes)
    epochs = int(config.train.epochs)
    accumulation_steps = int(getattr(config.train, "accumulation_steps", 1) or 1)

    print("=" * 100)
    print("最佳epoch混淆矩阵 + 父图投票 + 高置信错误图诊断开始")
    print(f"model: {model_name}")
    print(f"model_config: {MODEL_CONFIG_PATH}")
    print(f"dataset: {resolve_project_path(DATASET_ROOT)}")
    print(f"classes: {class_names}")
    print(f"device: {device}")
    print(f"epochs: {epochs}")
    print(f"output: {run_dir}")
    print("=" * 100)

    history: List[Dict[str, Any]] = []
    best: Dict[str, Dict[str, Any]] = {}
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device, accumulation_steps)
        val_metrics, _ = evaluate(model, val_loader, loss_fn, device, class_names)
        if scheduler is not None:
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(val_metrics["loss"])
        seconds = time.time() - start
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_mae": val_metrics["mae"],
            "val_qwk": val_metrics["qwk"],
            "seconds": seconds,
        }
        history.append(row)
        update_best(best, epoch, val_metrics, model)
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={row['train_loss']:.5f} train_acc={row['train_acc']:.4f} | "
            f"val_loss={row['val_loss']:.5f} val_acc={row['val_acc']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f} val_qwk={row['val_qwk']:.4f} | "
            f"time={seconds:.1f}s"
        )

    save_history(run_dir, history)
    best_summary = save_best_confusions(run_dir, model, val_loader, loss_fn, device, class_names, best)

    if PREDICTION_STATE_KEY not in best:
        raise RuntimeError(f"找不到 PREDICTION_STATE_KEY={PREDICTION_STATE_KEY} 对应的最佳状态。")
    model.load_state_dict(best[PREDICTION_STATE_KEY]["state_dict"])
    diag_metrics, diag_records = evaluate(model, split_loaders[DIAGNOSTIC_SPLIT], loss_fn, device, class_names)
    parent_summary = save_parent_diagnostics(run_dir, diag_records, class_names)
    error_summary = save_high_confidence_errors(run_dir, diag_records, class_names)

    final_summary = {
        "model_name": model_name,
        "common_config_path": COMMON_CONFIG_PATH.as_posix(),
        "model_config_path": MODEL_CONFIG_PATH.as_posix(),
        "dataset_root": str(resolve_project_path(DATASET_ROOT)),
        "input_image_size": INPUT_IMAGE_SIZE,
        "class_names": class_names,
        "device": str(device),
        "epochs": epochs,
        "best_epoch_summary": best_summary,
        "diagnostic_split": DIAGNOSTIC_SPLIT,
        "prediction_state_key": PREDICTION_STATE_KEY,
        "diagnostic_split_metrics": diag_metrics,
        "parent_summary": parent_summary,
        "high_confidence_error_summary": error_summary,
    }
    save_json(run_dir / "diagnostics_summary.json", final_summary)

    print("=" * 100)
    print("诊断完成")
    print(f"最佳epoch混淆矩阵：{run_dir / 'best_epoch_confusion_matrices'}")
    print(f"父图一致率/投票：{run_dir / f'parent_vote_{DIAGNOSTIC_SPLIT}_{PREDICTION_STATE_KEY}'}")
    print(f"高置信错误图：{run_dir / f'high_confidence_errors_{DIAGNOSTIC_SPLIT}_{PREDICTION_STATE_KEY}'}")
    print(f"总览JSON：{run_dir / 'diagnostics_summary.json'}")
    for key, record in best.items():
        print(f"{key}: epoch={record['epoch']}, {record['metric_name']}={record['metric_value']:.6f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
