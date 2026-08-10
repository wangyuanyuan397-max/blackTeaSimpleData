"""MixNet-S 的 30-patch 父图投票、patch disagreement 和错误图诊断。

默认使用 datasets_01234_grid30_408 的 manifest 恢复“1 张原图 -> 30 个
patch”映射。设置 CHECKPOINT_PATH 后可直接诊断已有 checkpoint，无需重新
训练；只有显式设置 RETRAIN_MODEL=True 才会进入训练流程。
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
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error


# =========================
# 右键运行前主要改这里
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_CONFIG_PATH = Path("configs/fixed_split_01234_grid30_408_train.yaml")
MODEL_CONFIG_PATH = Path("configs/fixed_split_01234_models/fixed_timm_mixnet_s.yaml")

# 每张原图按 5x6 固定网格裁成 30 个 408x408 patch。
DATASET_ROOT = Path("datasets_01234_grid30_408")
CROP_MANIFEST_PATH = DATASET_ROOT / "grid30_crop_manifest.csv"
EXPECTED_PATCHES_PER_PARENT = 30

# 关键：如果这里不改成 408，公共 YAML 里的 transform 仍会把图片 resize 回 224。
# 如果想切回旧的 224 数据集，把 DATASET_ROOT 改回 datasets_01234，并把这里改成 224。
INPUT_IMAGE_SIZE = 408

OUTPUT_ROOT = Path("temp/useOnce/best_epoch_parent_diagnostics_408_runs")

# 优先直接诊断已有 MixNet-S checkpoint。支持裸 state_dict，以及包含
# model_state_dict/state_dict/model 键的 checkpoint。None 时会尝试在
# runs_01234_grid30_408 下自动寻找最新的 MixNet-S best_model.pth。
CHECKPOINT_PATH: Optional[Path] = None
RETRAIN_MODEL = False

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
SAVE_EPOCH_TOP_LOSS_RECORDS = True
TOP_LOSS_RECORDS_PER_EPOCH = 100

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


def resolve_checkpoint_path() -> Optional[Path]:
    """Resolve an explicit checkpoint or auto-discover the newest standard MixNet-S run."""
    if CHECKPOINT_PATH is not None:
        checkpoint_path = resolve_project_path(CHECKPOINT_PATH)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到 CHECKPOINT_PATH：{checkpoint_path}")
        return checkpoint_path

    runs_root = PROJECT_ROOT / "runs_01234_grid30_408"
    if not runs_root.is_dir():
        return None
    candidates = [
        path
        for path in runs_root.rglob("best_model.pth")
        if "mixnet_s" in path.parent.name.lower()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path) -> Dict[str, Any]:
    """Load the common checkpoint formats used by this repository."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict: Any = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"checkpoint 中没有可用的 model state_dict：{checkpoint_path}")

    for prefix in ("module.", "model."):
        if all(str(key).startswith(prefix) for key in state_dict):
            state_dict = {str(key)[len(prefix):]: value for key, value in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint 与 {MODEL_CONFIG_PATH} 构建的模型不兼容：{checkpoint_path}\n{exc}"
        ) from exc

    metadata: Dict[str, Any] = {"checkpoint_path": str(checkpoint_path)}
    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            metadata["epoch"] = int(checkpoint["epoch"])
        if isinstance(checkpoint.get("metrics"), dict):
            metadata["saved_metrics"] = checkpoint["metrics"]
    return metadata


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


def build_config(run_dir: Path, device: torch.device, use_pretrained: bool = True) -> Tuple[str, TrainingConfig]:
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
    backbone_cfg = cfg["model"].get("backbone") if isinstance(cfg["model"], dict) else None
    if isinstance(backbone_cfg, dict) and not use_pretrained:
        # checkpoint 会完整覆盖权重，构建模型时无需下载 ImageNet 预训练权重。
        backbone_cfg["pretrained"] = False
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
    max_sample_loss = 0.0
    max_abs_logit = 0.0
    abs_logit_values: List[float] = []
    with torch.no_grad():
        for batch_data in loader:
            images, labels, paths = unpack_batch(batch_data)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            outputs = model(images)
            logits = extract_logits(outputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("验证阶段出现 NaN 或 Inf logits。")
            loss = compute_loss(loss_fn, outputs, logits, labels)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            if not torch.isfinite(per_sample_loss).all():
                raise RuntimeError("验证阶段出现 NaN 或 Inf per-sample loss。")
            abs_logits = logits.detach().abs()
            max_sample_loss = max(max_sample_loss, float(per_sample_loss.max().item()))
            max_abs_logit = max(max_abs_logit, float(abs_logits.max().item()))
            abs_logit_values.extend(float(value) for value in abs_logits.flatten().cpu().tolist())
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            cpu_labels = labels.detach().cpu().tolist()
            cpu_preds = preds.detach().cpu().tolist()
            cpu_confs = confs.detach().cpu().tolist()
            cpu_probs = probs.detach().cpu().tolist()
            cpu_losses = per_sample_loss.detach().cpu().tolist()
            cpu_logits = logits.detach().cpu().tolist()
            if paths is None:
                paths = [""] * batch_size
            y_true.extend(int(x) for x in cpu_labels)
            y_pred.extend(int(x) for x in cpu_preds)
            for path, true_label, pred_label, confidence, prob_vector, sample_loss, logit_vector in zip(
                paths,
                cpu_labels,
                cpu_preds,
                cpu_confs,
                cpu_probs,
                cpu_losses,
                cpu_logits,
            ):
                row = {
                    "path": str(path),
                    "true_label": int(true_label),
                    "true_class": class_names[int(true_label)],
                    "pred_label": int(pred_label),
                    "pred_class": class_names[int(pred_label)],
                    "confidence": float(confidence),
                    "sample_loss": float(sample_loss),
                    "sample_max_abs_logit": float(max(abs(value) for value in logit_vector)),
                    "true_logit": float(logit_vector[int(true_label)]),
                    "pred_logit": float(logit_vector[int(pred_label)]),
                }
                for index, class_name in enumerate(class_names):
                    row[f"prob_{class_name}"] = float(prob_vector[index])
                    row[f"logit_{class_name}"] = float(logit_vector[index])
                records.append(row)
    metrics = compute_metrics(y_true, y_pred, total_loss / max(1, total_samples), len(class_names))
    metrics["max_sample_loss"] = float(max_sample_loss)
    metrics["max_abs_logit"] = float(max_abs_logit)
    metrics["p99_abs_logit"] = float(np.percentile(abs_logit_values, 99)) if abs_logit_values else 0.0
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
    fields = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "val_mae",
        "val_qwk",
        "val_max_sample_loss",
        "val_max_abs_logit",
        "val_p99_abs_logit",
        "seconds",
    ]
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


def save_epoch_top_loss_records(run_dir: Path, epoch: int, records: Sequence[Dict[str, Any]]) -> None:
    """保存当前 epoch 验证集中单样本 CE loss 最大的样本，定位 loss 爆炸来源。"""
    if not SAVE_EPOCH_TOP_LOSS_RECORDS:
        return
    top_records = sorted(records, key=lambda row: float(row.get("sample_loss", 0.0)), reverse=True)[
        :TOP_LOSS_RECORDS_PER_EPOCH
    ]
    if not top_records:
        return
    output_dir = run_dir / "epoch_top_loss_records"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / f"epoch_{epoch:03d}_top_loss.csv", top_records)


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


def parse_parent_and_crop(path: str) -> Tuple[str, int]:
    """Parse supported patch names without silently treating a patch as its own parent."""
    stem = Path(path).stem
    patterns = (
        r"^(?P<parent>.+)__(?:grid30|random\d+)_(?P<crop>\d+)$",
        r"^(?P<parent>.+)_patch_?(?P<crop>\d+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group("parent"), int(match.group("crop"))
    raise ValueError(
        f"无法从 patch 文件名恢复 parent/patch_index：{path}。"
        "请提供 crop manifest，或使用 *_patch_01 / *__grid30_01 / *__random55_001 命名。"
    )


def crop_path_keys(path: Path | str) -> List[str]:
    """Create move-tolerant lookup aliases for one crop path."""
    raw = str(path).replace("\\", "/").casefold()
    candidate = Path(path)
    keys = [raw, candidate.name.casefold()]
    try:
        keys.append(str(candidate.resolve()).replace("\\", "/").casefold())
    except OSError:
        pass
    return list(dict.fromkeys(keys))


def load_crop_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    """Index source-image and patch metadata from the dataset crop manifest."""
    manifest_path = resolve_project_path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 crop manifest：{manifest_path}")

    dataset_root = resolve_project_path(DATASET_ROOT)
    index: Dict[str, Dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"source_image_id", "target_relpath"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"crop manifest 缺少列 {sorted(missing)}：{manifest_path}")
        patch_column = "patch_index" if "patch_index" in (reader.fieldnames or []) else "crop_index"
        if patch_column not in (reader.fieldnames or []):
            raise ValueError(f"crop manifest 缺少 patch_index/crop_index 列：{manifest_path}")

        for manifest_row in reader:
            metadata: Dict[str, Any] = {
                "parent_id": str(manifest_row["source_image_id"]),
                "patch_index": int(manifest_row[patch_column]),
                "source_relpath": str(manifest_row.get("source_relpath", "")),
                "target_relpath": str(manifest_row["target_relpath"]),
                "mapping_source": "manifest",
            }
            for column in ("split", "time_code", "row_index", "column_index", "left", "top", "right", "bottom"):
                if column in manifest_row and manifest_row[column] != "":
                    metadata[column] = manifest_row[column]

            aliases: List[str] = []
            aliases.extend(crop_path_keys(dataset_root / manifest_row["target_relpath"]))
            aliases.extend(crop_path_keys(manifest_row["target_relpath"]))
            if manifest_row.get("target_path"):
                aliases.extend(crop_path_keys(manifest_row["target_path"]))
            for alias in dict.fromkeys(aliases):
                existing = index.get(alias)
                if existing is not None and (
                    existing["parent_id"], existing["patch_index"]
                ) != (metadata["parent_id"], metadata["patch_index"]):
                    raise ValueError(f"crop manifest 路径别名冲突：{alias}")
                index[alias] = metadata
    if not index:
        raise ValueError(f"crop manifest 为空：{manifest_path}")
    return index


def resolve_crop_metadata(path: str, manifest_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for key in crop_path_keys(path):
        if key in manifest_index:
            return copy.deepcopy(manifest_index[key])
    parent_id, patch_index = parse_parent_and_crop(path)
    return {
        "parent_id": parent_id,
        "patch_index": patch_index,
        "source_relpath": "",
        "target_relpath": "",
        "mapping_source": "filename_fallback",
    }


def majority_vote(counts: Sequence[int]) -> int:
    return int(max(range(len(counts)), key=lambda i: (counts[i], -i)))


def save_parent_diagnostics(
    run_dir: Path,
    records: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    state_key: str,
    manifest_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    out = run_dir / f"parent_vote_{DIAGNOSTIC_SPLIT}_{state_key}"
    out.mkdir(parents=True, exist_ok=True)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    patch_rows: List[Dict[str, Any]] = []
    for row in records:
        metadata = resolve_crop_metadata(str(row["path"]), manifest_index)
        enriched = copy.deepcopy(row)
        enriched.update(metadata)
        enriched["patch_id"] = Path(str(row["path"])).stem
        enriched["ground_truth"] = str(row["true_class"])
        for class_name in class_names:
            enriched[f"p{class_name}"] = float(row[f"prob_{class_name}"])
        groups[str(enriched["parent_id"])].append(enriched)
        patch_rows.append(enriched)

    def patch_sort_key(row: Dict[str, Any]) -> Tuple[str, int]:
        return str(row["parent_id"]), int(row["patch_index"])

    patch_rows.sort(key=patch_sort_key)
    patch_fields = [
        "patch_id", "parent_id", "patch_index", "ground_truth", "true_label",
        "pred_class", "pred_label", *[f"p{name}" for name in class_names],
        "confidence", "source_relpath", "target_relpath", "mapping_source", "path",
    ]
    patch_export_rows = [
        {field: row.get(field, "") for field in patch_fields}
        for row in patch_rows
    ]
    write_rows(out / "patch_predictions.csv", patch_export_rows, patch_fields)

    group_sizes = Counter(len(rows) for rows in groups.values())
    invalid_parents: List[Dict[str, Any]] = []
    for parent_id, rows in sorted(groups.items()):
        patch_indices = [int(row["patch_index"]) for row in rows]
        labels = sorted({int(row["true_label"]) for row in rows})
        source_relpaths = sorted({str(row["source_relpath"]) for row in rows if row["source_relpath"]})
        unique_indices = sorted(set(patch_indices))
        contiguous = bool(unique_indices) and unique_indices == list(
            range(unique_indices[0], unique_indices[0] + len(unique_indices))
        )
        if (
            len(rows) != EXPECTED_PATCHES_PER_PARENT
            or len(unique_indices) != EXPECTED_PATCHES_PER_PARENT
            or not contiguous
            or len(labels) != 1
            or len(source_relpaths) > 1
        ):
            invalid_parents.append({
                "parent_id": parent_id,
                "n_patches": len(rows),
                "n_unique_patch_indices": len(unique_indices),
                "patch_indices": unique_indices,
                "true_labels": labels,
                "source_relpaths": source_relpaths,
            })

    mapping_source_counts = Counter(str(row["mapping_source"]) for row in patch_rows)
    mapping_audit = {
        "manifest_path": str(resolve_project_path(CROP_MANIFEST_PATH)),
        "expected_patches_per_parent": EXPECTED_PATCHES_PER_PARENT,
        "num_patch_records": len(patch_rows),
        "num_parent_images": len(groups),
        "patch_count_distribution": {str(key): value for key, value in sorted(group_sizes.items())},
        "mapping_source_counts": dict(sorted(mapping_source_counts.items())),
        "all_parents_valid": not invalid_parents,
        "invalid_parents": invalid_parents,
    }
    save_json(out / "parent_mapping_audit.json", mapping_audit)
    if invalid_parents:
        examples = ", ".join(
            f"{row['parent_id']}({row['n_patches']})" for row in invalid_parents[:10]
        )
        raise RuntimeError(
            f"父图-patch 映射校验失败：{len(invalid_parents)} 张父图不是恰好 "
            f"{EXPECTED_PATCHES_PER_PARENT} 个唯一连续 patch。示例：{examples}。"
            f"详见 {out / 'parent_mapping_audit.json'}"
        )

    parent_rows = []
    parent_true = []
    parent_pred = []
    num_classes = len(class_names)
    for parent_id, rows in sorted(groups.items()):
        true_label = Counter(int(row["true_label"]) for row in rows).most_common(1)[0][0]
        source_relpath = str(rows[0].get("source_relpath", ""))
        pred_counts = [0] * num_classes
        for row in rows:
            pred_counts[int(row["pred_label"])] += 1
        vote_label = majority_vote(pred_counts)
        consistency = max(pred_counts) / len(rows)
        disagreement = 1.0 - consistency
        item: Dict[str, Any] = {
            "parent_id": parent_id,
            "source_relpath": source_relpath,
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
            "disagreement": disagreement,
            "disagreement_percent": f"{disagreement * 100:.1f}%",
            "vote_correct": int(vote_label == true_label),
            "vote_error_distance": abs(vote_label - true_label),
            "adjacent_vote_error": int(vote_label != true_label and abs(vote_label - true_label) == 1),
            "far_vote_error": int(abs(vote_label - true_label) > 1),
        })
        parent_rows.append(item)
        parent_true.append(int(true_label))
        parent_pred.append(int(vote_label))

    fields = [
        "parent_id", "source_relpath", "true_class", "true_label", "n_crops",
        *[f"pred_{name}" for name in class_names],
        "majority_class", "majority_label", "consistency", "consistency_percent",
        "disagreement", "disagreement_percent",
        "vote_correct", "vote_error_distance", "adjacent_vote_error", "far_vote_error",
    ]
    write_rows(out / "parent_prediction_consistency.csv", parent_rows, fields)

    disagreement_rows: List[Dict[str, Any]] = []
    disagreement_by_stage: Dict[str, Any] = {}
    for label, class_name in enumerate(class_names):
        values = [
            float(row["disagreement"])
            for row in parent_rows
            if int(row["true_label"]) == label
        ]
        stage_row: Dict[str, Any] = {
            "ground_truth": class_name,
            "true_label": label,
            "num_parent_images": len(values),
            "mean_disagreement": float(np.mean(values)) if values else None,
            "std_disagreement": float(np.std(values)) if values else None,
            "median_disagreement": float(np.median(values)) if values else None,
            "min_disagreement": float(np.min(values)) if values else None,
            "max_disagreement": float(np.max(values)) if values else None,
        }
        disagreement_rows.append(stage_row)
        disagreement_by_stage[class_name] = stage_row
    write_rows(out / "patch_disagreement_by_stage.csv", disagreement_rows)
    save_json(out / "patch_disagreement_by_stage.json", disagreement_by_stage)

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
    disagreements = [float(row["disagreement"]) for row in parent_rows]
    summary = {
        "split": DIAGNOSTIC_SPLIT,
        "state_key": state_key,
        "num_crop_records": len(records),
        "num_parent_images": len(parent_rows),
        "expected_patches_per_parent": EXPECTED_PATCHES_PER_PARENT,
        "all_parents_have_expected_patch_count": True,
        "mean_parent_consistency": float(np.mean(consistencies)) if consistencies else 0.0,
        "median_parent_consistency": float(np.median(consistencies)) if consistencies else 0.0,
        "mean_patch_disagreement": float(np.mean(disagreements)) if disagreements else 0.0,
        "median_patch_disagreement": float(np.median(disagreements)) if disagreements else 0.0,
        "patch_disagreement_by_ground_truth": disagreement_by_stage,
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


def save_high_confidence_errors(
    run_dir: Path,
    records: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    state_key: str,
    manifest_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    out = run_dir / f"high_confidence_errors_{DIAGNOSTIC_SPLIT}_{state_key}"
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    errors = [row for row in records if int(row["true_label"]) != int(row["pred_label"])]
    errors = sorted(errors, key=lambda row: float(row["confidence"]), reverse=True)
    top = errors[:TOP_ERROR_COUNT]
    rows = []
    annotated_paths = []
    for rank, row in enumerate(top, start=1):
        source = Path(row["path"])
        crop_metadata = resolve_crop_metadata(str(row["path"]), manifest_index)
        parent_id = str(crop_metadata["parent_id"])
        crop_index = int(crop_metadata["patch_index"])
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
        "state_key": state_key,
        "total_errors": len(errors),
        "exported_errors": len(top),
        "output_dir": str(out),
    }
    save_json(out / "high_confidence_errors_summary.json", summary)
    return summary


def main() -> None:
    set_random_seed(RANDOM_SEED)
    device = resolve_device(DEVICE_NAME)
    checkpoint_path = resolve_checkpoint_path()
    if checkpoint_path is None and not RETRAIN_MODEL:
        raise FileNotFoundError(
            "未找到可用的 MixNet-S checkpoint，而 RETRAIN_MODEL=False。\n"
            "请将 CHECKPOINT_PATH 设为已有 best_model.pth，或将 checkpoint 放入 "
            "runs_01234_grid30_408/<mixnet_s_run>/best_model.pth。\n"
            "如果确实要重新训练，再显式设置 RETRAIN_MODEL=True。"
        )
    model_name_for_dir = str(load_yaml(MODEL_CONFIG_PATH).get("name", MODEL_CONFIG_PATH.stem))
    run_dir = make_run_dir(model_name_for_dir)
    model_name, config = build_config(run_dir, device, use_pretrained=checkpoint_path is None)

    builder = ComponentBuilder(config, device, logger=None)
    train_loader, val_loader, test_loader = builder.build_dataloaders()
    model, _strategy = builder.build_model()
    loss_fn = builder.build_loss()
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn.to(device)

    split_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    if DIAGNOSTIC_SPLIT not in split_loaders:
        raise ValueError(f"DIAGNOSTIC_SPLIT 必须是 train/val/test，当前={DIAGNOSTIC_SPLIT}")
    class_names = list(train_loader.dataset.classes)
    epochs = int(config.train.epochs)
    accumulation_steps = int(getattr(config.train, "accumulation_steps", 1) or 1)
    manifest_index = load_crop_manifest(CROP_MANIFEST_PATH)
    state_key = "checkpoint" if checkpoint_path is not None else PREDICTION_STATE_KEY

    print("=" * 100)
    print("MixNet-S 30-patch 父图投票 + patch disagreement + 高置信错误图诊断开始")
    print(f"model: {model_name}")
    print(f"model_config: {MODEL_CONFIG_PATH}")
    print(f"dataset: {resolve_project_path(DATASET_ROOT)}")
    print(f"crop_manifest: {resolve_project_path(CROP_MANIFEST_PATH)}")
    print(f"checkpoint: {checkpoint_path if checkpoint_path is not None else '不使用（将重新训练）'}")
    print(f"classes: {class_names}")
    print(f"device: {device}")
    print(f"epochs: {0 if checkpoint_path is not None else epochs}")
    print(f"output: {run_dir}")
    print("=" * 100)

    history: List[Dict[str, Any]] = []
    best: Dict[str, Dict[str, Any]] = {}
    checkpoint_metadata: Optional[Dict[str, Any]] = None
    if checkpoint_path is not None:
        checkpoint_metadata = load_model_checkpoint(model, checkpoint_path)
        val_metrics, _val_records = evaluate(model, val_loader, loss_fn, device, class_names)
        best["checkpoint"] = {
            "epoch": int(checkpoint_metadata.get("epoch", 0)),
            "metric_name": "accuracy",
            "metric_value": float(val_metrics["accuracy"]),
            "val_metrics": copy.deepcopy(val_metrics),
            "state_dict": clone_state(model),
        }
    else:
        optimizer = builder.build_optimizer(model)
        scheduler = builder.build_scheduler(optimizer)
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device, accumulation_steps)
            val_metrics, val_records = evaluate(model, val_loader, loss_fn, device, class_names)
            save_epoch_top_loss_records(run_dir, epoch, val_records)
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
                "val_max_sample_loss": val_metrics["max_sample_loss"],
                "val_max_abs_logit": val_metrics["max_abs_logit"],
                "val_p99_abs_logit": val_metrics["p99_abs_logit"],
                "seconds": seconds,
            }
            history.append(row)
            update_best(best, epoch, val_metrics, model)
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={row['train_loss']:.5f} train_acc={row['train_acc']:.4f} | "
                f"val_loss={row['val_loss']:.5f} val_acc={row['val_acc']:.4f} "
                f"val_f1={row['val_macro_f1']:.4f} val_qwk={row['val_qwk']:.4f} | "
                f"max_sample_loss={row['val_max_sample_loss']:.4f} "
                f"max_abs_logit={row['val_max_abs_logit']:.4f} "
                f"p99_abs_logit={row['val_p99_abs_logit']:.4f} | "
                f"time={seconds:.1f}s"
            )
        save_history(run_dir, history)

    best_summary = save_best_confusions(run_dir, model, val_loader, loss_fn, device, class_names, best)

    if state_key not in best:
        raise RuntimeError(f"找不到 state_key={state_key} 对应的模型状态。")
    model.load_state_dict(best[state_key]["state_dict"])
    diag_metrics, diag_records = evaluate(model, split_loaders[DIAGNOSTIC_SPLIT], loss_fn, device, class_names)
    parent_summary = save_parent_diagnostics(run_dir, diag_records, class_names, state_key, manifest_index)
    error_summary = save_high_confidence_errors(run_dir, diag_records, class_names, state_key, manifest_index)

    final_summary = {
        "model_name": model_name,
        "common_config_path": COMMON_CONFIG_PATH.as_posix(),
        "model_config_path": MODEL_CONFIG_PATH.as_posix(),
        "dataset_root": str(resolve_project_path(DATASET_ROOT)),
        "input_image_size": INPUT_IMAGE_SIZE,
        "class_names": class_names,
        "device": str(device),
        "epochs": 0 if checkpoint_path is not None else epochs,
        "checkpoint": checkpoint_metadata,
        "best_epoch_summary": best_summary,
        "diagnostic_split": DIAGNOSTIC_SPLIT,
        "prediction_state_key": state_key,
        "diagnostic_split_metrics": diag_metrics,
        "parent_summary": parent_summary,
        "high_confidence_error_summary": error_summary,
    }
    save_json(run_dir / "diagnostics_summary.json", final_summary)

    print("=" * 100)
    print("诊断完成")
    print(f"最佳epoch混淆矩阵：{run_dir / 'best_epoch_confusion_matrices'}")
    print(f"父图一致率/投票：{run_dir / f'parent_vote_{DIAGNOSTIC_SPLIT}_{state_key}'}")
    print(f"高置信错误图：{run_dir / f'high_confidence_errors_{DIAGNOSTIC_SPLIT}_{state_key}'}")
    print(f"总览JSON：{run_dir / 'diagnostics_summary.json'}")
    for key, record in best.items():
        print(f"{key}: epoch={record['epoch']}, {record['metric_name']}={record['metric_value']:.6f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
