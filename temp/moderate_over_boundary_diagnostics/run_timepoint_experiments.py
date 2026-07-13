"""Timepoint-based Moderate/Over diagnostics on 30-patch data.

This script is the new timepoint route:

* A: all Moderate vs Over timepoints.
* B: typical Moderate vs Over, excluding the 45/50 boundary.
* C: hard boundary, 45 vs 50 only.
* D: not a separate training job; every test run also aggregates the 30 patch
  predictions back to each original image and reports source-image metrics.

It reads datas_test_point_30_patches/patch_manifest.csv directly, so it does
not require creating temporary ImageFolder train/val/test directories.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import html
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (  # noqa: E402
    build_patch_eval_transform,
    build_patch_train_transform,
)
import src.models  # noqa: E402,F401 - registers models/backbones/heads
from src.utils import MODELS  # noqa: E402


TEMP_ROOT = Path(__file__).resolve().parent
DEFAULT_PATCH_ROOT = PROJECT_ROOT / "datas_test_point_30_patches"
DEFAULT_MANIFEST = DEFAULT_PATCH_ROOT / "patch_manifest.csv"
RESULTS_ROOT = TEMP_ROOT / "timepoint_results"

PYCHARM_DEVICE = "auto"
PYCHARM_DRY_RUN = False
PYCHARM_KEEP_PTH = False
PYCHARM_GROUP_KEY = "source_stem"

BINARY_CLASS_NAMES = ["moderate", "over"]
BINARY_CLASS_TO_IDX = {"moderate": 0, "over": 1}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_path: Path


@dataclass(frozen=True)
class DiagnosticSpec:
    name: str
    description: str
    moderate_times: tuple[str, ...]
    over_times: tuple[str, ...]

    @property
    def time_codes(self) -> tuple[str, ...]:
        return self.moderate_times + self.over_times


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
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    keep_pth: bool = False
    group_key: str = PYCHARM_GROUP_KEY


MODEL_CONFIG_LIST = (
    ModelSpec(
        "mambaout_tiny",
        PROJECT_ROOT / "configs/fixed_split_patches_models/mambaout_tiny.yaml",
    ),
    ModelSpec(
        "resnet50",
        PROJECT_ROOT / "configs/fixed_split_patches_models/resnet50.yaml",
    ),
    ModelSpec(
        "convnext_tiny",
        PROJECT_ROOT / "configs/fixed_split_patches_models/convnext_tiny.yaml",
    ),
    ModelSpec(
        "safnet_imagenet",
        PROJECT_ROOT / "configs/fixed_split_patches_models/safnet_imagenet.yaml",
    ),
)

DIAGNOSTIC_LIST = (
    DiagnosticSpec(
        name="a_full_moderate_over",
        description="All Moderate timepoints 30/35/40/45 vs Over 50/55/60.",
        moderate_times=("30", "35", "40", "45"),
        over_times=("50", "55", "60"),
    ),
    DiagnosticSpec(
        name="b_typical_moderate_over",
        description="Typical Moderate 30/35/40 vs typical Over 55/60.",
        moderate_times=("30", "35", "40"),
        over_times=("55", "60"),
    ),
    DiagnosticSpec(
        name="c_hard_45_vs_50",
        description="Hard adjacent boundary: 45 vs 50.",
        moderate_times=("45",),
        over_times=("50",),
    ),
)


class TimepointPatchDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        patch_root: Path,
        transform,
    ) -> None:
        self.rows = list(rows)
        self.patch_root = patch_root
        self.transform = transform
        self.targets = [int(row["label"]) for row in self.rows]
        self.classes = list(BINARY_CLASS_NAMES)
        self.class_to_idx = dict(BINARY_CLASS_TO_IDX)
        if not self.rows:
            raise ValueError("Dataset has no rows after filtering.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.patch_root / str(row["patch_relpath"])
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), row["source_image_id"], str(image_path)


def set_random_seed(seed: int) -> None:
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
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(name)


def safe_name(value: str) -> str:
    allowed = []
    for char in str(value):
        allowed.append(char if char.isalnum() or char in "._-" else "_")
    cleaned = "".join(allowed).strip("._-")
    return cleaned or "run"


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Patch manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Patch manifest is empty: {path}")
    required = {
        "patch_relpath",
        "source_image_id",
        "source_stem",
        "name_part_1",
        "name_part_2",
        "time_code",
        "severity",
        "patch_index",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Patch manifest is missing columns: {missing}")
    return rows


def add_binary_labels(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled_rows = []
    for row in rows:
        severity = str(row["severity"])
        if severity not in BINARY_CLASS_TO_IDX:
            continue
        copied = dict(row)
        copied["label"] = BINARY_CLASS_TO_IDX[severity]
        labeled_rows.append(copied)
    return labeled_rows


def filter_rows_for_diagnostic(
    manifest_rows: Sequence[dict[str, Any]],
    diagnostic: DiagnosticSpec,
    patch_root: Path,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in add_binary_labels(manifest_rows)
        if str(row["time_code"]) in diagnostic.time_codes
    ]
    expected_by_source: dict[str, int] = defaultdict(int)
    for row in selected:
        expected_by_source[str(row["source_image_id"])] += 1
    bad_sources = {
        source_id: count
        for source_id, count in expected_by_source.items()
        if count != 30
    }
    if bad_sources:
        preview = dict(list(sorted(bad_sources.items()))[:10])
        raise ValueError(
            f"{diagnostic.name} contains source images without 30 patches: {preview}"
        )
    missing_files = [
        row["patch_relpath"]
        for row in selected
        if not (patch_root / str(row["patch_relpath"])).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{diagnostic.name} has missing patch files: {missing_files[:10]}"
        )
    labels = Counter(int(row["label"]) for row in selected)
    if sorted(labels) != [0, 1]:
        raise ValueError(f"{diagnostic.name} must contain both classes, got {labels}.")
    return selected


def split_group_values(
    rows: Sequence[dict[str, Any]],
    settings: RuntimeSettings,
) -> dict[str, str]:
    group_values = sorted({str(row[settings.group_key]) for row in rows})
    if len(group_values) < 3:
        raise ValueError(
            f"Need at least 3 groups for train/val/test, got {len(group_values)}."
        )
    rng = random.Random(settings.seed)
    shuffled = list(group_values)
    rng.shuffle(shuffled)
    train_count = max(1, int(len(shuffled) * settings.train_ratio))
    val_count = max(1, int(len(shuffled) * settings.val_ratio))
    if train_count + val_count >= len(shuffled):
        val_count = 1
        train_count = len(shuffled) - 2
    split_by_group = {}
    for value in shuffled[:train_count]:
        split_by_group[value] = "train"
    for value in shuffled[train_count : train_count + val_count]:
        split_by_group[value] = "val"
    for value in shuffled[train_count + val_count :]:
        split_by_group[value] = "test"
    return split_by_group


def assign_splits(
    rows: Sequence[dict[str, Any]],
    settings: RuntimeSettings,
) -> dict[str, list[dict[str, Any]]]:
    split_by_group = split_group_values(rows, settings)
    split_rows = {"train": [], "val": [], "test": []}
    for row in rows:
        split_name = split_by_group[str(row[settings.group_key])]
        copied = dict(row)
        copied["diagnostic_split"] = split_name
        copied["diagnostic_group_key"] = settings.group_key
        copied["diagnostic_group_value"] = str(row[settings.group_key])
        split_rows[split_name].append(copied)

    for split_name, items in split_rows.items():
        labels = Counter(int(row["label"]) for row in items)
        if sorted(labels) != [0, 1]:
            raise ValueError(
                f"{split_name} split lacks one class with group_key="
                f"{settings.group_key}: {dict(labels)}"
            )
    return split_rows


def dataset_summary(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {}
    for split_name, rows in split_rows.items():
        source_ids = {str(row["source_image_id"]) for row in rows}
        group_values = {str(row["diagnostic_group_value"]) for row in rows}
        class_counts = Counter(str(row["severity"]) for row in rows)
        time_counts = Counter(str(row["time_code"]) for row in rows)
        summary[split_name] = {
            "patches": len(rows),
            "source_images": len(source_ids),
            "groups": len(group_values),
            "class_patch_counts": dict(sorted(class_counts.items())),
            "time_patch_counts": dict(sorted(time_counts.items())),
        }
    return summary


def build_loaders(
    split_rows: dict[str, list[dict[str, Any]]],
    patch_root: Path,
    settings: RuntimeSettings,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = build_patch_train_transform(settings.image_size)
    eval_transform = build_patch_eval_transform(settings.image_size)
    datasets = {
        "train": TimepointPatchDataset(split_rows["train"], patch_root, train_transform),
        "val": TimepointPatchDataset(split_rows["val"], patch_root, eval_transform),
        "test": TimepointPatchDataset(split_rows["test"], patch_root, eval_transform),
    }
    generator = torch.Generator().manual_seed(settings.seed)
    common = {
        "num_workers": settings.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": settings.num_workers > 0,
        "drop_last": False,
    }
    return (
        DataLoader(
            datasets["train"],
            batch_size=settings.batch_size,
            shuffle=True,
            generator=generator,
            **common,
        ),
        DataLoader(
            datasets["val"],
            batch_size=settings.val_batch_size,
            shuffle=False,
            **common,
        ),
        DataLoader(
            datasets["test"],
            batch_size=settings.test_batch_size,
            shuffle=False,
            **common,
        ),
    )


def load_model_document(model_spec: ModelSpec) -> dict[str, Any]:
    if not model_spec.config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_spec.config_path}")
    with model_spec.config_path.open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)
    if not isinstance(document, dict):
        raise ValueError(f"Model config must be a YAML mapping: {model_spec.config_path}")
    if document.get("name") != model_spec.name:
        raise ValueError(
            f"Model config name mismatch: list={model_spec.name!r}, "
            f"yaml={document.get('name')!r}"
        )
    model_config = document.get("model")
    if not isinstance(model_config, dict):
        raise ValueError(f"Model config lacks 'model': {model_spec.config_path}")
    return document


def prepare_model_document(
    model_spec: ModelSpec,
    pretrained_override: bool | None,
) -> dict[str, Any]:
    document = copy.deepcopy(load_model_document(model_spec))
    model_config = document["model"]
    backbone_config = model_config.get("backbone")
    head_config = model_config.get("head")
    if not isinstance(backbone_config, dict) or not isinstance(head_config, dict):
        raise ValueError(f"Invalid model config sections: {model_spec.config_path}")
    if pretrained_override is not None:
        backbone_config["pretrained"] = bool(pretrained_override)
    head_config["num_classes"] = 2
    if head_config.get("type") == "identity" or "num_classes" in backbone_config:
        backbone_config["num_classes"] = 2
    document["resolved_num_classes"] = 2
    return document


def build_model(model_document: dict[str, Any]) -> nn.Module:
    model_config = copy.deepcopy(model_document["model"])
    model_type = model_config.pop("type")
    model_config.pop("strategy", None)
    model_class = MODELS.get(model_type)
    return model_class(**model_config)


def primary_logits(outputs) -> torch.Tensor:
    return outputs[0] if isinstance(outputs, tuple) else outputs


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str] = BINARY_CLASS_NAMES,
) -> dict[str, Any]:
    labels_list = list(range(len(class_names)))
    matrix = confusion_matrix(labels, predictions, labels=labels_list)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=labels_list,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "class_names": list(class_names),
        "class_wise": {},
    }
    for index, class_name in enumerate(class_names):
        metrics["class_wise"][class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }
    if len(class_names) == 2:
        metrics["moderate_to_over_count"] = int(matrix[0, 1])
        metrics["over_to_moderate_count"] = int(matrix[1, 0])
    return metrics


def build_scheduler(optimizer: AdamW, settings: RuntimeSettings):
    cosine_epochs = max(1, settings.epochs - settings.warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=settings.min_learning_rate,
    )
    if settings.warmup_epochs <= 0:
        return cosine
    warmup = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=settings.warmup_epochs,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[settings.warmup_epochs],
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels_all = []
    predictions_all = []
    probabilities_all = []
    source_ids_all = []
    paths_all = []
    started = time.perf_counter()
    with torch.no_grad():
        for images, labels, source_ids, paths in tqdm(loader, desc=description, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = primary_logits(model(images))
            loss = F.cross_entropy(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            labels_all.append(labels.detach().cpu())
            predictions_all.append(predictions.detach().cpu())
            probabilities_all.append(probabilities.detach().cpu())
            source_ids_all.extend(str(item) for item in source_ids)
            paths_all.extend(str(item) for item in paths)
    labels_array = torch.cat(labels_all).numpy()
    predictions_array = torch.cat(predictions_all).numpy()
    probabilities_array = torch.cat(probabilities_all).numpy()
    return {
        "loss": total_loss / max(total_samples, 1),
        "samples": total_samples,
        "labels": labels_array,
        "predictions": predictions_array,
        "probabilities": probabilities_array,
        "source_ids": source_ids_all,
        "paths": paths_all,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": compute_metrics(labels_array, predictions_array),
    }


def aggregate_by_source(evaluation: dict[str, Any]) -> dict[str, Any]:
    grouped_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_labels: dict[str, set[int]] = defaultdict(set)
    grouped_patch_counts: Counter[str] = Counter()
    source_ids = evaluation["source_ids"]
    labels = evaluation["labels"]
    probabilities = evaluation["probabilities"]
    for source_id, label, probability in zip(source_ids, labels, probabilities):
        grouped_probabilities[str(source_id)].append(probability)
        grouped_labels[str(source_id)].add(int(label))
        grouped_patch_counts[str(source_id)] += 1

    rows = []
    source_labels = []
    source_predictions = []
    for source_id in sorted(grouped_probabilities):
        label_values = grouped_labels[source_id]
        if len(label_values) != 1:
            raise RuntimeError(f"Source image has mixed labels: {source_id}")
        mean_probability = np.stack(grouped_probabilities[source_id], axis=0).mean(axis=0)
        prediction = int(mean_probability.argmax())
        label = next(iter(label_values))
        source_labels.append(label)
        source_predictions.append(prediction)
        rows.append(
            {
                "source_image_id": source_id,
                "label": label,
                "label_name": BINARY_CLASS_NAMES[label],
                "pred": prediction,
                "pred_name": BINARY_CLASS_NAMES[prediction],
                "prob_moderate": float(mean_probability[0]),
                "prob_over": float(mean_probability[1]),
                "patch_count": int(grouped_patch_counts[source_id]),
                "correct": int(label == prediction),
            }
        )
    labels_array = np.asarray(source_labels, dtype=np.int64)
    predictions_array = np.asarray(source_predictions, dtype=np.int64)
    return {
        "rows": rows,
        "metrics": compute_metrics(labels_array, predictions_array),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    settings: RuntimeSettings,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float]:
    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = build_scheduler(optimizer, settings)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = -float("inf")
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, settings.epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for images, labels, _, _ in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{settings.epochs}",
            leave=False,
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = primary_logits(model(images))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.shape[0])
            train_loss += float(loss.item()) * batch_size
            train_samples += batch_size
        scheduler.step()
        validation = evaluate_model(model, val_loader, device, "Validation")
        val_acc = float(validation["metrics"]["accuracy"])
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss / max(train_samples, 1),
            "val_loss": float(validation["loss"]),
            "val_accuracy": val_acc,
            "val_macro_f1": float(validation["metrics"]["macro_f1"]),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_score:
            best_score = val_acc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings.patience:
                print(f"Early stopping at epoch {epoch}; best epoch is {best_epoch}.")
                break
    return best_state, history, best_epoch, time.perf_counter() - started


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, metrics: dict[str, Any]) -> None:
    class_names = metrics["class_names"]
    matrix = metrics["confusion_matrix"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])


def write_patch_predictions(path: Path, evaluation: dict[str, Any]) -> None:
    rows = []
    for path_text, source_id, label, pred, probability in zip(
        evaluation["paths"],
        evaluation["source_ids"],
        evaluation["labels"],
        evaluation["predictions"],
        evaluation["probabilities"],
    ):
        rows.append(
            {
                "patch_path": path_text,
                "source_image_id": source_id,
                "label": int(label),
                "label_name": BINARY_CLASS_NAMES[int(label)],
                "pred": int(pred),
                "pred_name": BINARY_CLASS_NAMES[int(pred)],
                "prob_moderate": float(probability[0]),
                "prob_over": float(probability[1]),
                "correct": int(int(label) == int(pred)),
            }
        )
    write_rows(path, rows)


def write_run_report(
    run_directory: Path,
    model_name: str,
    diagnostic: DiagnosticSpec,
    patch_metrics: dict[str, Any],
    source_metrics: dict[str, Any],
    dataset_info: dict[str, Any],
) -> Path:
    cards = [
        ("patch_accuracy", patch_metrics["accuracy"]),
        ("patch_macro_f1", patch_metrics["macro_f1"]),
        ("source_accuracy", source_metrics["accuracy"]),
        ("source_macro_f1", source_metrics["macro_f1"]),
    ]
    card_html = "".join(
        f"<div><span>{html.escape(name)}</span><strong>{float(value):.6f}</strong></div>"
        for name, value in cards
    )
    matrix_rows = "".join(
        "<tr><th>"
        + html.escape(class_name)
        + "</th>"
        + "".join(f"<td>{int(value)}</td>" for value in row)
        + "</tr>"
        for class_name, row in zip(
            source_metrics["class_names"],
            source_metrics["confusion_matrix"],
        )
    )
    split_rows = "".join(
        f"<tr><td>{html.escape(split)}</td><td>{info['patches']}</td>"
        f"<td>{info['source_images']}</td><td>{info['groups']}</td></tr>"
        for split, info in dataset_info.items()
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(model_name)} - {html.escape(diagnostic.name)}</title>
<style>
body{{max-width:980px;margin:36px auto;padding:0 18px;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#172033}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.cards div{{padding:14px;border:1px solid #dfe5ef;border-radius:8px}}
.cards span{{display:block;color:#667085}}.cards strong{{font-size:20px}}
table{{width:100%;border-collapse:collapse;margin-top:12px}}th,td{{border:1px solid #dfe5ef;padding:8px;text-align:center}}th{{background:#f4f6fa}}
</style></head><body>
<h1>{html.escape(model_name)}</h1>
<p>{html.escape(diagnostic.name)}: {html.escape(diagnostic.description)}</p>
<section class="cards">{card_html}</section>
<h2>Dataset split</h2><table><thead><tr><th>split</th><th>patches</th><th>source images</th><th>groups</th></tr></thead><tbody>{split_rows}</tbody></table>
<h2>Source-image confusion matrix</h2><table><thead><tr><th>true/pred</th><th>moderate</th><th>over</th></tr></thead><tbody>{matrix_rows}</tbody></table>
<p><a href="metrics.json">metrics.json</a> | <a href="source_predictions.csv">source_predictions.csv</a> | <a href="patch_predictions.csv">patch_predictions.csv</a></p>
</body></html>"""
    report_path = run_directory / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def run_one_job(
    model_spec: ModelSpec,
    diagnostic: DiagnosticSpec,
    manifest_rows: Sequence[dict[str, Any]],
    patch_root: Path,
    batch_directory: Path,
    device: torch.device,
    settings: RuntimeSettings,
) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"Start job: {diagnostic.name}/{model_spec.name}")
    print("=" * 80)
    set_random_seed(settings.seed)
    run_directory = batch_directory / diagnostic.name / model_spec.name
    run_directory.mkdir(parents=True, exist_ok=False)

    filtered_rows = filter_rows_for_diagnostic(manifest_rows, diagnostic, patch_root)
    split_rows = assign_splits(filtered_rows, settings)
    data_info = dataset_summary(split_rows)
    write_json(run_directory / "dataset_summary.json", data_info)
    for split_name, rows in split_rows.items():
        write_rows(run_directory / f"{split_name}_manifest.csv", rows)

    train_loader, val_loader, test_loader = build_loaders(split_rows, patch_root, settings)
    model_document = prepare_model_document(model_spec, pretrained_override=None)
    write_json(
        run_directory / "config.json",
        {
            "model_name": model_spec.name,
            "model_config_path": model_spec.config_path.relative_to(PROJECT_ROOT).as_posix(),
            "resolved_model_config": model_document,
            "diagnostic": asdict(diagnostic),
            "runtime": asdict(settings),
            "patch_root": str(patch_root),
        },
    )
    model = build_model(model_document).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    best_state, history, best_epoch, train_seconds = train_model(
        model,
        train_loader,
        val_loader,
        device,
        settings,
    )
    model.load_state_dict(best_state, strict=True)
    if settings.keep_pth:
        torch.save(
            {
                "model_state_dict": best_state,
                "model_name": model_spec.name,
                "diagnostic_name": diagnostic.name,
                "best_epoch": best_epoch,
            },
            run_directory / "best_model.pth",
        )
    testing = evaluate_model(model, test_loader, device, "Final testing")
    source_result = aggregate_by_source(testing)
    patch_metrics = testing["metrics"]
    source_metrics = source_result["metrics"]
    metrics = {
        "model_name": model_spec.name,
        "diagnostic_name": diagnostic.name,
        "patch_metrics": patch_metrics,
        "source_image_metrics": source_metrics,
        "best_epoch": best_epoch,
        "training_time_seconds": train_seconds,
        "test_inference_seconds": testing["elapsed_seconds"],
        "test_ms_per_patch": testing["elapsed_seconds"] * 1000.0 / max(testing["samples"], 1),
        "parameters_total": total_parameters,
        "parameters_trainable": trainable_parameters,
        "keep_pth": settings.keep_pth,
    }
    write_json(run_directory / "metrics.json", metrics)
    write_rows(run_directory / "history.csv", history)
    write_confusion_csv(run_directory / "patch_confusion_matrix.csv", patch_metrics)
    write_confusion_csv(run_directory / "source_confusion_matrix.csv", source_metrics)
    write_patch_predictions(run_directory / "patch_predictions.csv", testing)
    write_rows(run_directory / "source_predictions.csv", source_result["rows"])
    report_path = write_run_report(
        run_directory,
        model_spec.name,
        diagnostic,
        patch_metrics,
        source_metrics,
        data_info,
    )
    result = {
        "diagnostic_name": diagnostic.name,
        "model_name": model_spec.name,
        "status": "success",
        "patch_accuracy": patch_metrics["accuracy"],
        "patch_macro_f1": patch_metrics["macro_f1"],
        "source_accuracy": source_metrics["accuracy"],
        "source_macro_f1": source_metrics["macro_f1"],
        "best_epoch": best_epoch,
        "run_directory": str(run_directory),
        "report_path": str(report_path),
    }
    del model, best_state, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def dry_run(
    model_specs: Sequence[ModelSpec],
    diagnostics: Sequence[DiagnosticSpec],
    manifest_rows: Sequence[dict[str, Any]],
    patch_root: Path,
    device: torch.device,
    settings: RuntimeSettings,
) -> None:
    print(f"Patch root: {patch_root}")
    print(f"Manifest rows: {len(manifest_rows)}")
    for diagnostic in diagnostics:
        rows = filter_rows_for_diagnostic(manifest_rows, diagnostic, patch_root)
        split_rows = assign_splits(rows, settings)
        print(f"\n{diagnostic.name}")
        print(f"  {diagnostic.description}")
        for split_name, info in dataset_summary(split_rows).items():
            print(
                f"  {split_name}: patches={info['patches']}, "
                f"sources={info['source_images']}, groups={info['groups']}, "
                f"classes={info['class_patch_counts']}"
            )
    print("\nModel checks with pretrained=False:")
    set_random_seed(settings.seed)
    for model_spec in model_specs:
        model_document = prepare_model_document(model_spec, pretrained_override=False)
        model = build_model(model_document).to(device)
        model.eval()
        with torch.no_grad():
            images = torch.randn(2, 3, settings.image_size, settings.image_size, device=device)
            logits = primary_logits(model(images))
        if tuple(logits.shape) != (2, 2):
            raise RuntimeError(
                f"{model_spec.name} output shape should be (2, 2), got {tuple(logits.shape)}"
            )
        print(f"  {model_spec.name}: output={tuple(logits.shape)} PASS")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\ndry-run PASS")


def write_batch_report(batch_directory: Path, results: Sequence[dict[str, Any]]) -> Path:
    write_rows(batch_directory / "summary.csv", results)
    rows = []
    for result in results:
        report_path = result.get("report_path")
        report_link = ""
        if report_path:
            report_link = Path(str(report_path)).relative_to(batch_directory).as_posix()
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result['diagnostic_name']))}</td>"
            f"<td>{html.escape(str(result['model_name']))}</td>"
            f"<td>{html.escape(str(result['status']))}</td>"
            f"<td>{result.get('patch_accuracy', '')}</td>"
            f"<td>{result.get('source_accuracy', '')}</td>"
            f"<td>{result.get('source_macro_f1', '')}</td>"
            + (
                f'<td><a href="{html.escape(report_link, quote=True)}">report</a></td>'
                if report_link
                else "<td></td>"
            )
            + "</tr>"
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Timepoint Moderate/Over diagnostics</title>
<style>body{{max-width:1000px;margin:36px auto;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:9px;text-align:center}}th{{background:#f4f6fa}}</style>
</head><body><h1>Timepoint Moderate/Over diagnostics</h1>
<table><thead><tr><th>diagnostic</th><th>model</th><th>status</th><th>patch acc</th><th>source acc</th><th>source macro-F1</th><th>report</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    path = batch_directory / "summary.html"
    path.write_text(document, encoding="utf-8")
    return path


def select_models(names: Sequence[str] | None) -> list[ModelSpec]:
    by_name = {model_spec.name: model_spec for model_spec in MODEL_CONFIG_LIST}
    if not names:
        return list(MODEL_CONFIG_LIST)
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    return [by_name[name] for name in names]


def select_diagnostics(names: Sequence[str] | None) -> list[DiagnosticSpec]:
    by_name = {diagnostic.name: diagnostic for diagnostic in DIAGNOSTIC_LIST}
    if not names:
        return list(DIAGNOSTIC_LIST)
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown diagnostics: {unknown}")
    return [by_name[name] for name in names]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run timepoint Moderate/Over diagnostics from patch_manifest.csv."
    )
    parser.add_argument("--patch-root", type=Path, default=DEFAULT_PATCH_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--diagnostics", nargs="+")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=PYCHARM_DEVICE)
    parser.add_argument("--dry-run", action="store_true", default=PYCHARM_DRY_RUN)
    parser.add_argument("--keep-pth", action="store_true", default=PYCHARM_KEEP_PTH)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-diagnostics", action="store_true")
    parser.add_argument("--epochs", type=int, default=RuntimeSettings.epochs)
    parser.add_argument("--batch-size", type=int, default=RuntimeSettings.batch_size)
    parser.add_argument("--num-workers", type=int, default=RuntimeSettings.num_workers)
    parser.add_argument(
        "--group-key",
        choices=("source_stem", "name_part_1", "source_image_id"),
        default=PYCHARM_GROUP_KEY,
        help=(
            "Group used for train/val/test split. Default source_stem keeps the "
            "same 1-1 style series together across all timepoints."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    model_specs = select_models(args.models)
    diagnostics = select_diagnostics(args.diagnostics)
    if args.list_models:
        print("Models:")
        for model_spec in model_specs:
            print(f"  {model_spec.name}: {model_spec.config_path}")
    if args.list_diagnostics:
        print("Diagnostics:")
        for diagnostic in diagnostics:
            print(f"  {diagnostic.name}: {diagnostic.description}")
    if args.list_models or args.list_diagnostics:
        return

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    patch_root = args.patch_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if not patch_root.is_dir():
        raise FileNotFoundError(f"Patch root does not exist: {patch_root}")
    settings = replace(
        RuntimeSettings(),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        keep_pth=bool(args.keep_pth),
        num_workers=int(args.num_workers),
        group_key=str(args.group_key),
    )
    manifest_rows = read_manifest(manifest_path)
    device = resolve_device(args.device)
    if args.dry_run:
        dry_run(model_specs, diagnostics, manifest_rows, patch_root, device, settings)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_directory = RESULTS_ROOT / f"batch_{timestamp}"
    batch_directory.mkdir(parents=True, exist_ok=False)
    results = []
    for diagnostic in diagnostics:
        for model_spec in model_specs:
            try:
                result = run_one_job(
                    model_spec,
                    diagnostic,
                    manifest_rows,
                    patch_root,
                    batch_directory,
                    device,
                    settings,
                )
            except Exception as error:
                failure_directory = batch_directory / diagnostic.name / model_spec.name
                failure_directory.mkdir(parents=True, exist_ok=True)
                (failure_directory / "failure.txt").write_text(
                    f"{type(error).__name__}: {error}",
                    encoding="utf-8",
                )
                print(f"{diagnostic.name}/{model_spec.name} failed: {error}")
                result = {
                    "diagnostic_name": diagnostic.name,
                    "model_name": model_spec.name,
                    "status": "failed",
                    "patch_accuracy": None,
                    "patch_macro_f1": None,
                    "source_accuracy": None,
                    "source_macro_f1": None,
                    "best_epoch": None,
                    "run_directory": str(failure_directory),
                    "report_path": "",
                }
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results.append(result)
    summary_path = write_batch_report(batch_directory, results)
    print(f"\nAll timepoint diagnostics finished: {summary_path}")
    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
