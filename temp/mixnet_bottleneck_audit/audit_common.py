"""Shared data, model, metric, and post-hoc utilities for the MixNet-S audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF


TIME_CODES = ("00", "10", "20", "30", "40")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_PATCHES_PER_PARENT = 30


@dataclass(frozen=True)
class Sample:
    split: str
    time_code: str
    source_image_id: str
    patch_index: int
    image_path: Path
    target_relpath: str


@dataclass(frozen=True)
class JobSpec:
    name: str
    hypothesis: str
    description: str
    time_codes: tuple[str, ...]
    train_policy: str = "full"
    loss_type: str = "ce"
    label_smoothing: float = 0.0
    adjacent_alpha: float = 0.20
    ordinal_weight: float = 0.50


def seed_everything(seed: int) -> None:
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
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(name)


def safe_name(value: str) -> str:
    chars = [char if char.isalnum() or char in "._-" else "_" for char in str(value)]
    return "".join(chars).strip("._-") or "run"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_and_audit_manifest(
    manifest_path: Path,
    dataset_root: Path,
    *,
    check_files: bool = True,
) -> tuple[list[Sample], dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Grid manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"Grid manifest is empty: {manifest_path}")

    required = {"split", "time_code", "source_image_id", "patch_index", "target_relpath"}
    missing = sorted(required - set(raw_rows[0]))
    if missing:
        raise ValueError(f"Grid manifest is missing columns: {missing}")

    samples: list[Sample] = []
    seen_targets: set[str] = set()
    parent_rows: dict[str, list[Sample]] = defaultdict(list)
    missing_files: list[str] = []
    for raw in raw_rows:
        split = str(raw["split"]).strip()
        time_code = str(raw["time_code"]).strip().zfill(2)
        parent_id = str(raw["source_image_id"]).strip()
        relpath = str(raw["target_relpath"]).replace("\\", "/")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unexpected split {split!r} in manifest.")
        if time_code not in TIME_CODES:
            raise ValueError(f"Unexpected time code {time_code!r} in manifest.")
        if not parent_id:
            raise ValueError("Empty source_image_id in manifest.")
        if relpath in seen_targets:
            raise ValueError(f"Duplicate target_relpath in manifest: {relpath}")
        seen_targets.add(relpath)
        image_path = dataset_root / Path(relpath)
        if check_files and not image_path.is_file():
            missing_files.append(str(image_path))
        sample = Sample(
            split=split,
            time_code=time_code,
            source_image_id=parent_id,
            patch_index=int(raw["patch_index"]),
            image_path=image_path,
            target_relpath=relpath,
        )
        samples.append(sample)
        parent_rows[parent_id].append(sample)

    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(f"{len(missing_files)} manifest images are missing. First entries:\n{preview}")

    bad_parents: dict[str, Any] = {}
    parent_to_split: dict[str, str] = {}
    parent_to_class: dict[str, str] = {}
    for parent_id, rows in parent_rows.items():
        splits = {row.split for row in rows}
        classes = {row.time_code for row in rows}
        patch_indices = [row.patch_index for row in rows]
        problems: list[str] = []
        if len(rows) != EXPECTED_PATCHES_PER_PARENT:
            problems.append(f"patch_count={len(rows)}")
        if len(set(patch_indices)) != EXPECTED_PATCHES_PER_PARENT:
            problems.append("patch_index_not_unique_1_to_30")
        if set(patch_indices) != set(range(1, EXPECTED_PATCHES_PER_PARENT + 1)):
            problems.append("patch_index_not_exactly_1_to_30")
        if len(splits) != 1:
            problems.append(f"cross_split={sorted(splits)}")
        if len(classes) != 1:
            problems.append(f"cross_class={sorted(classes)}")
        if problems:
            bad_parents[parent_id] = problems
        else:
            parent_to_split[parent_id] = next(iter(splits))
            parent_to_class[parent_id] = next(iter(classes))
    if bad_parents:
        preview = dict(list(bad_parents.items())[:10])
        raise ValueError(f"Parent integrity audit failed for {len(bad_parents)} parents: {preview}")

    patch_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    parent_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        patch_counts[sample.split][sample.time_code] += 1
    for parent_id, split in parent_to_split.items():
        parent_counts[split][parent_to_class[parent_id]] += 1

    audit = {
        "manifest": str(manifest_path),
        "dataset_root": str(dataset_root),
        "time_codes": list(TIME_CODES),
        "patches_per_parent": EXPECTED_PATCHES_PER_PARENT,
        "total_patches": len(samples),
        "total_parents": len(parent_rows),
        "patch_counts": {split: dict(counts) for split, counts in patch_counts.items()},
        "parent_counts": {split: dict(counts) for split, counts in parent_counts.items()},
        "parent_integrity_passed": True,
        "cross_split_parent_count": 0,
        "missing_file_count": len(missing_files),
    }
    return samples, audit


class PatchDataset(Dataset):
    """Manifest-backed patches with labels remapped inside the selected class subset."""

    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        split: str,
        time_codes: Sequence[str],
        image_size: int,
        training: bool,
        color_variant: str = "rgb",
        max_parents_per_class: int | None = None,
    ) -> None:
        self.time_codes = tuple(time_codes)
        self.class_to_idx = {code: index for index, code in enumerate(self.time_codes)}
        self.classes = list(self.time_codes)
        self.samples = [
            sample
            for sample in samples
            if sample.split == split and sample.time_code in self.class_to_idx
        ]
        if max_parents_per_class is not None:
            allowed: set[str] = set()
            for time_code in self.time_codes:
                parent_ids = sorted(
                    {
                        sample.source_image_id
                        for sample in self.samples
                        if sample.time_code == time_code
                    }
                )
                allowed.update(parent_ids[:max_parents_per_class])
            self.samples = [sample for sample in self.samples if sample.source_image_id in allowed]
        if not self.samples:
            raise ValueError(f"No samples for split={split}, time_codes={self.time_codes}")
        self.targets = [self.class_to_idx[sample.time_code] for sample in self.samples]
        self.training = bool(training)
        self.color_variant = color_variant
        operations: list[Any] = [T.Resize((image_size, image_size), antialias=True)]
        if training:
            operations.extend([T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)])
        operations.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.tensor_transform = T.Compose(operations)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _stable_uniform(key: str, low: float, high: float, offset: int) -> float:
        digest = hashlib.sha256(f"{key}:{offset}".encode("utf-8")).digest()
        integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
        unit = integer / float(2**64 - 1)
        return low + (high - low) * unit

    def _apply_color_variant(self, image: Image.Image, sample: Sample) -> Image.Image:
        variant = self.color_variant
        if variant == "rgb":
            return image
        if variant == "grayscale":
            return TF.to_grayscale(image, num_output_channels=3)
        if variant == "color_jitter_stress":
            key = f"{sample.source_image_id}:{sample.patch_index}"
            image = TF.adjust_brightness(image, self._stable_uniform(key, 0.70, 1.30, 1))
            image = TF.adjust_contrast(image, self._stable_uniform(key, 0.70, 1.30, 2))
            image = TF.adjust_saturation(image, self._stable_uniform(key, 0.70, 1.30, 3))
            image = TF.adjust_hue(image, self._stable_uniform(key, -0.05, 0.05, 4))
            return image
        if variant == "clahe_luminance":
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("clahe_luminance requires opencv-python (cv2).") from exc
            rgb = np.asarray(image)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        raise ValueError(f"Unknown color variant: {variant}")

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as opened:
            image = opened.convert("RGB")
        image = self._apply_color_variant(image, sample)
        tensor = self.tensor_transform(image)
        return tensor, int(self.targets[index]), int(index)


def make_loader(
    dataset: PatchDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def create_mixnet(num_classes: int, *, pretrained: bool) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("The MixNet audit requires timm.") from exc
    return timm.create_model("mixnet_s", pretrained=pretrained, num_classes=num_classes)


def configure_train_policy(model: nn.Module, policy: str) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(policy == "full")

    if policy == "head_only":
        for parameter in model.get_classifier().parameters():
            parameter.requires_grad_(True)
    elif policy == "last_stage":
        modules = [model.blocks[-1], model.conv_head, model.bn2, model.get_classifier()]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif policy != "full":
        raise ValueError(f"Unknown train policy: {policy}")

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"parameters_total": int(total), "parameters_trainable": int(trainable)}


def set_train_mode(model: nn.Module, policy: str) -> None:
    if policy == "full":
        model.train()
        return
    # Keep every frozen stochastic layer (BN/dropout/drop-path) deterministic.
    model.eval()
    model.get_classifier().train()
    if policy == "last_stage":
        for module in (model.blocks[-1], model.conv_head, model.bn2):
            module.train()


def extract_prelogits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    features = model.forward_features(images)
    try:
        features = model.forward_head(features, pre_logits=True)
    except TypeError:
        features = model.forward_head(features)
    if isinstance(features, (tuple, list)):
        features = features[-1]
    if features.ndim > 2:
        features = torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
    return features


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    spec: JobSpec,
) -> torch.Tensor:
    if spec.loss_type == "ce":
        return F.cross_entropy(logits, labels, label_smoothing=float(spec.label_smoothing))
    if spec.loss_type == "adjacent_soft":
        num_classes = logits.shape[1]
        alpha = float(spec.adjacent_alpha)
        soft = torch.zeros_like(logits)
        soft.scatter_(1, labels[:, None], 1.0 - alpha)
        for row, label in enumerate(labels.tolist()):
            neighbors = [candidate for candidate in (label - 1, label + 1) if 0 <= candidate < num_classes]
            share = alpha / len(neighbors)
            for neighbor in neighbors:
                soft[row, neighbor] = share
        return -(soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    if spec.loss_type == "ce_ordinal_aux":
        ce = F.cross_entropy(logits, labels)
        probabilities = F.softmax(logits, dim=1)
        class_axis = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
        expected = (probabilities * class_axis[None, :]).sum(dim=1)
        scale = max(logits.shape[1] - 1, 1)
        ordinal = F.smooth_l1_loss(expected / scale, labels.to(logits.dtype) / scale)
        return ce + float(spec.ordinal_weight) * ordinal
    raise ValueError(f"Unknown loss type: {spec.loss_type}")


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not mask.any():
            continue
        accuracy = np.mean(predictions[mask] == labels[mask])
        ece += float(mask.mean()) * abs(float(accuracy) - float(confidences[mask].mean()))
    return float(ece)


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval; used at the independent-parent level."""
    if total <= 0:
        return float("nan"), float("nan")
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    num_classes = len(class_names)
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    errors = np.abs(labels - predictions)
    result: dict[str, Any] = {
        "sample_count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mae": float(errors.mean()),
        "qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
        "nll": float(-np.log(clipped).mean()),
        "ece_15bin": expected_calibration_error(probabilities, labels),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "confusion_matrix": matrix.tolist(),
        "class_names": list(class_names),
    }
    if num_classes > 2:
        wrong = predictions != labels
        wrong_count = int(wrong.sum())
        adjacent = int(((errors == 1) & wrong).sum())
        far = int((errors > 1).sum())
        result.update(
            {
                "error_count": wrong_count,
                "adjacent_error_count": adjacent,
                "far_error_count": far,
                "adjacent_error_fraction": float(adjacent / wrong_count) if wrong_count else 0.0,
                "far_error_fraction": float(far / wrong_count) if wrong_count else 0.0,
            }
        )
    for index, name in enumerate(class_names):
        denom = int(matrix[index].sum())
        result[f"recall_{name}"] = float(matrix[index, index] / denom) if denom else 0.0
    return result


def aggregate_parent_predictions(
    dataset: PatchDataset,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row_index, sample in enumerate(dataset.samples):
        grouped[sample.source_image_id].append(row_index)

    parent_rows: list[dict[str, Any]] = []
    parent_labels: list[int] = []
    mean_probabilities: list[np.ndarray] = []
    majority_predictions: list[int] = []
    for parent_id in sorted(grouped):
        indices = grouped[parent_id]
        true_values = labels[indices]
        if len(set(true_values.tolist())) != 1:
            raise ValueError(f"Parent {parent_id} has multiple remapped labels.")
        parent_label = int(true_values[0])
        patch_predictions = probabilities[indices].argmax(axis=1)
        counts = np.bincount(patch_predictions, minlength=len(dataset.classes))
        mean_probability = probabilities[indices].mean(axis=0)
        majority_candidates = np.flatnonzero(counts == counts.max())
        majority_prediction = int(
            majority_candidates[np.argmax(mean_probability[majority_candidates])]
        )
        mean_prediction = int(mean_probability.argmax())
        consistency = float(counts.max() / len(indices))
        normalized_entropy = 0.0
        proportions = counts / counts.sum()
        nonzero = proportions[proportions > 0]
        if len(dataset.classes) > 1:
            normalized_entropy = float(-(nonzero * np.log(nonzero)).sum() / math.log(len(dataset.classes)))
        row = {
            "source_image_id": parent_id,
            "time_code": dataset.samples[indices[0]].time_code,
            "label": parent_label,
            "mean_probability_prediction": mean_prediction,
            "majority_prediction": majority_prediction,
            "mean_probability_correct": int(mean_prediction == parent_label),
            "majority_correct": int(majority_prediction == parent_label),
            "patch_consistency": consistency,
            "patch_disagreement": 1.0 - consistency,
            "prediction_entropy_normalized": normalized_entropy,
            "patch_count": len(indices),
        }
        for class_index, class_name in enumerate(dataset.classes):
            row[f"vote_{class_name}"] = int(counts[class_index])
            row[f"mean_prob_{class_name}"] = float(mean_probability[class_index])
        parent_rows.append(row)
        parent_labels.append(parent_label)
        mean_probabilities.append(mean_probability)
        majority_predictions.append(majority_prediction)

    parent_label_array = np.asarray(parent_labels, dtype=np.int64)
    mean_probability_array = np.vstack(mean_probabilities)
    metrics = classification_metrics(parent_label_array, mean_probability_array, dataset.classes)
    metrics["aggregation"] = "mean_probability"
    mean_correct = int((mean_probability_array.argmax(axis=1) == parent_label_array).sum())
    majority_correct = int((np.asarray(majority_predictions) == parent_label_array).sum())
    mean_ci_low, mean_ci_high = wilson_interval(mean_correct, len(parent_label_array))
    majority_ci_low, majority_ci_high = wilson_interval(majority_correct, len(parent_label_array))
    metrics["accuracy_wilson95_low"] = mean_ci_low
    metrics["accuracy_wilson95_high"] = mean_ci_high
    metrics["majority_vote_accuracy"] = float(majority_correct / len(parent_label_array))
    metrics["majority_vote_accuracy_wilson95_low"] = majority_ci_low
    metrics["majority_vote_accuracy_wilson95_high"] = majority_ci_high
    metrics["mean_patch_consistency"] = float(
        np.mean([row["patch_consistency"] for row in parent_rows])
    )
    metrics["mean_patch_disagreement"] = float(
        np.mean([row["patch_disagreement"] for row in parent_rows])
    )
    metrics["mean_prediction_entropy_normalized"] = float(
        np.mean([row["prediction_entropy_normalized"] for row in parent_rows])
    )
    by_class: dict[str, dict[str, float]] = {}
    for class_name in dataset.classes:
        rows = [row for row in parent_rows if row["time_code"] == class_name]
        by_class[class_name] = {
            "parent_count": len(rows),
            "mean_patch_consistency": float(np.mean([row["patch_consistency"] for row in rows])),
            "mean_patch_disagreement": float(np.mean([row["patch_disagreement"] for row in rows])),
        }
    metrics["by_class"] = by_class
    return metrics, parent_rows


def prediction_rows(
    dataset: PatchDataset,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    predictions = probabilities.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset.samples):
        row: dict[str, Any] = {
            "split": sample.split,
            "time_code": sample.time_code,
            "source_image_id": sample.source_image_id,
            "patch_index": sample.patch_index,
            "target_relpath": sample.target_relpath,
            "label": int(labels[index]),
            "prediction": int(predictions[index]),
            "correct": int(labels[index] == predictions[index]),
            "confidence": float(probabilities[index].max()),
        }
        for class_index, class_name in enumerate(dataset.classes):
            row[f"prob_{class_name}"] = float(probabilities[index, class_index])
        rows.append(row)
    return rows


def job_to_dict(spec: JobSpec) -> dict[str, Any]:
    return asdict(spec)
