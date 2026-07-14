"""Run four-class soft-supervision diagnostics.

This script is intentionally independent from the main training entrypoints.
It compares:

* CE baseline
* CE retrain with equal teacher+student training budget
* label smoothing
* structural label smoothing
* bootstrap-soft
* fixed-teacher self-distillation

The default dataset is datasets_split_patches. Results are written under
temp/soft_supervision_diagnostics/results. Checkpoints are not kept by default.
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
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    pairwise_distances,
    precision_recall_fscore_support,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from scipy.sparse.csgraph import minimum_spanning_tree
except ImportError:  # pragma: no cover - scipy is expected with sklearn, fallback is below.
    minimum_spanning_tree = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (  # noqa: E402
    ImageFolderWithPaths,
    build_patch_eval_transform,
    build_patch_train_transform,
)
import src.models  # noqa: E402,F401 - registers models/backbones/heads
from src.utils import MODELS  # noqa: E402


TEMP_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets_split_patches"
RESULTS_ROOT = TEMP_ROOT / "results"

PYCHARM_DEVICE = "auto"
PYCHARM_DRY_RUN = False
PYCHARM_KEEP_PTH = False
DEFAULT_SEEDS = (2026,)

CLASS_TO_IDX = {"pre": 0, "slight": 1, "moderate": 2, "over": 3}
CLASS_NAMES = ["pre", "slight", "moderate", "over"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_path: Path


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    method: str
    description: str
    label_smoothing: float = 0.0
    bootstrap_beta: float = 0.8
    bootstrap_warmup_epochs: int = 30
    kd_temperature: float = 2.0
    kd_hard_weight: float = 0.7
    sls_average_alpha: float = 0.1
    sls_beta: float = 0.2
    sls_cluster_count: int = 32
    sls_pca_dim: int = 128
    sls_min_alpha: float = 0.0
    sls_max_alpha: float = 0.3
    sls_feature_model: str = "resnet50"


@dataclass(frozen=True)
class RuntimeSettings:
    seed: int = 2026
    epochs: int = 150
    batch_size: int = 32
    val_batch_size: int = 64
    test_batch_size: int = 64
    num_workers: int = 0
    patience: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 2
    min_learning_rate: float = 1e-6
    image_size: int = 224
    keep_pth: bool = False


DEFAULT_SETTINGS = RuntimeSettings()

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

EXPERIMENT_LIST = (
    ExperimentSpec(
        name="ce_baseline",
        method="ce",
        description="Plain four-class cross entropy.",
    ),
    ExperimentSpec(
        name="ce_retrain_same_cost",
        method="ce_retrain",
        description=(
            "Equal-budget control: train/use a CE teacher stage, then reinitialize "
            "a student and train CE without KD."
        ),
    ),
    ExperimentSpec(
        name="label_smoothing_eps0.1",
        method="label_smoothing",
        description="Cross entropy with uniform label smoothing epsilon=0.1.",
        label_smoothing=0.1,
    ),
    ExperimentSpec(
        name="label_smoothing_eps0.05",
        method="label_smoothing",
        description="Cross entropy with uniform label smoothing epsilon=0.05.",
        label_smoothing=0.05,
    ),
    ExperimentSpec(
        name="label_smoothing_eps0.2",
        method="label_smoothing",
        description="Cross entropy with uniform label smoothing epsilon=0.2.",
        label_smoothing=0.2,
    ),
    ExperimentSpec(
        name="sls_c16_a0.1_b0.2",
        method="structural_label_smoothing",
        description=(
            "Structural label smoothing: ResNet-50 features, PCA-128, "
            "KMeans C=16, average alpha=0.1, beta=0.2."
        ),
        sls_average_alpha=0.1,
        sls_beta=0.2,
        sls_cluster_count=16,
    ),
    ExperimentSpec(
        name="sls_c32_a0.1_b0.2",
        method="structural_label_smoothing",
        description=(
            "Structural label smoothing: ResNet-50 features, PCA-128, "
            "KMeans C=32, average alpha=0.1, beta=0.2."
        ),
        sls_average_alpha=0.1,
        sls_beta=0.2,
        sls_cluster_count=32,
    ),
    ExperimentSpec(
        name="sls_c64_a0.1_b0.2",
        method="structural_label_smoothing",
        description=(
            "Structural label smoothing: ResNet-50 features, PCA-128, "
            "KMeans C=64, average alpha=0.1, beta=0.2."
        ),
        sls_average_alpha=0.1,
        sls_beta=0.2,
        sls_cluster_count=64,
    ),
    ExperimentSpec(
        name="reverse_sls_c32_a0.1_b0.2",
        method="reverse_structural_label_smoothing",
        description=(
            "Reverse-SLS control: intentionally gives stronger smoothing to "
            "lower-overlap regions."
        ),
        sls_average_alpha=0.1,
        sls_beta=0.2,
        sls_cluster_count=32,
    ),
    ExperimentSpec(
        name="bootstrap_soft_beta0.8_warm30",
        method="bootstrap_soft",
        description="CE warmup, then beta*y + (1-beta)*stopgrad(p_model).",
        bootstrap_beta=0.8,
        bootstrap_warmup_epochs=30,
    ),
    ExperimentSpec(
        name="self_distill_t2_alpha0.7",
        method="self_distill",
        description="Same-architecture fixed CE teacher, T=2, hard weight=0.7.",
        kd_temperature=2.0,
        kd_hard_weight=0.7,
    ),
    ExperimentSpec(
        name="self_distill_t2_alpha0.5",
        method="self_distill",
        description="Optional alpha sweep: fixed T=2, hard weight=0.5.",
        kd_temperature=2.0,
        kd_hard_weight=0.5,
    ),
    ExperimentSpec(
        name="self_distill_t2_alpha0.9",
        method="self_distill",
        description="Optional alpha sweep: fixed T=2, hard weight=0.9.",
        kd_temperature=2.0,
        kd_hard_weight=0.9,
    ),
)

DEFAULT_EXPERIMENT_NAMES = (
    "ce_baseline",
    "label_smoothing_eps0.1",
    "sls_c16_a0.1_b0.2",
    "sls_c32_a0.1_b0.2",
    "sls_c64_a0.1_b0.2",
    "reverse_sls_c32_a0.1_b0.2",
)

TEACHER_CE_SPEC = ExperimentSpec(
    name="teacher_ce",
    method="ce",
    description="Internal CE teacher for self-distillation.",
)

SLS_METHODS = {
    "structural_label_smoothing",
    "reverse_structural_label_smoothing",
}


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


def primary_logits(outputs) -> torch.Tensor:
    return outputs[0] if isinstance(outputs, tuple) else outputs


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(to_builtin(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    head_config["num_classes"] = len(CLASS_NAMES)
    if head_config.get("type") == "identity" or "num_classes" in backbone_config:
        backbone_config["num_classes"] = len(CLASS_NAMES)
    document["resolved_num_classes"] = len(CLASS_NAMES)
    return document


def build_model(model_document: dict[str, Any]) -> nn.Module:
    model_config = copy.deepcopy(model_document["model"])
    model_type = model_config.pop("type")
    model_config.pop("strategy", None)
    model_class = MODELS.get(model_type)
    return model_class(**model_config)


def build_datasets(dataset_root: Path, settings: RuntimeSettings):
    train_transform = build_patch_train_transform(settings.image_size)
    eval_transform = build_patch_eval_transform(settings.image_size)
    return {
        "train": ImageFolderWithPaths(
            dataset_root / "train",
            transform=train_transform,
            class_to_idx=CLASS_TO_IDX,
        ),
        "val": ImageFolderWithPaths(
            dataset_root / "val",
            transform=eval_transform,
            class_to_idx=CLASS_TO_IDX,
        ),
        "test": ImageFolderWithPaths(
            dataset_root / "test",
            transform=eval_transform,
            class_to_idx=CLASS_TO_IDX,
        ),
    }


def build_loaders(datasets: dict[str, Dataset], settings: RuntimeSettings):
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


def dataset_counts(dataset) -> dict[str, int]:
    counts = Counter(int(target) for target in dataset.targets)
    return {
        class_name: int(counts.get(class_index, 0))
        for class_index, class_name in enumerate(dataset.classes)
    }


def dataset_summary(datasets: dict[str, Dataset]) -> dict[str, Any]:
    summary = {}
    for split_name, dataset in datasets.items():
        sample_image, sample_label, sample_path = dataset[0]
        summary[split_name] = {
            "total": len(dataset),
            "classes": dataset_counts(dataset),
            "sample_shape": list(sample_image.shape),
            "sample_label": int(sample_label),
            "sample_path": str(sample_path),
        }
    return summary


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probabilities).sum(dim=1).mean()


def sample_dependent_smoothing_targets(
    labels: torch.Tensor,
    alphas: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than 1.")
    if alphas.ndim != 1 or alphas.shape[0] != labels.shape[0]:
        raise ValueError(
            f"alphas must be shape [{labels.shape[0]}], got {tuple(alphas.shape)}"
        )
    alphas = alphas.to(dtype=torch.float32).clamp(0.0, 1.0)
    off_value = alphas / float(num_classes - 1)
    targets = off_value.unsqueeze(1).repeat(1, num_classes)
    targets.scatter_(1, labels.unsqueeze(1), (1.0 - alphas).unsqueeze(1))
    return targets


def mst_cross_label_rate(features: np.ndarray, labels: np.ndarray) -> float:
    sample_count = int(labels.shape[0])
    if sample_count <= 1 or len(set(int(label) for label in labels)) <= 1:
        return 0.0
    distances = pairwise_distances(features, metric="euclidean")
    np.fill_diagonal(distances, 0.0)
    if minimum_spanning_tree is not None:
        tree = minimum_spanning_tree(distances).tocoo()
        if tree.row.size == 0:
            return 0.0
        cross_edges = labels[tree.row] != labels[tree.col]
        return float(np.mean(cross_edges))

    # Conservative fallback if scipy is unavailable: use each sample's nearest
    # non-self neighbor as a local crossing proxy.
    distances[distances == 0.0] = np.inf
    nearest = np.argmin(distances, axis=1)
    return float(np.mean(labels != labels[nearest]))


def rebalance_alphas_to_weighted_mean(
    alpha_values: np.ndarray,
    weights: np.ndarray,
    target_mean: float,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    values = np.clip(alpha_values.astype(np.float64), minimum, maximum)
    weights = weights.astype(np.float64)
    if weights.sum() <= 0:
        return values
    weights = weights / weights.sum()
    for _ in range(32):
        current_mean = float(np.sum(values * weights))
        delta = float(target_mean - current_mean)
        if abs(delta) < 1e-8:
            break
        if delta > 0:
            movable = values < maximum - 1e-12
        else:
            movable = values > minimum + 1e-12
        movable_weight = float(np.sum(weights[movable]))
        if movable_weight <= 0:
            break
        values[movable] = np.clip(
            values[movable] + delta / movable_weight,
            minimum,
            maximum,
        )
    return values


def compute_sls_cluster_alphas(
    overlap_scores: np.ndarray,
    cluster_sizes: np.ndarray,
    experiment: ExperimentSpec,
) -> np.ndarray:
    average_alpha = float(experiment.sls_average_alpha)
    beta = float(experiment.sls_beta)
    minimum = float(experiment.sls_min_alpha)
    maximum = float(experiment.sls_max_alpha)
    if not 0.0 <= average_alpha <= 1.0:
        raise ValueError("sls_average_alpha must be in [0, 1].")
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("SLS alpha bounds must satisfy 0 <= min <= max <= 1.")
    if not minimum <= average_alpha <= maximum:
        raise ValueError("sls_average_alpha must fall inside [sls_min_alpha, sls_max_alpha].")
    weights = cluster_sizes.astype(np.float64)
    weights = weights / max(float(weights.sum()), 1.0)
    weighted_score_mean = float(np.sum(overlap_scores * weights))
    direction = -1.0 if experiment.method == "reverse_structural_label_smoothing" else 1.0
    raw_alphas = average_alpha + direction * beta * (overlap_scores - weighted_score_mean)
    return rebalance_alphas_to_weighted_mean(
        raw_alphas,
        weights,
        target_mean=average_alpha,
        minimum=minimum,
        maximum=maximum,
    )


def build_sls_feature_extractor(
    feature_model_name: str,
    device: torch.device,
) -> nn.Module:
    model_specs = {model_spec.name: model_spec for model_spec in MODEL_CONFIG_LIST}
    if feature_model_name not in model_specs:
        raise ValueError(
            f"Unknown SLS feature model {feature_model_name!r}; "
            f"available={sorted(model_specs)}"
        )
    model_document = prepare_model_document(
        model_specs[feature_model_name],
        pretrained_override=None,
    )
    model_document["model"]["return_embeddings"] = True
    model = build_model(model_document).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_sls_features(
    train_dataset: Dataset,
    device: torch.device,
    settings: RuntimeSettings,
    feature_model_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_dataset = ImageFolderWithPaths(
        train_dataset.root,
        transform=build_patch_eval_transform(settings.image_size),
        class_to_idx=CLASS_TO_IDX,
    )
    feature_loader = DataLoader(
        feature_dataset,
        batch_size=settings.test_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=settings.num_workers > 0,
        drop_last=False,
    )
    feature_model = build_sls_feature_extractor(feature_model_name, device)
    features_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    paths_all: list[str] = []
    with torch.no_grad():
        for images, labels, paths in tqdm(
            feature_loader,
            desc=f"SLS feature extraction ({feature_model_name})",
            leave=False,
        ):
            images = images.to(device, non_blocking=True)
            outputs = feature_model(images)
            if isinstance(outputs, tuple):
                features = outputs[1]
            else:
                features = primary_logits(outputs)
            features = F.normalize(features.float(), p=2, dim=1)
            features_all.append(features.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())
            paths_all.extend(str(path) for path in paths)
    del feature_model, feature_loader, feature_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        np.concatenate(features_all, axis=0),
        np.concatenate(labels_all, axis=0).astype(np.int64),
        paths_all,
    )


def build_structural_label_smoothing_plan(
    train_dataset: Dataset,
    device: torch.device,
    settings: RuntimeSettings,
    experiment: ExperimentSpec,
) -> dict[str, Any]:
    started = time.perf_counter()
    features, labels, paths = extract_sls_features(
        train_dataset=train_dataset,
        device=device,
        settings=settings,
        feature_model_name=experiment.sls_feature_model,
    )
    if features.shape[0] != labels.shape[0] or features.shape[0] != len(paths):
        raise RuntimeError("SLS feature extraction returned inconsistent lengths.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    pca_dim = int(min(experiment.sls_pca_dim, scaled_features.shape[0] - 1, scaled_features.shape[1]))
    if pca_dim <= 0:
        raise ValueError("SLS PCA dimension must be positive after clipping.")
    pca = PCA(n_components=pca_dim, random_state=settings.seed)
    reduced_features = pca.fit_transform(scaled_features)

    cluster_count = int(min(experiment.sls_cluster_count, reduced_features.shape[0]))
    if cluster_count <= 1:
        raise ValueError("SLS requires at least two clusters.")
    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=settings.seed,
        n_init=10,
    )
    cluster_ids = kmeans.fit_predict(reduced_features)

    cluster_sizes = np.zeros(cluster_count, dtype=np.int64)
    overlap_scores = np.zeros(cluster_count, dtype=np.float64)
    class_count_matrix = np.zeros((cluster_count, len(CLASS_NAMES)), dtype=np.int64)
    for cluster_index in range(cluster_count):
        mask = cluster_ids == cluster_index
        cluster_labels = labels[mask]
        cluster_features = reduced_features[mask]
        cluster_sizes[cluster_index] = int(cluster_labels.shape[0])
        class_count_matrix[cluster_index] = np.bincount(
            cluster_labels,
            minlength=len(CLASS_NAMES),
        )
        overlap_scores[cluster_index] = mst_cross_label_rate(
            cluster_features,
            cluster_labels,
        )

    cluster_alphas = compute_sls_cluster_alphas(
        overlap_scores=overlap_scores,
        cluster_sizes=cluster_sizes,
        experiment=experiment,
    )
    sample_alphas = cluster_alphas[cluster_ids]
    sample_alpha_by_path = {
        str(path): float(alpha)
        for path, alpha in zip(paths, sample_alphas)
    }

    cluster_rows: list[dict[str, Any]] = []
    for cluster_index in range(cluster_count):
        counts = class_count_matrix[cluster_index]
        row: dict[str, Any] = {
            "cluster_id": int(cluster_index),
            "sample_count": int(cluster_sizes[cluster_index]),
            "overlap_score_mst_cross_edge_rate": float(overlap_scores[cluster_index]),
            "alpha": float(cluster_alphas[cluster_index]),
            "unique_classes": int(np.count_nonzero(counts)),
        }
        for class_index, class_name in enumerate(CLASS_NAMES):
            row[f"count_{class_name}"] = int(counts[class_index])
            row[f"ratio_{class_name}"] = (
                float(counts[class_index] / max(cluster_sizes[cluster_index], 1))
            )
        cluster_rows.append(row)

    sample_rows = []
    for path, label, cluster_id, alpha in zip(paths, labels, cluster_ids, sample_alphas):
        sample_rows.append(
            {
                "image_path": str(path),
                "label": int(label),
                "label_name": CLASS_NAMES[int(label)],
                "cluster_id": int(cluster_id),
                "cluster_overlap_score": float(overlap_scores[int(cluster_id)]),
                "alpha": float(alpha),
            }
        )

    weights = cluster_sizes.astype(np.float64) / max(float(cluster_sizes.sum()), 1.0)
    summary = {
        "method": experiment.method,
        "feature_model": experiment.sls_feature_model,
        "requested_clusters": int(experiment.sls_cluster_count),
        "actual_clusters": int(cluster_count),
        "requested_pca_dim": int(experiment.sls_pca_dim),
        "actual_pca_dim": int(pca_dim),
        "average_alpha_target": float(experiment.sls_average_alpha),
        "average_alpha_actual": float(np.mean(sample_alphas)),
        "weighted_cluster_alpha": float(np.sum(cluster_alphas * weights)),
        "alpha_min": float(np.min(sample_alphas)),
        "alpha_max": float(np.max(sample_alphas)),
        "alpha_std": float(np.std(sample_alphas)),
        "overlap_score_min": float(np.min(overlap_scores)),
        "overlap_score_max": float(np.max(overlap_scores)),
        "overlap_score_weighted_mean": float(np.sum(overlap_scores * weights)),
        "single_class_clusters": int(
            sum(1 for row in cluster_rows if int(row["unique_classes"]) == 1)
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return {
        "sample_alpha_by_path": sample_alpha_by_path,
        "cluster_rows": cluster_rows,
        "sample_rows": sample_rows,
        "summary": summary,
    }


def self_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    hard_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0.0 <= hard_weight <= 1.0:
        raise ValueError("hard_weight must be in [0, 1].")
    hard_loss = F.cross_entropy(student_logits, targets)
    teacher_prob_t = F.softmax(teacher_logits / temperature, dim=1)
    student_log_prob_t = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = (
        F.kl_div(student_log_prob_t, teacher_prob_t, reduction="batchmean")
        * (temperature**2)
    )
    total_loss = hard_weight * hard_loss + (1.0 - hard_weight) * kd_loss
    return total_loss, {
        "hard_ce": float(hard_loss.detach()),
        "kd": float(kd_loss.detach()),
    }


def compute_training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    experiment: ExperimentSpec,
    epoch: int,
    teacher: nn.Module | None = None,
    sample_alphas: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    logits = primary_logits(model(images))
    components: dict[str, float] = {}
    if experiment.method in ("ce", "ce_retrain"):
        loss = F.cross_entropy(logits, labels)
        components["hard_ce"] = float(loss.detach())
        return loss, components, logits
    if experiment.method == "label_smoothing":
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=float(experiment.label_smoothing),
        )
        components["label_smoothing_ce"] = float(loss.detach())
        return loss, components, logits
    if experiment.method in SLS_METHODS:
        if sample_alphas is None:
            raise ValueError(f"{experiment.method} requires sample_alphas.")
        soft_target = sample_dependent_smoothing_targets(
            labels=labels,
            alphas=sample_alphas.to(device=labels.device),
            num_classes=logits.shape[1],
        ).to(dtype=logits.dtype)
        hard_loss = F.cross_entropy(logits, labels)
        loss = soft_cross_entropy(logits, soft_target)
        components["hard_ce"] = float(hard_loss.detach())
        components["sls_soft_ce"] = float(loss.detach())
        components["sls_alpha_mean"] = float(sample_alphas.detach().mean().item())
        components["sls_alpha_min"] = float(sample_alphas.detach().min().item())
        components["sls_alpha_max"] = float(sample_alphas.detach().max().item())
        return loss, components, logits
    if experiment.method == "bootstrap_soft":
        hard_loss = F.cross_entropy(logits, labels)
        if epoch <= int(experiment.bootstrap_warmup_epochs):
            components["hard_ce"] = float(hard_loss.detach())
            components["bootstrap_active"] = 0.0
            return hard_loss, components, logits
        one_hot = F.one_hot(labels, num_classes=logits.shape[1]).to(logits.dtype)
        model_prob = F.softmax(logits.detach(), dim=1)
        beta = float(experiment.bootstrap_beta)
        soft_target = beta * one_hot + (1.0 - beta) * model_prob
        loss = soft_cross_entropy(logits, soft_target)
        components["hard_ce"] = float(hard_loss.detach())
        components["bootstrap_soft_ce"] = float(loss.detach())
        components["bootstrap_active"] = 1.0
        return loss, components, logits
    if experiment.method == "self_distill":
        if teacher is None:
            raise ValueError("self_distill requires a fixed teacher model.")
        teacher.eval()
        with torch.no_grad():
            teacher_logits = primary_logits(teacher(images))
        loss, kd_components = self_distillation_loss(
            student_logits=logits,
            teacher_logits=teacher_logits,
            targets=labels,
            temperature=float(experiment.kd_temperature),
            hard_weight=float(experiment.kd_hard_weight),
        )
        components.update(kd_components)
        return loss, components, logits
    raise ValueError(f"Unknown experiment method: {experiment.method}")


def compute_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels_list = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(labels, predictions, labels=labels_list)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=labels_list,
        zero_division=0,
    )
    qwk = cohen_kappa_score(
        labels,
        predictions,
        labels=labels_list,
        weights="quadratic",
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mae": float(np.mean(np.abs(labels - predictions))),
        "qwk": float(qwk) if np.isfinite(qwk) else 0.0,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(CLASS_NAMES),
        "class_wise": {},
    }
    for index, class_name in enumerate(CLASS_NAMES):
        metrics["class_wise"][class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }
    for left, right in ((0, 1), (1, 2), (2, 3)):
        mask = (labels == left) | (labels == right)
        metrics[f"acc_{left}_{right}"] = (
            float(np.mean(predictions[mask] == labels[mask]))
            if np.any(mask)
            else None
        )
    for source, target in ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)):
        metrics[f"error_{source}_to_{target}"] = int(matrix[source, target])
    metrics["moderate_to_over_count"] = int(matrix[2, 3])
    metrics["over_to_moderate_count"] = int(matrix[3, 2])
    metrics["far_error_count"] = int(np.sum(np.abs(labels - predictions) >= 2))
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
    paths_all = []
    started = time.perf_counter()
    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc=description, leave=False):
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
            paths_all.extend(str(path) for path in paths)
    labels_array = torch.cat(labels_all).numpy()
    predictions_array = torch.cat(predictions_all).numpy()
    probabilities_array = torch.cat(probabilities_all).numpy()
    return {
        "loss": total_loss / max(total_samples, 1),
        "samples": total_samples,
        "labels": labels_array,
        "predictions": predictions_array,
        "probabilities": probabilities_array,
        "paths": paths_all,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": compute_metrics(labels_array, predictions_array),
    }


def summarize_values(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def probability_diagnostics(evaluation: dict[str, Any]) -> dict[str, Any]:
    probabilities = evaluation["probabilities"]
    labels = evaluation["labels"]
    predictions = evaluation["predictions"]
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    confidence = np.max(probabilities, axis=1)
    delta_mo = np.abs(probabilities[:, 2] - probabilities[:, 3])
    correct_mask = labels == predictions
    result: dict[str, Any] = {
        "overall": {
            "entropy": summarize_values(entropy),
            "max_confidence": summarize_values(confidence),
            "delta_moderate_over": summarize_values(delta_mo),
        },
        "correct_vs_wrong": {
            "correct_entropy": summarize_values(entropy[correct_mask]),
            "wrong_entropy": summarize_values(entropy[~correct_mask]),
            "correct_confidence": summarize_values(confidence[correct_mask]),
            "wrong_confidence": summarize_values(confidence[~correct_mask]),
        },
        "by_true_class": {},
    }
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_index
        result["by_true_class"][class_name] = {
            "entropy": summarize_values(entropy[mask]),
            "max_confidence": summarize_values(confidence[mask]),
            "delta_moderate_over": summarize_values(delta_mo[mask]),
        }
    mo_mask = (labels == 2) | (labels == 3)
    result["moderate_over_only"] = {
        "entropy": summarize_values(entropy[mo_mask]),
        "max_confidence": summarize_values(confidence[mo_mask]),
        "delta_moderate_over": summarize_values(delta_mo[mo_mask]),
    }
    return result


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    settings: RuntimeSettings,
    experiment: ExperimentSpec,
    teacher: nn.Module | None = None,
    sample_alpha_by_path: dict[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float]:
    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = build_scheduler(optimizer, settings)
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_epoch = 0
    best_score = (-float("inf"), -float("inf"))
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    if teacher is not None:
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    for epoch in range(1, settings.epochs + 1):
        model.train()
        train_samples = 0
        component_sums: dict[str, float] = defaultdict(float)
        for images, labels, paths in tqdm(
            train_loader,
            desc=f"{experiment.name} epoch {epoch}/{settings.epochs}",
            leave=False,
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sample_alphas = None
            if sample_alpha_by_path is not None:
                missing_paths = [
                    str(path)
                    for path in paths
                    if str(path) not in sample_alpha_by_path
                ]
                if missing_paths:
                    raise KeyError(
                        "SLS alpha missing for training paths, first missing: "
                        f"{missing_paths[0]}"
                    )
                sample_alphas = torch.tensor(
                    [sample_alpha_by_path[str(path)] for path in paths],
                    dtype=torch.float32,
                    device=device,
                )
            optimizer.zero_grad(set_to_none=True)
            loss, components, _ = compute_training_loss(
                model=model,
                images=images,
                labels=labels,
                experiment=experiment,
                epoch=epoch,
                teacher=teacher,
                sample_alphas=sample_alphas,
            )
            loss.backward()
            optimizer.step()
            batch_size = int(labels.shape[0])
            train_samples += batch_size
            component_sums["train_loss"] += float(loss.item()) * batch_size
            for key, value in components.items():
                component_sums[key] += float(value) * batch_size
        scheduler.step()
        validation = evaluate_model(model, val_loader, device, "Validation")
        metrics = validation["metrics"]
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": component_sums["train_loss"] / max(train_samples, 1),
            "val_loss": float(validation["loss"]),
            "val_accuracy": float(metrics["accuracy"]),
            "val_macro_f1": float(metrics["macro_f1"]),
            "val_acc_2_3": metrics.get("acc_2_3"),
        }
        for key, value in sorted(component_sums.items()):
            if key == "train_loss":
                continue
            row[f"train_{key}"] = value / max(train_samples, 1)
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_acc={row['val_accuracy']:.4f} "
            f"val_acc_2_3={row['val_acc_2_3']}"
        )
        score = (float(metrics["accuracy"]), float(metrics["macro_f1"]))
        if score > best_score:
            best_score = score
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


def write_confusion_csv(path: Path, metrics: dict[str, Any]) -> None:
    matrix = metrics["confusion_matrix"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row])


def write_predictions(path: Path, evaluation: dict[str, Any]) -> None:
    rows = []
    for path_text, label, prediction, probability in zip(
        evaluation["paths"],
        evaluation["labels"],
        evaluation["predictions"],
        evaluation["probabilities"],
    ):
        rows.append(
            {
                "image_path": path_text,
                "label": int(label),
                "label_name": CLASS_NAMES[int(label)],
                "pred": int(prediction),
                "pred_name": CLASS_NAMES[int(prediction)],
                "prob_pre": float(probability[0]),
                "prob_slight": float(probability[1]),
                "prob_moderate": float(probability[2]),
                "prob_over": float(probability[3]),
                "delta_moderate_over": float(abs(probability[2] - probability[3])),
                "max_confidence": float(np.max(probability)),
                "correct": int(int(label) == int(prediction)),
            }
        )
    write_rows(path, rows)


def write_run_report(
    run_directory: Path,
    model_name: str,
    experiment: ExperimentSpec,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> Path:
    cards = [
        ("accuracy", metrics["accuracy"]),
        ("macro_f1", metrics["macro_f1"]),
        ("qwk", metrics["qwk"]),
        ("acc_2_3", metrics.get("acc_2_3")),
        ("M_to_O", metrics["moderate_to_over_count"]),
        ("O_to_M", metrics["over_to_moderate_count"]),
    ]
    card_html = "".join(
        f"<div><span>{html.escape(name)}</span><strong>{value}</strong></div>"
        for name, value in cards
    )
    matrix_rows = "".join(
        "<tr><th>"
        + html.escape(class_name)
        + "</th>"
        + "".join(f"<td>{int(value)}</td>" for value in row)
        + "</tr>"
        for class_name, row in zip(CLASS_NAMES, metrics["confusion_matrix"])
    )
    best_epoch = history[-1]["epoch"] if history else ""
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(model_name)} - {html.escape(experiment.name)}</title>
<style>
body{{max-width:1000px;margin:36px auto;padding:0 18px;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#172033}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.cards div{{padding:14px;border:1px solid #dfe5ef;border-radius:8px}}
.cards span{{display:block;color:#667085}}.cards strong{{font-size:20px}}
table{{width:100%;border-collapse:collapse;margin-top:12px}}th,td{{border:1px solid #dfe5ef;padding:8px;text-align:center}}th{{background:#f4f6fa}}
</style></head><body>
<h1>{html.escape(model_name)}</h1>
<p>{html.escape(experiment.name)}: {html.escape(experiment.description)}</p>
<section class="cards">{card_html}</section>
<h2>Confusion matrix</h2>
<table><thead><tr><th>true/pred</th><th>pre</th><th>slight</th><th>moderate</th><th>over</th></tr></thead><tbody>{matrix_rows}</tbody></table>
<p>Last logged epoch: {best_epoch}</p>
<p><a href="metrics.json">metrics.json</a> | <a href="history.csv">history.csv</a> | <a href="predictions.csv">predictions.csv</a></p>
</body></html>"""
    path = run_directory / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


def save_run_outputs(
    run_directory: Path,
    model_spec: ModelSpec,
    experiment: ExperimentSpec,
    model_document: dict[str, Any],
    settings: RuntimeSettings,
    dataset_info: dict[str, Any],
    history: list[dict[str, Any]],
    best_epoch: int,
    training_seconds: float,
    testing: dict[str, Any],
    parameters_total: int,
    parameters_trainable: int,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    metrics = copy.deepcopy(testing["metrics"])
    metrics.update(
        {
            "model_name": model_spec.name,
            "experiment_name": experiment.name,
            "method": experiment.method,
            "best_epoch": best_epoch,
            "training_time_seconds": training_seconds,
            "test_inference_seconds": testing["elapsed_seconds"],
            "test_ms_per_sample": testing["elapsed_seconds"] * 1000.0 / max(testing["samples"], 1),
            "parameters_total": parameters_total,
            "parameters_trainable": parameters_trainable,
            "keep_pth": settings.keep_pth,
        }
    )
    if extra:
        metrics.update(extra)
    write_json(
        run_directory / "config.json",
        {
            "model_name": model_spec.name,
            "model_config_path": model_spec.config_path.relative_to(PROJECT_ROOT).as_posix(),
            "resolved_model_config": model_document,
            "experiment": asdict(experiment),
            "runtime": asdict(settings),
        },
    )
    write_json(run_directory / "dataset_summary.json", dataset_info)
    write_json(run_directory / "metrics.json", metrics)
    write_json(
        run_directory / "probability_diagnostics.json",
        probability_diagnostics(testing),
    )
    write_rows(run_directory / "history.csv", history)
    write_confusion_csv(run_directory / "confusion_matrix.csv", metrics)
    write_predictions(run_directory / "predictions.csv", testing)
    report_path = write_run_report(run_directory, model_spec.name, experiment, metrics, history)
    return metrics, report_path


def model_parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return int(total), int(trainable)


def run_standard_experiment(
    model_spec: ModelSpec,
    experiment: ExperimentSpec,
    datasets: dict[str, Dataset],
    dataset_info: dict[str, Any],
    batch_directory: Path,
    device: torch.device,
    settings: RuntimeSettings,
    sls_plan_cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    print("\n" + "=" * 80)
    print(f"Start job: {model_spec.name}/{experiment.name}")
    print("=" * 80)
    set_random_seed(settings.seed)
    run_directory = batch_directory / model_spec.name / experiment.name
    run_directory.mkdir(parents=True, exist_ok=False)
    sls_plan = None
    sample_alpha_by_path = None
    extra: dict[str, Any] = {}
    if experiment.method in SLS_METHODS:
        cache_key = (
            f"seed={settings.seed}|method={experiment.method}|"
            f"feature={experiment.sls_feature_model}|"
            f"clusters={experiment.sls_cluster_count}|"
            f"pca={experiment.sls_pca_dim}|"
            f"avg={experiment.sls_average_alpha}|"
            f"beta={experiment.sls_beta}|"
            f"min={experiment.sls_min_alpha}|"
            f"max={experiment.sls_max_alpha}"
        )
        if sls_plan_cache is not None and cache_key in sls_plan_cache:
            sls_plan = sls_plan_cache[cache_key]
        else:
            sls_plan = build_structural_label_smoothing_plan(
                train_dataset=datasets["train"],
                device=device,
                settings=settings,
                experiment=experiment,
            )
            if sls_plan_cache is not None:
                sls_plan_cache[cache_key] = sls_plan
        sample_alpha_by_path = sls_plan["sample_alpha_by_path"]
        write_json(run_directory / "sls_summary.json", sls_plan["summary"])
        write_rows(run_directory / "sls_cluster_summary.csv", sls_plan["cluster_rows"])
        write_rows(run_directory / "sls_sample_assignments.csv", sls_plan["sample_rows"])
        extra["sls_summary"] = sls_plan["summary"]
    train_loader, val_loader, test_loader = build_loaders(datasets, settings)
    model_document = prepare_model_document(model_spec, pretrained_override=None)
    model = build_model(model_document).to(device)
    parameters_total, parameters_trainable = model_parameter_counts(model)
    best_state, history, best_epoch, training_seconds = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        settings=settings,
        experiment=experiment,
        sample_alpha_by_path=sample_alpha_by_path,
    )
    model.load_state_dict(best_state, strict=True)
    if settings.keep_pth:
        torch.save(
            {
                "model_state_dict": best_state,
                "model_name": model_spec.name,
                "experiment_name": experiment.name,
                "best_epoch": best_epoch,
            },
            run_directory / "best_model.pth",
        )
    testing = evaluate_model(model, test_loader, device, "Final testing")
    metrics, report_path = save_run_outputs(
        run_directory=run_directory,
        model_spec=model_spec,
        experiment=experiment,
        model_document=model_document,
        settings=settings,
        dataset_info=dataset_info,
        history=history,
        best_epoch=best_epoch,
        training_seconds=training_seconds,
        testing=testing,
        parameters_total=parameters_total,
        parameters_trainable=parameters_trainable,
        extra=extra if extra else None,
    )
    result = {
        "model_name": model_spec.name,
        "experiment_name": experiment.name,
        "method": experiment.method,
        "status": "success",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "qwk": metrics["qwk"],
        "mae": metrics["mae"],
        "acc_2_3": metrics["acc_2_3"],
        "moderate_recall": metrics["class_wise"]["moderate"]["recall"],
        "over_recall": metrics["class_wise"]["over"]["recall"],
        "moderate_to_over_count": metrics["moderate_to_over_count"],
        "over_to_moderate_count": metrics["over_to_moderate_count"],
        "best_epoch": best_epoch,
        "run_directory": str(run_directory),
        "report_path": str(report_path),
    }
    if sls_plan is not None:
        result.update(
            {
                "sls_average_alpha_actual": sls_plan["summary"]["average_alpha_actual"],
                "sls_alpha_min": sls_plan["summary"]["alpha_min"],
                "sls_alpha_max": sls_plan["summary"]["alpha_max"],
                "sls_overlap_score_weighted_mean": sls_plan["summary"][
                    "overlap_score_weighted_mean"
                ],
            }
        )
    del model, train_loader, val_loader, test_loader, testing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, best_state


def train_internal_teacher(
    model_spec: ModelSpec,
    datasets: dict[str, Dataset],
    dataset_info: dict[str, Any],
    parent_directory: Path,
    device: torch.device,
    settings: RuntimeSettings,
    teacher_for: str,
) -> dict[str, torch.Tensor]:
    teacher_directory = parent_directory / "teacher_ce"
    teacher_directory.mkdir(parents=True, exist_ok=False)
    train_loader, val_loader, test_loader = build_loaders(datasets, settings)
    model_document = prepare_model_document(model_spec, pretrained_override=None)
    teacher = build_model(model_document).to(device)
    parameters_total, parameters_trainable = model_parameter_counts(teacher)
    best_state, history, best_epoch, training_seconds = train_model(
        model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        settings=settings,
        experiment=TEACHER_CE_SPEC,
    )
    teacher.load_state_dict(best_state, strict=True)
    testing = evaluate_model(teacher, test_loader, device, "Teacher testing")
    save_run_outputs(
        run_directory=teacher_directory,
        model_spec=model_spec,
        experiment=TEACHER_CE_SPEC,
        model_document=model_document,
        settings=settings,
        dataset_info=dataset_info,
        history=history,
        best_epoch=best_epoch,
        training_seconds=training_seconds,
        testing=testing,
        parameters_total=parameters_total,
        parameters_trainable=parameters_trainable,
        extra={"teacher_for": teacher_for},
    )
    del teacher, train_loader, val_loader, test_loader, testing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_state


def run_ce_retrain_experiment(
    model_spec: ModelSpec,
    experiment: ExperimentSpec,
    datasets: dict[str, Dataset],
    dataset_info: dict[str, Any],
    batch_directory: Path,
    device: torch.device,
    settings: RuntimeSettings,
    teacher_state: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    print("\n" + "=" * 80)
    print(f"Start job: {model_spec.name}/{experiment.name}")
    print("=" * 80)
    set_random_seed(settings.seed)
    run_directory = batch_directory / model_spec.name / experiment.name
    run_directory.mkdir(parents=True, exist_ok=False)
    teacher_source = "ce_baseline_cache" if teacher_state is not None else "internal_teacher"
    if teacher_state is None:
        teacher_state = train_internal_teacher(
            model_spec=model_spec,
            datasets=datasets,
            dataset_info=dataset_info,
            parent_directory=run_directory,
            device=device,
            settings=settings,
            teacher_for=experiment.name,
        )

    train_loader, val_loader, test_loader = build_loaders(datasets, settings)
    teacher_document = prepare_model_document(model_spec, pretrained_override=None)
    teacher = build_model(teacher_document).to(device)
    teacher.load_state_dict(teacher_state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_test = evaluate_model(teacher, test_loader, device, "CE teacher testing")
    write_json(run_directory / "teacher_test_metrics.json", teacher_test["metrics"])

    student_document = prepare_model_document(model_spec, pretrained_override=None)
    student = build_model(student_document).to(device)
    parameters_total, parameters_trainable = model_parameter_counts(student)
    best_state, history, best_epoch, training_seconds = train_model(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        settings=settings,
        experiment=experiment,
    )
    student.load_state_dict(best_state, strict=True)
    if settings.keep_pth:
        torch.save(
            {
                "model_state_dict": best_state,
                "model_name": model_spec.name,
                "experiment_name": experiment.name,
                "best_epoch": best_epoch,
            },
            run_directory / "best_model.pth",
        )
        torch.save(
            {
                "model_state_dict": teacher_state,
                "model_name": model_spec.name,
                "experiment_name": "teacher_ce",
            },
            run_directory / "teacher_model.pth",
        )
    testing = evaluate_model(student, test_loader, device, "Final CE-retrain testing")
    metrics, report_path = save_run_outputs(
        run_directory=run_directory,
        model_spec=model_spec,
        experiment=experiment,
        model_document=student_document,
        settings=settings,
        dataset_info=dataset_info,
        history=history,
        best_epoch=best_epoch,
        training_seconds=training_seconds,
        testing=testing,
        parameters_total=parameters_total,
        parameters_trainable=parameters_trainable,
        extra={
            "teacher_source": teacher_source,
            "teacher_test_accuracy": teacher_test["metrics"]["accuracy"],
            "teacher_test_acc_2_3": teacher_test["metrics"]["acc_2_3"],
            "equal_budget_control": True,
        },
    )
    result = {
        "model_name": model_spec.name,
        "experiment_name": experiment.name,
        "method": experiment.method,
        "status": "success",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "qwk": metrics["qwk"],
        "mae": metrics["mae"],
        "acc_2_3": metrics["acc_2_3"],
        "moderate_recall": metrics["class_wise"]["moderate"]["recall"],
        "over_recall": metrics["class_wise"]["over"]["recall"],
        "moderate_to_over_count": metrics["moderate_to_over_count"],
        "over_to_moderate_count": metrics["over_to_moderate_count"],
        "teacher_test_accuracy": teacher_test["metrics"]["accuracy"],
        "teacher_test_acc_2_3": teacher_test["metrics"]["acc_2_3"],
        "best_epoch": best_epoch,
        "run_directory": str(run_directory),
        "report_path": str(report_path),
    }
    del teacher, student, train_loader, val_loader, test_loader, teacher_test, testing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, teacher_state


def run_self_distillation_experiment(
    model_spec: ModelSpec,
    experiment: ExperimentSpec,
    datasets: dict[str, Dataset],
    dataset_info: dict[str, Any],
    batch_directory: Path,
    device: torch.device,
    settings: RuntimeSettings,
    teacher_state: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    print("\n" + "=" * 80)
    print(f"Start job: {model_spec.name}/{experiment.name}")
    print("=" * 80)
    set_random_seed(settings.seed)
    run_directory = batch_directory / model_spec.name / experiment.name
    run_directory.mkdir(parents=True, exist_ok=False)
    teacher_source = "ce_baseline_cache" if teacher_state is not None else "internal_teacher"
    if teacher_state is None:
        teacher_state = train_internal_teacher(
            model_spec=model_spec,
            datasets=datasets,
            dataset_info=dataset_info,
            parent_directory=run_directory,
            device=device,
            settings=settings,
            teacher_for=experiment.name,
        )

    train_loader, val_loader, test_loader = build_loaders(datasets, settings)
    teacher_document = prepare_model_document(model_spec, pretrained_override=None)
    teacher = build_model(teacher_document).to(device)
    teacher.load_state_dict(teacher_state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    teacher_val = evaluate_model(teacher, val_loader, device, "Teacher validation")
    teacher_test = evaluate_model(teacher, test_loader, device, "Teacher testing")
    write_json(run_directory / "teacher_val_metrics.json", teacher_val["metrics"])
    write_json(run_directory / "teacher_test_metrics.json", teacher_test["metrics"])
    write_json(
        run_directory / "teacher_val_probability_diagnostics.json",
        probability_diagnostics(teacher_val),
    )
    write_json(
        run_directory / "teacher_test_probability_diagnostics.json",
        probability_diagnostics(teacher_test),
    )

    student_document = prepare_model_document(model_spec, pretrained_override=None)
    student = build_model(student_document).to(device)
    parameters_total, parameters_trainable = model_parameter_counts(student)
    best_state, history, best_epoch, training_seconds = train_model(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        settings=settings,
        experiment=experiment,
        teacher=teacher,
    )
    student.load_state_dict(best_state, strict=True)
    if settings.keep_pth:
        torch.save(
            {
                "model_state_dict": best_state,
                "model_name": model_spec.name,
                "experiment_name": experiment.name,
                "best_epoch": best_epoch,
            },
            run_directory / "best_model.pth",
        )
        torch.save(
            {
                "model_state_dict": teacher_state,
                "model_name": model_spec.name,
                "experiment_name": "teacher_ce",
            },
            run_directory / "teacher_model.pth",
        )
    testing = evaluate_model(student, test_loader, device, "Final student testing")
    metrics, report_path = save_run_outputs(
        run_directory=run_directory,
        model_spec=model_spec,
        experiment=experiment,
        model_document=student_document,
        settings=settings,
        dataset_info=dataset_info,
        history=history,
        best_epoch=best_epoch,
        training_seconds=training_seconds,
        testing=testing,
        parameters_total=parameters_total,
        parameters_trainable=parameters_trainable,
        extra={
            "teacher_source": teacher_source,
            "teacher_test_accuracy": teacher_test["metrics"]["accuracy"],
            "teacher_test_acc_2_3": teacher_test["metrics"]["acc_2_3"],
        },
    )
    result = {
        "model_name": model_spec.name,
        "experiment_name": experiment.name,
        "method": experiment.method,
        "status": "success",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "qwk": metrics["qwk"],
        "mae": metrics["mae"],
        "acc_2_3": metrics["acc_2_3"],
        "moderate_recall": metrics["class_wise"]["moderate"]["recall"],
        "over_recall": metrics["class_wise"]["over"]["recall"],
        "moderate_to_over_count": metrics["moderate_to_over_count"],
        "over_to_moderate_count": metrics["over_to_moderate_count"],
        "teacher_test_accuracy": teacher_test["metrics"]["accuracy"],
        "teacher_test_acc_2_3": teacher_test["metrics"]["acc_2_3"],
        "best_epoch": best_epoch,
        "run_directory": str(run_directory),
        "report_path": str(report_path),
    }
    del teacher, student, train_loader, val_loader, test_loader
    del teacher_val, teacher_test, testing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, teacher_state


def run_dry_check(
    model_specs: Sequence[ModelSpec],
    experiments: Sequence[ExperimentSpec],
    datasets: dict[str, Dataset],
    dataset_info: dict[str, Any],
    device: torch.device,
    settings: RuntimeSettings,
) -> None:
    print(f"Dataset root: {DEFAULT_DATASET_ROOT}")
    print("Dataset summary:")
    for split_name, info in dataset_info.items():
        print(f"  {split_name}: total={info['total']}, classes={info['classes']}")
    images = torch.randn(4, 3, settings.image_size, settings.image_size, device=device)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    print("\nModel and loss checks with pretrained=False:")
    for model_spec in model_specs:
        model_document = prepare_model_document(model_spec, pretrained_override=False)
        model = build_model(model_document).to(device)
        model.train()
        for experiment in experiments:
            if experiment.method == "self_distill":
                teacher = build_model(model_document).to(device)
                teacher.eval()
                loss, components, logits = compute_training_loss(
                    model=model,
                    images=images,
                    labels=labels,
                    experiment=experiment,
                    epoch=1,
                    teacher=teacher,
                )
                del teacher
            elif experiment.method in SLS_METHODS:
                sample_alphas = torch.full(
                    (labels.shape[0],),
                    float(experiment.sls_average_alpha),
                    dtype=torch.float32,
                    device=device,
                )
                loss, components, logits = compute_training_loss(
                    model=model,
                    images=images,
                    labels=labels,
                    experiment=experiment,
                    epoch=1,
                    sample_alphas=sample_alphas,
                )
            else:
                epoch = experiment.bootstrap_warmup_epochs + 1
                loss, components, logits = compute_training_loss(
                    model=model,
                    images=images,
                    labels=labels,
                    experiment=experiment,
                    epoch=epoch,
                )
            expected_shape = (images.shape[0], len(CLASS_NAMES))
            if tuple(logits.shape) != expected_shape:
                raise RuntimeError(
                    f"{model_spec.name}/{experiment.name} output should be "
                    f"{expected_shape}, got {tuple(logits.shape)}"
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"{model_spec.name}/{experiment.name} loss is not finite.")
            model.zero_grad(set_to_none=True)
            loss.backward()
            print(
                f"  {model_spec.name}/{experiment.name}: output={tuple(logits.shape)}, "
                f"loss={float(loss.detach()):.6f}, components={components}, PASS"
            )
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
            f"<td>{html.escape(str(result.get('seed', '')))}</td>"
            f"<td>{html.escape(str(result['model_name']))}</td>"
            f"<td>{html.escape(str(result['experiment_name']))}</td>"
            f"<td>{html.escape(str(result['status']))}</td>"
            f"<td>{result.get('accuracy', '')}</td>"
            f"<td>{result.get('macro_f1', '')}</td>"
            f"<td>{result.get('acc_2_3', '')}</td>"
            f"<td>{result.get('moderate_to_over_count', '')}</td>"
            f"<td>{result.get('over_to_moderate_count', '')}</td>"
            + (
                f'<td><a href="{html.escape(report_link, quote=True)}">report</a></td>'
                if report_link
                else "<td></td>"
            )
            + "</tr>"
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Soft supervision diagnostics</title>
<style>body{{max-width:1120px;margin:36px auto;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:9px;text-align:center}}th{{background:#f4f6fa}}</style>
</head><body><h1>Soft supervision diagnostics</h1>
<table><thead><tr><th>seed</th><th>model</th><th>experiment</th><th>status</th><th>accuracy</th><th>macro-F1</th><th>acc_2_3</th><th>M to O</th><th>O to M</th><th>report</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    path = batch_directory / "summary.html"
    path.write_text(document, encoding="utf-8")
    return path


def mean_std(values: Sequence[Any]) -> tuple[float | None, float | None]:
    numeric = [
        float(value)
        for value in values
        if value is not None and value != ""
    ]
    if not numeric:
        return None, None
    if len(numeric) == 1:
        return numeric[0], 0.0
    return float(np.mean(numeric)), float(np.std(numeric, ddof=1))


def write_aggregate_report(
    batch_directory: Path,
    results: Sequence[dict[str, Any]],
) -> Path:
    metric_names = (
        "accuracy",
        "macro_f1",
        "qwk",
        "mae",
        "acc_2_3",
        "moderate_recall",
        "over_recall",
        "moderate_to_over_count",
        "over_to_moderate_count",
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if result.get("status") != "success":
            continue
        groups[(str(result["model_name"]), str(result["experiment_name"]))].append(result)

    rows = []
    for (model_name, experiment_name), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "model_name": model_name,
            "experiment_name": experiment_name,
            "n": len(items),
            "seeds": " ".join(str(item.get("seed", "")) for item in items),
        }
        for metric_name in metric_names:
            mean_value, std_value = mean_std([item.get(metric_name) for item in items])
            row[f"{metric_name}_mean"] = mean_value
            row[f"{metric_name}_std"] = std_value
        rows.append(row)
    write_rows(batch_directory / "aggregate_summary.csv", rows)

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_name']))}</td>"
            f"<td>{html.escape(str(row['experiment_name']))}</td>"
            f"<td>{row['n']}</td>"
            f"<td>{row.get('accuracy_mean')}</td>"
            f"<td>{row.get('accuracy_std')}</td>"
            f"<td>{row.get('acc_2_3_mean')}</td>"
            f"<td>{row.get('acc_2_3_std')}</td>"
            f"<td>{row.get('moderate_to_over_count_mean')}</td>"
            f"<td>{row.get('over_to_moderate_count_mean')}</td>"
            + "</tr>"
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Soft supervision aggregate summary</title>
<style>body{{max-width:1120px;margin:36px auto;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:9px;text-align:center}}th{{background:#f4f6fa}}</style>
</head><body><h1>Soft supervision aggregate summary</h1>
<table><thead><tr><th>model</th><th>experiment</th><th>n</th><th>accuracy mean</th><th>accuracy std</th><th>acc_2_3 mean</th><th>acc_2_3 std</th><th>M to O mean</th><th>O to M mean</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
</body></html>"""
    path = batch_directory / "aggregate_summary.html"
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


def select_experiments(names: Sequence[str] | None) -> list[ExperimentSpec]:
    by_name = {experiment.name: experiment for experiment in EXPERIMENT_LIST}
    if not names:
        return [by_name[name] for name in DEFAULT_EXPERIMENT_NAMES]
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    return [by_name[name] for name in names]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run four-class CE, equal-budget CE retrain, label smoothing, "
            "bootstrap-soft, and self-distillation."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--experiments", nargs="+")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=PYCHARM_DEVICE)
    parser.add_argument("--dry-run", action="store_true", default=PYCHARM_DRY_RUN)
    parser.add_argument("--keep-pth", action="store_true", default=PYCHARM_KEEP_PTH)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_SETTINGS.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SETTINGS.batch_size)
    parser.add_argument("--val-batch-size", type=int, default=DEFAULT_SETTINGS.val_batch_size)
    parser.add_argument("--test-batch-size", type=int, default=DEFAULT_SETTINGS.test_batch_size)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_SETTINGS.num_workers)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds to repeat. Example: --seeds 2026 2027 2028",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    model_specs = select_models(args.models)
    experiments = select_experiments(args.experiments)
    if args.list_models:
        print("Models:")
        for model_spec in model_specs:
            print(f"  {model_spec.name}: {model_spec.config_path}")
    if args.list_experiments:
        print("Experiments:")
        for experiment in EXPERIMENT_LIST:
            print(f"  {experiment.name}: {experiment.description}")
    if args.list_models or args.list_experiments:
        return

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0 or args.val_batch_size <= 0 or args.test_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if not args.seeds:
        raise ValueError("--seeds must include at least one seed.")
    seeds = [int(seed) for seed in args.seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains duplicate values: {seeds}")

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    base_settings = replace(
        DEFAULT_SETTINGS,
        seed=seeds[0],
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        val_batch_size=int(args.val_batch_size),
        test_batch_size=int(args.test_batch_size),
        num_workers=int(args.num_workers),
        keep_pth=bool(args.keep_pth),
    )
    device = resolve_device(args.device)
    datasets = build_datasets(dataset_root, base_settings)
    info = dataset_summary(datasets)

    if args.dry_run:
        run_dry_check(model_specs, experiments, datasets, info, device, base_settings)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_directory = RESULTS_ROOT / f"batch_{timestamp}"
    batch_directory.mkdir(parents=True, exist_ok=False)
    write_json(
        batch_directory / "batch_config.json",
        {
            "dataset_root": str(dataset_root),
            "models": [asdict(model_spec) for model_spec in model_specs],
            "experiments": [asdict(experiment) for experiment in experiments],
            "runtime": asdict(base_settings),
            "seeds": seeds,
        },
    )

    results = []
    for seed in seeds:
        settings = replace(base_settings, seed=seed)
        seed_directory = batch_directory / f"seed_{seed}"
        seed_directory.mkdir(parents=True, exist_ok=False)
        teacher_state_cache: dict[str, dict[str, torch.Tensor]] = {}
        sls_plan_cache: dict[str, dict[str, Any]] = {}
        for model_spec in model_specs:
            for experiment in experiments:
                try:
                    if experiment.method == "self_distill":
                        result, teacher_state = run_self_distillation_experiment(
                            model_spec=model_spec,
                            experiment=experiment,
                            datasets=datasets,
                            dataset_info=info,
                            batch_directory=seed_directory,
                            device=device,
                            settings=settings,
                            teacher_state=teacher_state_cache.get(model_spec.name),
                        )
                        teacher_state_cache[model_spec.name] = teacher_state
                    elif experiment.method == "ce_retrain":
                        result, teacher_state = run_ce_retrain_experiment(
                            model_spec=model_spec,
                            experiment=experiment,
                            datasets=datasets,
                            dataset_info=info,
                            batch_directory=seed_directory,
                            device=device,
                            settings=settings,
                            teacher_state=teacher_state_cache.get(model_spec.name),
                        )
                        teacher_state_cache[model_spec.name] = teacher_state
                    else:
                        result, best_state = run_standard_experiment(
                            model_spec=model_spec,
                            experiment=experiment,
                            datasets=datasets,
                            dataset_info=info,
                            batch_directory=seed_directory,
                            device=device,
                            settings=settings,
                            sls_plan_cache=sls_plan_cache,
                        )
                        if experiment.name == "ce_baseline":
                            teacher_state_cache[model_spec.name] = best_state
                except Exception as error:
                    failure_directory = seed_directory / model_spec.name / experiment.name
                    failure_directory.mkdir(parents=True, exist_ok=True)
                    (failure_directory / "failure.txt").write_text(
                        f"{type(error).__name__}: {error}",
                        encoding="utf-8",
                    )
                    print(f"{model_spec.name}/{experiment.name} seed={seed} failed: {error}")
                    result = {
                        "model_name": model_spec.name,
                        "experiment_name": experiment.name,
                        "method": experiment.method,
                        "status": "failed",
                        "accuracy": None,
                        "macro_f1": None,
                        "qwk": None,
                        "mae": None,
                        "acc_2_3": None,
                        "moderate_recall": None,
                        "over_recall": None,
                        "moderate_to_over_count": None,
                        "over_to_moderate_count": None,
                        "best_epoch": None,
                        "run_directory": str(failure_directory),
                        "report_path": "",
                    }
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                result["seed"] = seed
                results.append(result)
        del teacher_state_cache, sls_plan_cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary_path = write_batch_report(batch_directory, results)
    aggregate_path = write_aggregate_report(batch_directory, results)
    print(f"\nAll soft-supervision diagnostics finished: {summary_path}")
    print(f"Aggregate mean/std report: {aggregate_path}")
    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
