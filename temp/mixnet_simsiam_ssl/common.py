"""Shared utilities for the MixNet-S SimSiam experiment.

The files in this folder are intentionally standalone. They reuse only public
PyTorch, torchvision, and timm APIs so this temporary experiment does not need
changes in src/.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable short runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class SeedWorkerInit:
    """Pickle-safe dataloader worker seeding for Windows spawn workers."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def __call__(self, worker_id: int) -> None:
        worker_seed = self.seed + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)


def make_worker_init_fn(seed: int) -> SeedWorkerInit:
    """Create a dataloader worker initializer with deterministic worker seeds."""
    return SeedWorkerInit(seed)


def list_image_files(roots: Iterable[str | Path]) -> list[Path]:
    """Collect image files recursively from one or more directories."""
    image_paths: list[Path] = []
    for root in roots:
        resolved_root = resolve_project_path(root)
        if not resolved_root.is_dir():
            raise FileNotFoundError(f"Image root does not exist: {resolved_root}")
        image_paths.extend(
            path.resolve()
            for path in resolved_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise ValueError(f"No image files found under: {list(roots)}")
    return image_paths


class TwoViewImageDataset(Dataset):
    """Return two independently augmented views from the same source image."""

    def __init__(self, image_paths: Sequence[str | Path], transform) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        return self.transform(image), self.transform(image), str(image_path)


class ImageFolderDataset(Dataset):
    """A small ImageFolder clone that also returns the source path."""

    def __init__(
        self,
        root: str | Path,
        transform=None,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.root = resolve_project_path(root)
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset split does not exist: {self.root}")

        available_classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if not available_classes:
            raise ValueError(f"No class folders found in: {self.root}")

        if class_to_idx is None:
            self.classes = available_classes
            self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        else:
            normalized = {str(name): int(index) for name, index in class_to_idx.items()}
            missing = sorted(set(normalized) - set(available_classes))
            unexpected = sorted(set(available_classes) - set(normalized))
            if missing or unexpected:
                raise ValueError(
                    f"class_to_idx does not match {self.root}: missing={missing}, "
                    f"unexpected={unexpected}"
                )
            self.class_to_idx = normalized
            self.classes = [name for name, _ in sorted(normalized.items(), key=lambda item: item[1])]

        self.samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            label = self.class_to_idx[class_name]
            class_images = sorted(
                path.resolve()
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not class_images:
                raise ValueError(f"No images found in class folder: {class_dir}")
            self.samples.extend((path, label) for path in class_images)
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(image_path)


def build_ssl_transform(
    image_size: int,
    crop_min: float = 0.55,
    jitter_strength: float = 0.35,
) -> T.Compose:
    """Build SimSiam augmentations for tea patch pretraining."""
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(crop_min, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=jitter_strength,
                        contrast=jitter_strength,
                        saturation=jitter_strength,
                        hue=min(0.1, jitter_strength / 4.0),
                    )
                ],
                p=0.8,
            ),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_finetune_train_transform(
    image_size: int,
    crop_min: float = 0.85,
    jitter_strength: float = 0.08,
    mode: str = "crop",
) -> T.Compose:
    """Build label-preserving train-time augmentation for five-class finetuning."""
    normalized_mode = str(mode).lower()
    transforms: list[Any]
    if normalized_mode == "crop":
        transforms = [
            T.RandomResizedCrop(image_size, scale=(crop_min, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]
    elif normalized_mode in {"resize", "resize_flip"}:
        transforms = [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]
    else:
        raise ValueError(f"Unsupported finetune transform mode: {mode}")

    if jitter_strength > 0:
        transforms.append(
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=jitter_strength,
                        contrast=jitter_strength,
                        saturation=jitter_strength,
                        hue=0.02,
                    )
                ],
                p=0.35,
            )
        )
    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return T.Compose(transforms)


def build_eval_transform(image_size: int) -> T.Compose:
    """Build deterministic validation/test preprocessing."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _to_feature_vector(features: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    if isinstance(features, (tuple, list)):
        features = features[-1]
    if features.ndim == 2:
        return features
    if features.ndim == 4:
        return torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
    if features.ndim == 3:
        return features.mean(dim=1)
    return torch.flatten(features, 1)


class TimmFeatureBackbone(nn.Module):
    """Wrap a timm model and always return a 2D feature tensor."""

    def __init__(self, model_name: str, pretrained: bool, image_size: int) -> None:
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.out_features = self._infer_out_features(image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        return _to_feature_vector(features)

    def _infer_out_features(self, image_size: int) -> int:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, image_size, image_size)
                features = self.forward(dummy)
        finally:
            self.model.train(was_training)
        if features.ndim != 2 or features.shape[1] <= 0:
            raise ValueError(f"Backbone returned invalid feature shape: {tuple(features.shape)}")
        return int(features.shape[1])


class ProjectionHead(nn.Module):
    """Three-layer SimSiam projection MLP."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHead(nn.Module):
    """Two-layer SimSiam prediction MLP."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimSiamModel(nn.Module):
    """MixNet-S encoder plus SimSiam projector and predictor."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        image_size: int,
        projection_dim: int = 2048,
        projection_hidden_dim: int = 2048,
        prediction_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = TimmFeatureBackbone(model_name, pretrained=pretrained, image_size=image_size)
        self.projector = ProjectionHead(
            self.backbone.out_features,
            projection_hidden_dim,
            projection_dim,
        )
        self.predictor = PredictionHead(projection_dim, prediction_hidden_dim, projection_dim)

    def forward_one(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        z = self.projector(features)
        p = self.predictor(z)
        return p, z

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p1, z1 = self.forward_one(x1)
        p2, z2 = self.forward_one(x2)
        return p1, z1, p2, z2


class TimmClassifier(nn.Module):
    """Feature backbone plus a linear classification head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        image_size: int,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = TimmFeatureBackbone(model_name, pretrained=pretrained, image_size=image_size)
        dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.head = nn.Sequential(dropout, nn.Linear(self.backbone.out_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def simsiam_loss(
    p1: torch.Tensor,
    z1: torch.Tensor,
    p2: torch.Tensor,
    z2: torch.Tensor,
) -> torch.Tensor:
    """Negative cosine similarity with stop-gradient on target projections."""
    return 0.5 * (
        negative_cosine_similarity(p1, z2.detach())
        + negative_cosine_similarity(p2, z1.detach())
    )


def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def build_optimizer(
    parameters,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = max(0, int(warmup_epochs))
    epochs = max(1, int(epochs))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        progress_epochs = max(1, epochs - warmup_epochs)
        progress = float(epoch_index - warmup_epochs + 1) / float(progress_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    loss: float | None = None,
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(true, pred):
        if 0 <= target < num_classes and 0 <= prediction < num_classes:
            matrix[target, prediction] += 1

    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    per_class = []
    f1_values = []
    recalls = []
    for class_index in range(num_classes):
        tp = float(matrix[class_index, class_index])
        fp = float(matrix[:, class_index].sum() - tp)
        fn = float(matrix[class_index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class.append(
            {
                "class_index": class_index,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(matrix[class_index, :].sum()),
            }
        )
        f1_values.append(f1)
        recalls.append(recall)

    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "mae": float(np.mean(np.abs(true - pred))) if total else 0.0,
        "qwk": quadratic_weighted_kappa(matrix),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def quadratic_weighted_kappa(confusion_matrix: np.ndarray) -> float:
    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    num_classes = matrix.shape[0]
    total = matrix.sum()
    if total <= 0 or num_classes <= 1:
        return 0.0

    observed = matrix / total
    row_hist = observed.sum(axis=1)
    col_hist = observed.sum(axis=0)
    expected = np.outer(row_hist, col_hist)
    weights = np.zeros_like(observed)
    denom = float((num_classes - 1) ** 2)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / denom

    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    if denominator <= 0:
        return 0.0
    return 1.0 - numerator / denominator


def save_json(data: Any, path: str | Path) -> None:
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_history_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a state_dict from common checkpoint layouts."""
    if isinstance(checkpoint, dict):
        for key in ("backbone_state_dict", "model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return dict(value)
    if isinstance(checkpoint, dict):
        return dict(checkpoint)
    raise TypeError("Checkpoint does not contain a state_dict.")


def load_backbone_checkpoint(
    backbone: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load SSL weights into the finetune backbone."""
    resolved = resolve_project_path(checkpoint_path)
    checkpoint = torch.load(resolved, map_location="cpu")
    raw_state = extract_state_dict(checkpoint)

    stripped_state: dict[str, torch.Tensor] = {}
    current_keys = set(backbone.state_dict())
    for key, value in raw_state.items():
        candidates = [
            key,
            key.removeprefix("backbone."),
            key.removeprefix("module.backbone."),
        ]
        matched_key = next((candidate for candidate in candidates if candidate in current_keys), None)
        if matched_key is not None:
            stripped_state[matched_key] = value

    if not stripped_state:
        raise RuntimeError(f"No backbone weights from {resolved} matched the current model.")
    incompatible = backbone.load_state_dict(stripped_state, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
