"""FineVAD-style progressive MixNet-S experiment for 0/1/2/3/4h black tea images.

All new code for this attempt lives in temp/finevad_mixnet_progressive on purpose.
The shared src/ training framework is left untouched so this can be deleted or moved
as one experimental sandbox.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover - compatibility with older torch releases
    from torch.cuda.amp import GradScaler, autocast


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = THIS_DIR / "default_config.yaml"
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class StageSpec:
    name: str
    epochs: int
    lr: float


class ImageFolderWithPaths(Dataset):
    """Small local image-folder dataset: root/split/class_name/image."""

    def __init__(
        self,
        root: str | Path,
        transform: Any = None,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.root = resolve_project_path(root)
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset split directory not found: {self.root}")

        available_classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if not available_classes:
            raise ValueError(f"No class folders found in: {self.root}")

        if class_to_idx:
            normalized = {str(key): int(value) for key, value in class_to_idx.items()}
            missing = sorted(set(normalized) - set(available_classes))
            extra = sorted(set(available_classes) - set(normalized))
            if missing or extra:
                raise ValueError(
                    f"class_to_idx does not match folders under {self.root}: "
                    f"missing={missing}, extra={extra}"
                )
            expected = list(range(len(normalized)))
            actual = sorted(normalized.values())
            if actual != expected:
                raise ValueError(f"class_to_idx values must be contiguous from 0, got {actual}")
            self.class_to_idx = normalized
            self.classes = [name for name, _ in sorted(normalized.items(), key=lambda item: item[1])]
        else:
            self.classes = available_classes
            self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            files = sorted(
                path.resolve()
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not files:
                raise ValueError(f"No images found in class folder: {class_dir}")
            label = self.class_to_idx[class_name]
            self.samples.extend((path, label) for path in files)
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label), str(image_path)


class GatedLogitFusion(nn.Module):
    """Project logits to feature space, then gate them into the base feature."""

    def __init__(self, feature_dim: int, context_classes: int, dropout: float) -> None:
        super().__init__()
        self.context_projection = nn.Sequential(
            nn.Linear(context_classes, feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )
        self.proposal = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        context = self.context_projection(torch.softmax(logits, dim=1))
        joined = torch.cat([features, context], dim=1)
        return self.norm(features + self.gate(joined) * self.proposal(joined))


class FineVADMixNet(nn.Module):
    """MixNet-S backbone with coarse, mid, and fine progressive heads."""

    def __init__(
        self,
        backbone_name: str = "mixnet_s",
        pretrained: bool = True,
        num_classes: int = 5,
        mid_classes: int = 3,
        coarse_classes: int = 2,
        dropout: float = 0.2,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feature_dim = int(getattr(self.backbone, "num_features", 0) or self._infer_feature_dim())
        self.feature_dim = feature_dim
        self.num_classes = int(num_classes)
        self.mid_classes = int(mid_classes)
        self.coarse_classes = int(coarse_classes)

        self.head_coarse = nn.Linear(feature_dim, coarse_classes)
        self.fusion_coarse_to_mid = GatedLogitFusion(feature_dim, coarse_classes, dropout)
        self.head_mid = nn.Linear(feature_dim, mid_classes)
        self.fusion_mid_to_fine = GatedLogitFusion(feature_dim, mid_classes, dropout)
        self.head_fine = nn.Linear(feature_dim, num_classes)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def _infer_feature_dim(self) -> int:
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                features = self.backbone(torch.zeros(1, 3, 224, 224))
        finally:
            self.backbone.train(was_training)
        if features.ndim != 2:
            features = torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
        return int(features.shape[1])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        if features.ndim != 2:
            features = torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
        logits_coarse = self.head_coarse(features)
        feat_mid = self.fusion_coarse_to_mid(features, logits_coarse)
        logits_mid = self.head_mid(feat_mid)
        feat_fine = self.fusion_mid_to_fine(feat_mid, logits_mid)
        logits_fine = self.head_fine(feat_fine)
        return {
            "features": features,
            "projected_features": self.projection(features),
            "feat_mid": feat_mid,
            "feat_fine": feat_fine,
            "logits_coarse": logits_coarse,
            "logits_mid": logits_mid,
            "logits_fine": logits_fine,
        }


class ProgressiveFineVADLoss(nn.Module):
    """Stage-specific loss for the FineVAD-style progressive attempt."""

    def __init__(
        self,
        coarse_map: list[int],
        mid_map: list[int],
        lambda_supcon: float = 0.2,
        supcon_temperature: float = 0.1,
        stage2_consistency_weight: float = 1.0,
        stage2_entropy_weight: float = 0.01,
        sibling_penalty_weight: float = 2.0,
        stage3_aux_coarse_weight: float = 0.1,
        stage3_aux_mid_weight: float = 0.05,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_supcon = float(lambda_supcon)
        self.supcon_temperature = float(supcon_temperature)
        self.stage2_consistency_weight = float(stage2_consistency_weight)
        self.stage2_entropy_weight = float(stage2_entropy_weight)
        self.sibling_penalty_weight = float(sibling_penalty_weight)
        self.stage3_aux_coarse_weight = float(stage3_aux_coarse_weight)
        self.stage3_aux_mid_weight = float(stage3_aux_mid_weight)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("coarse_map", torch.tensor(coarse_map, dtype=torch.long))
        self.register_buffer("mid_map", torch.tensor(mid_map, dtype=torch.long))
        self.register_buffer("mid_to_coarse", torch.tensor([0, 1, 1], dtype=torch.long))
        self.register_buffer("fine_sibling_mask", self._build_sibling_mask(mid_map))

    @staticmethod
    def _build_sibling_mask(mid_map: list[int]) -> torch.Tensor:
        size = len(mid_map)
        mask = torch.zeros(size, size, dtype=torch.float32)
        for true_index in range(size):
            for pred_index in range(size):
                if true_index != pred_index and mid_map[true_index] == mid_map[pred_index]:
                    mask[true_index, pred_index] = 1.0
        return mask

    def targets(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.coarse_map[labels], self.mid_map[labels]

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        stage_name: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        coarse_targets, mid_targets = self.targets(labels)
        if stage_name == "stage1_coarse":
            return self._stage1_loss(outputs, coarse_targets)
        if stage_name == "stage2_mid_fusion":
            return self._stage2_loss(outputs)
        if stage_name == "stage3_fine":
            return self._stage3_loss(outputs, labels, coarse_targets, mid_targets)
        raise ValueError(f"Unknown stage: {stage_name}")

    def _stage1_loss(
        self,
        outputs: dict[str, torch.Tensor],
        coarse_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        coarse_ce = F.cross_entropy(outputs["logits_coarse"], coarse_targets)
        supcon = self._supervised_contrastive_loss(outputs["projected_features"], coarse_targets)
        loss = coarse_ce + self.lambda_supcon * supcon
        return loss, {
            "coarse_ce": float(coarse_ce.detach().cpu()),
            "supcon": float(supcon.detach().cpu()),
        }

    def _stage2_loss(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        coarse_prob = torch.softmax(outputs["logits_coarse"].detach(), dim=1)
        mid_prob = torch.softmax(outputs["logits_mid"], dim=1)
        mid_as_coarse = torch.stack(
            [
                mid_prob[:, self.mid_to_coarse == 0].sum(dim=1),
                mid_prob[:, self.mid_to_coarse == 1].sum(dim=1),
            ],
            dim=1,
        ).clamp_min(1e-8)
        consistency = -(coarse_prob * torch.log(mid_as_coarse)).sum(dim=1).mean()
        entropy = -(mid_prob * torch.log(mid_prob.clamp_min(1e-8))).sum(dim=1).mean()
        loss = self.stage2_consistency_weight * consistency + self.stage2_entropy_weight * entropy
        return loss, {
            "mid_coarse_consistency": float(consistency.detach().cpu()),
            "mid_entropy": float(entropy.detach().cpu()),
        }

    def _stage3_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        coarse_targets: torch.Tensor,
        mid_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        fine_ce = F.cross_entropy(
            outputs["logits_fine"],
            labels,
            label_smoothing=self.label_smoothing,
        )
        fine_prob = torch.softmax(outputs["logits_fine"], dim=1)
        sibling_mass = (fine_prob * self.fine_sibling_mask[labels]).sum(dim=1)
        sibling_penalty = -torch.log((1.0 - sibling_mass).clamp_min(1e-8)).mean()

        aux_coarse = F.cross_entropy(outputs["logits_coarse"], coarse_targets)
        aux_mid = F.cross_entropy(outputs["logits_mid"], mid_targets)
        loss = (
            fine_ce
            + self.sibling_penalty_weight * sibling_penalty
            + self.stage3_aux_coarse_weight * aux_coarse
            + self.stage3_aux_mid_weight * aux_mid
        )
        return loss, {
            "fine_ce": float(fine_ce.detach().cpu()),
            "sibling_penalty": float(sibling_penalty.detach().cpu()),
            "aux_coarse": float(aux_coarse.detach().cpu()),
            "aux_mid": float(aux_mid.detach().cpu()),
        }

    def _supervised_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        logits = features @ features.T / self.supcon_temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        label_matrix = torch.eq(labels[:, None], labels[None, :]).to(features.dtype)
        self_mask = torch.eye(labels.numel(), device=labels.device, dtype=features.dtype)
        positives = label_matrix * (1.0 - self_mask)
        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
        positive_count = positives.sum(dim=1)
        valid = positive_count > 0
        if not torch.any(valid):
            return features.new_zeros(())
        mean_log_prob = (positives * log_prob).sum(dim=1)[valid] / positive_count[valid]
        return -mean_log_prob.mean()


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(name)


def build_transform(config: dict[str, Any], train: bool) -> T.Compose:
    image_size = int(config["data"].get("image_size", 408))
    mean = tuple(float(value) for value in config["data"].get("imagenet_mean", [0.485, 0.456, 0.406]))
    std = tuple(float(value) for value in config["data"].get("imagenet_std", [0.229, 0.224, 0.225]))
    transforms: list[Any] = [T.Resize((image_size, image_size), antialias=True)]
    if train:
        transforms.extend([T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)])
        if bool(config["data"].get("autoaugment", False)):
            transforms.append(T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET))
        if bool(config["data"].get("color_jitter", True)):
            transforms.append(
                T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01)
            )
    transforms.extend([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return T.Compose(transforms)


def build_loaders(config: dict[str, Any], args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_root = resolve_project_path(config.get("dataset_root", "datasets_01234_BaSic"))
    class_to_idx = config.get("class_to_idx")
    train_dataset = ImageFolderWithPaths(dataset_root / "train", build_transform(config, train=True), class_to_idx)
    val_dataset = ImageFolderWithPaths(dataset_root / "val", build_transform(config, train=False), class_to_idx)
    test_dataset = ImageFolderWithPaths(dataset_root / "test", build_transform(config, train=False), class_to_idx)

    batch_size = int(args.batch_size or config["train"].get("batch_size", 32))
    eval_batch_size = int(args.eval_batch_size or config["train"].get("eval_batch_size", 64))
    num_workers = int(args.num_workers if args.num_workers is not None else config["train"].get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


def count_by_class(dataset: ImageFolderWithPaths) -> dict[str, int]:
    counts = Counter(int(target) for target in dataset.targets)
    return {
        class_name: int(counts.get(class_index, 0))
        for class_index, class_name in enumerate(dataset.classes)
    }


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def configure_stage_trainable(model: FineVADMixNet, stage_name: str) -> None:
    set_trainable(model, False)
    if stage_name == "stage1_coarse":
        set_trainable(model.backbone, True)
        set_trainable(model.head_coarse, True)
        set_trainable(model.projection, True)
    elif stage_name == "stage2_mid_fusion":
        set_trainable(model.fusion_coarse_to_mid, True)
        set_trainable(model.head_mid, True)
    elif stage_name == "stage3_fine":
        set_trainable(model, True)
    else:
        raise ValueError(f"Unknown stage: {stage_name}")


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = list(trainable_parameters(model))
    if not params:
        raise ValueError("No trainable parameters for this stage.")
    return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs)),
        eta_min=float(min_lr),
    )


def make_grad_scaler(device: torch.device, enabled: bool) -> GradScaler:
    try:
        return GradScaler(device.type, enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    try:
        return autocast(device_type=device.type, enabled=enabled)
    except TypeError:
        return autocast(enabled=enabled)


def train_one_epoch(
    model: FineVADMixNet,
    loss_fn: ProgressiveFineVADLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    stage_name: str,
    epoch_index: int,
    stage_epochs: int,
    use_amp: bool,
    max_train_batches: int | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    optimizer_steps = 0
    component_sums: dict[str, float] = {}
    total_batches = len(loader) if max_train_batches is None else min(len(loader), max_train_batches)
    iterator = loader if max_train_batches is None else islice(loader, max_train_batches)
    pbar = tqdm(iterator, total=total_batches, desc=f"{stage_name} {epoch_index + 1}/{stage_epochs}")
    for _batch_index, (images, labels, _paths) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            outputs = model(images)
            loss, components = loss_fn(outputs, labels, stage_name)
        if use_amp:
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                optimizer_steps += 1
        else:
            loss.backward()
            optimizer.step()
            optimizer_steps += 1

        if stage_name == "stage1_coarse":
            logits = outputs["logits_coarse"]
            metric_targets = loss_fn.coarse_map[labels]
        elif stage_name == "stage2_mid_fusion":
            logits = outputs["logits_mid"]
            metric_targets = loss_fn.mid_map[labels]
        else:
            logits = outputs["logits_fine"]
            metric_targets = labels
        predictions = logits.argmax(dim=1)
        correct = int((predictions == metric_targets).sum().item())
        batch_size = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += correct
        total_samples += batch_size
        for key, value in components.items():
            component_sums[key] = component_sums.get(key, 0.0) + value * batch_size
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}", acc=f"{total_correct / total_samples:.4f}")

    metrics = {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
        "optimizer_steps": optimizer_steps,
    }
    metrics.update({key: value / max(1, total_samples) for key, value in component_sums.items()})
    return metrics


@torch.no_grad()
def evaluate(
    model: FineVADMixNet,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    desc: str,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_loss = 0.0
    total_samples = 0
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    iterator = loader if max_batches is None else islice(loader, max_batches)
    for images, labels, _paths in tqdm(iterator, total=total_batches, desc=desc):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        logits = outputs["logits_fine"]
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        for true_label, pred_label in zip(labels.cpu(), predictions.cpu()):
            confusion[int(true_label), int(pred_label)] += 1
        batch_size = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    return metrics_from_confusion(confusion, total_loss / max(1, total_samples))


def metrics_from_confusion(confusion: torch.Tensor, loss: float) -> dict[str, Any]:
    matrix = confusion.cpu().numpy().astype(np.int64)
    support = matrix.sum(axis=1)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    accuracy = correct / total if total else 0.0

    f1_scores = []
    class_metrics: dict[str, dict[str, float | int]] = {}
    for index in range(matrix.shape[0]):
        tp = int(matrix[index, index])
        predicted_total = int(matrix[:, index].sum())
        actual_total = int(support[index])
        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / actual_total if actual_total else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        class_metrics[str(index)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_total,
        }

    mae_numerator = 0
    for true_index in range(matrix.shape[0]):
        for pred_index in range(matrix.shape[1]):
            mae_numerator += abs(true_index - pred_index) * int(matrix[true_index, pred_index])
    mae = mae_numerator / total if total else 0.0
    qwk = quadratic_weighted_kappa(matrix)
    adjacent_errors = 0
    distant_errors = 0
    for true_index in range(matrix.shape[0]):
        for pred_index in range(matrix.shape[1]):
            if true_index == pred_index:
                continue
            count = int(matrix[true_index, pred_index])
            if abs(true_index - pred_index) == 1:
                adjacent_errors += count
            else:
                distant_errors += count

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "mae": float(mae),
        "qwk": float(qwk),
        "confusion_matrix": matrix.tolist(),
        "class_metrics": class_metrics,
        "adjacent_error_count": int(adjacent_errors),
        "distant_error_count": int(distant_errors),
    }


def quadratic_weighted_kappa(confusion: np.ndarray) -> float:
    num_classes = confusion.shape[0]
    total = confusion.sum()
    if total == 0:
        return 0.0
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    actual_hist = confusion.sum(axis=1)
    pred_hist = confusion.sum(axis=0)
    expected = np.outer(actual_hist, pred_hist) / total
    observed_score = float((weights * confusion).sum())
    expected_score = float((weights * expected).sum())
    if expected_score == 0:
        return 1.0 if observed_score == 0 else 0.0
    return 1.0 - observed_score / expected_score


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_run_dir(config: dict[str, Any]) -> Path:
    runs_root = resolve_project_path(config.get("runs_root", THIS_DIR / "runs_BaSic"))
    name = str(config.get("experiment_name", "finevad_mixnet_s_progressive"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = runs_root / f"{name}_{timestamp}"
    suffix = 2
    while candidate.exists():
        candidate = runs_root / f"{name}_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def stage_specs(config: dict[str, Any], args: argparse.Namespace) -> list[StageSpec]:
    train_cfg = config["train"]
    return [
        StageSpec(
            "stage1_coarse",
            int(args.epochs_stage1 if args.epochs_stage1 is not None else train_cfg.get("stage1_epochs", 40)),
            float(train_cfg.get("stage1_lr", 1e-4)),
        ),
        StageSpec(
            "stage2_mid_fusion",
            int(args.epochs_stage2 if args.epochs_stage2 is not None else train_cfg.get("stage2_epochs", 20)),
            float(train_cfg.get("stage2_lr", 3e-4)),
        ),
        StageSpec(
            "stage3_fine",
            int(args.epochs_stage3 if args.epochs_stage3 is not None else train_cfg.get("stage3_epochs", 90)),
            float(train_cfg.get("stage3_lr", 1e-4)),
        ),
    ]


def build_model(config: dict[str, Any], args: argparse.Namespace, device: torch.device) -> FineVADMixNet:
    model_cfg = config["model"]
    pretrained = bool(model_cfg.get("pretrained", True))
    if args.no_pretrained:
        pretrained = False
    model = FineVADMixNet(
        backbone_name=str(model_cfg.get("backbone_name", "mixnet_s")),
        pretrained=pretrained,
        num_classes=5,
        mid_classes=3,
        coarse_classes=2,
        dropout=float(model_cfg.get("dropout", 0.2)),
        projection_dim=int(model_cfg.get("projection_dim", 128)),
    )
    return model.to(device)


def summarize_data_loaders(loaders: tuple[DataLoader, DataLoader, DataLoader]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, loader in zip(("train", "val", "test"), loaders):
        dataset = loader.dataset
        image, label, path = dataset[0]
        summary[split_name] = {
            "total": len(dataset),
            "classes": list(dataset.classes),
            "class_to_idx": dict(dataset.class_to_idx),
            "counts": count_by_class(dataset),
            "sample_shape": list(image.shape),
            "sample_label": int(label),
            "sample_path": str(path),
        }
    return summary


def dry_run(
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
) -> None:
    summary = summarize_data_loaders(loaders)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    model = build_model(config, args, device)
    train_loader = loaders[0]
    images, labels, _paths = next(iter(train_loader))
    images = images[: min(2, images.shape[0])].to(device)
    with torch.no_grad():
        outputs = model(images)
    shapes = {key: list(value.shape) for key, value in outputs.items()}
    print(json.dumps({"device": str(device), "output_shapes": shapes}, ensure_ascii=False, indent=2))


def save_checkpoint(
    path: Path,
    model: FineVADMixNet,
    config: dict[str, Any],
    epoch_global: int,
    stage_name: str,
    val_metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": int(epoch_global),
            "stage": stage_name,
            "val_metrics": val_metrics,
        },
        path,
    )


def train(config: dict[str, Any], args: argparse.Namespace) -> Path:
    seed = int(config.get("random_seed", 2026))
    set_seed(seed)
    device = resolve_device(args.device)
    loaders = build_loaders(config, args)
    if args.dry_run:
        dry_run(config, args, device, loaders)
        return THIS_DIR

    run_dir = make_run_dir(config)
    save_json(run_dir / "resolved_config.json", config)
    save_json(run_dir / "dataset_summary.json", summarize_data_loaders(loaders))

    model = build_model(config, args, device)
    loss_cfg = config["loss"]
    loss_fn = ProgressiveFineVADLoss(
        coarse_map=[0, 0, 1, 1, 1],
        mid_map=[0, 0, 1, 1, 2],
        lambda_supcon=float(loss_cfg.get("lambda_supcon", 0.2)),
        supcon_temperature=float(loss_cfg.get("supcon_temperature", 0.1)),
        stage2_consistency_weight=float(loss_cfg.get("stage2_consistency_weight", 1.0)),
        stage2_entropy_weight=float(loss_cfg.get("stage2_entropy_weight", 0.01)),
        sibling_penalty_weight=float(loss_cfg.get("sibling_penalty_weight", 2.0)),
        stage3_aux_coarse_weight=float(loss_cfg.get("stage3_aux_coarse_weight", 0.1)),
        stage3_aux_mid_weight=float(loss_cfg.get("stage3_aux_mid_weight", 0.05)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
    ).to(device)

    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(device, enabled=use_amp)
    weight_decay = float(config["train"].get("weight_decay", 5e-4))
    min_lr = float(config["train"].get("min_lr", 1e-6))
    patience = int(config["train"].get("patience", 30))
    max_train_batches = int(args.max_train_batches) if args.max_train_batches is not None else None
    max_eval_batches = int(args.max_eval_batches) if args.max_eval_batches is not None else None

    best_val_acc = -math.inf
    best_epoch = -1
    epochs_since_improvement = 0
    global_epoch = 0
    history: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    for spec in stage_specs(config, args):
        if spec.epochs <= 0:
            continue
        configure_stage_trainable(model, spec.name)
        optimizer = build_optimizer(model, spec.lr, weight_decay)
        scheduler = build_scheduler(optimizer, spec.epochs, min_lr)
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"\n== {spec.name}: epochs={spec.epochs}, lr={spec.lr}, trainable_params={trainable_count:,}")
        for local_epoch in range(spec.epochs):
            train_metrics = train_one_epoch(
                model=model,
                loss_fn=loss_fn,
                loader=loaders[0],
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                stage_name=spec.name,
                epoch_index=local_epoch,
                stage_epochs=spec.epochs,
                use_amp=use_amp,
                max_train_batches=max_train_batches,
            )
            val_metrics = evaluate(
                model,
                loaders[1],
                device,
                num_classes=5,
                desc="valid",
                max_batches=max_eval_batches,
            )
            if train_metrics.get("optimizer_steps", 0) > 0:
                scheduler.step()
            row = {
                "global_epoch": global_epoch,
                "stage": spec.name,
                "stage_epoch": local_epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items() if key != "confusion_matrix"},
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            save_history_csv(run_dir / "history.csv", history)
            save_json(run_dir / "latest_val_metrics.json", val_metrics)
            print(
                f"epoch={global_epoch:03d} stage={spec.name} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_mae={val_metrics['mae']:.4f}"
            )
            is_fine_selection_stage = spec.name == "stage3_fine"
            if is_fine_selection_stage and val_metrics["accuracy"] > best_val_acc:
                best_val_acc = float(val_metrics["accuracy"])
                best_epoch = global_epoch
                epochs_since_improvement = 0
                save_checkpoint(run_dir / "best_model.pth", model, config, global_epoch, spec.name, val_metrics)
                save_json(run_dir / "best_val_metrics.json", val_metrics)
            elif is_fine_selection_stage:
                epochs_since_improvement += 1
            global_epoch += 1
            if patience > 0 and spec.name == "stage3_fine" and epochs_since_improvement >= patience:
                print(f"Early stopping in stage3 after {patience} epochs without val_acc improvement.")
                break

    best_model_path = run_dir / "best_model.pth"
    if not best_model_path.exists():
        save_checkpoint(
            best_model_path,
            model,
            config,
            max(0, global_epoch - 1),
            "final_without_stage3_selection",
            {"accuracy": None},
        )
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        loaders[2],
        device,
        num_classes=5,
        desc="test",
        max_batches=max_eval_batches,
    )
    elapsed = time.perf_counter() - started_at
    test_metrics.update({"best_epoch": best_epoch, "best_val_acc": best_val_acc, "train_seconds": elapsed})
    save_json(run_dir / "test_metrics.json", test_metrics)
    keep_best = bool(config["train"].get("keep_best_pth", True))
    if not keep_best:
        (run_dir / "best_model.pth").unlink(missing_ok=True)
    print(f"\nDone. run_dir={run_dir}")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    return run_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FineVAD-style progressive MixNet-S experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs-stage1", type=int, default=None)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    parser.add_argument("--epochs-stage3", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = load_config(resolve_project_path(args.config))
    train(config, args)


if __name__ == "__main__":
    main()
