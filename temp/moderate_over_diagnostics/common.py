"""Shared utilities for Moderate-vs-Over diagnostic experiments."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
ORIGINAL_ROOT = PROJECT_ROOT / "datas_test_point"
PATCH_ROOT = PROJECT_ROOT / "datas_test_point_30_patches"
OUTPUT_ROOT = THIS_DIR / "outputs"

MODERATE_CODES = ("30", "35", "40", "45")
OVER_CODES = ("50", "55", "60")
TARGET_CODES = MODERATE_CODES + OVER_CODES
LABEL_BY_CODE = {code: 0 for code in MODERATE_CODES} | {code: 1 for code in OVER_CODES}
LABEL_NAME_BY_ID = {0: "moderate", 1: "over"}
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_run_dir(run_dir: Optional[str | Path] = None) -> Path:
    return ensure_dir(run_dir) if run_dir else ensure_dir(OUTPUT_ROOT / timestamp())


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 cuda，但当前环境没有可用 CUDA。")
    return torch.device(device)


def label_from_time_code(time_code: str) -> Tuple[int, str]:
    time_code = str(time_code).zfill(2)
    if time_code not in LABEL_BY_CODE:
        raise ValueError(f"时间点 {time_code} 不属于 Moderate/Over 二分类诊断范围。")
    label = LABEL_BY_CODE[time_code]
    return label, LABEL_NAME_BY_ID[label]


def list_image_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def image_to_tensor(image_path: str | Path, image_size: int = 224, variant: str = "rgb", augment: bool = False) -> torch.Tensor:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if variant == "gray":
        image = ImageOps.grayscale(image).convert("RGB")
    elif variant == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
    elif variant != "rgb":
        raise ValueError(f"未知图像输入变体：{variant}")
    if augment and random.random() < 0.5:
        image = ImageOps.mirror(image)
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class BinaryImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_column: str, image_size: int, variant: str = "rgb", augment: bool = False):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_column = image_column
        self.image_size = image_size
        self.variant = variant
        self.augment = augment

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        tensor = image_to_tensor(row[self.image_column], self.image_size, self.variant, self.augment)
        return tensor, int(row["label"]), index


def build_torchvision_binary_model(model_name: str = "resnet18", num_classes: int = 2, pretrained: str | bool = "auto") -> nn.Module:
    from torchvision import models

    name = model_name.lower()
    use_pretrained = str(pretrained).lower() not in {"0", "false", "none", "no", "random"}
    auto_fallback = str(pretrained).lower() == "auto"

    def build_with_weights(weights):
        if name == "resnet18":
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        if name == "resnet50":
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        if name == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        if name == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        raise ValueError(f"暂不支持的诊断模型：{model_name}")

    weight_attr = {
        "resnet18": "ResNet18_Weights",
        "resnet50": "ResNet50_Weights",
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
        "efficientnet_v2_s": "EfficientNet_V2_S_Weights",
    }.get(name)
    weights = getattr(models, weight_attr).DEFAULT if use_pretrained and weight_attr else None
    try:
        return build_with_weights(weights)
    except Exception:
        if not auto_fallback:
            raise
        print(f"[warn] {model_name} 预训练权重加载失败，自动退回随机初始化。")
        return build_with_weights(None)


def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    labels = [0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "class_metrics": {
            LABEL_NAME_BY_ID[label]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }


def add_per_time_accuracy(rows: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for time_code, group in rows.groupby("time_code"):
        records.append({
            "time_code": str(time_code).zfill(2),
            "count": int(len(group)),
            "accuracy": float((group["label"].astype(int) == group["pred_label"].astype(int)).mean()),
        })
    return sorted(records, key=lambda item: item["time_code"])


def save_confusion_matrix_png(confusion: Sequence[Sequence[int]], output_path: str | Path, title: str) -> None:
    matrix = np.asarray(confusion, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], ["moderate", "over"])
    ax.set_yticks([0, 1], ["moderate", "over"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_history_plot(history: List[Dict[str, float]], output_path: str | Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["train_acc"], label="train")
    axes[1].plot(frame["epoch"], frame["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def source_metadata_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "metadata" / "source_metadata.csv"


def patch_metadata_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "metadata" / "patch_metadata.csv"


def source_splits_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "splits" / "source_splits.csv"
