"""诊断实验共用组件。

本目录不注册主工程组件，也不修改主线代码。实验遵循两个原则：
1. 一次只改变一个因素；2. 同时报告 patch 级和原图聚合级结果。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CLASSES = ("00", "10", "20", "30", "40")


def set_random_seed(seed: int) -> None:
    """统一设置各随机源，保证条件间对比可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    """解析 auto/cpu/cuda 设备。"""

    if name.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 检测不到可用 GPU。")
    return torch.device(name)


def parse_int_list(text: str) -> list[int]:
    """解析逗号分隔的正整数。"""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"需要正整数列表，实际得到：{text!r}")
    return values


def parse_float_list(text: str) -> list[float]:
    """解析逗号分隔的浮点数。"""

    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("浮点数列表不能为空。")
    return values


def stable_seed(key: str, seed: int) -> int:
    """从路径生成跨进程稳定的种子；不能使用受进程影响的 hash()。"""

    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def infer_parent_id(path: Path, class_name: str) -> str:
    """把派生 patch 映射回原图 ID，防止伪重复样本抬高置信度。"""

    match = re.match(
        r"^t[^_]+__(.+?)__(?:random\d+|grid\d+)_\d+$",
        path.stem,
        flags=re.IGNORECASE,
    )
    stem = match.group(1) if match else path.stem
    return f"{class_name}/{stem}"


def _resize_short_side(image: Image.Image, size: int) -> Image.Image:
    """仅在短边不够裁剪时等比例放大。"""

    width, height = image.size
    if min(width, height) >= size:
        return image
    scale = size / min(width, height)
    return TF.resize(
        image,
        [max(size, round(height * scale)), max(size, round(width * scale))],
        antialias=True,
    )


def fixed_positions(width: int, height: int, size: int, count: int) -> list[tuple[int, int]]:
    """生成中心、四角及均匀网格位置，供确定性多裁剪评估。"""

    max_left, max_top = max(0, width - size), max(0, height - size)
    center = (max_top // 2, max_left // 2)
    if count == 1:
        return [center]
    candidates = [(0, 0), (0, max_left), (max_top, 0), (max_top, max_left), center]
    positions: list[tuple[int, int]] = []
    for position in candidates:
        if position not in positions:
            positions.append(position)
        if len(positions) == count:
            return positions
    grid = max(2, math.ceil(math.sqrt(count)))
    for row in range(grid):
        top = round(max_top * row / (grid - 1))
        for column in range(grid):
            left = round(max_left * column / (grid - 1))
            if (top, left) not in positions:
                positions.append((top, left))
            if len(positions) == count:
                return positions
    return positions


def shuffle_tiles(image: Image.Image, grid: int, rng: random.Random) -> Image.Image:
    """打乱等尺寸块；像素内容保留，主要破坏空间关系。"""

    if grid < 2:
        return image.copy()
    width, height = image.size
    tile_width, tile_height = width // grid, height // grid
    if min(tile_width, tile_height) <= 0:
        return image.copy()
    tiles = [
        image.crop(
            (
                column * tile_width,
                row * tile_height,
                (column + 1) * tile_width,
                (row + 1) * tile_height,
            )
        )
        for row in range(grid)
        for column in range(grid)
    ]
    rng.shuffle(tiles)
    result = image.copy()
    for index, tile in enumerate(tiles):
        row, column = divmod(index, grid)
        result.paste(tile, (column * tile_width, row * tile_height))
    return result


def texture_shuffle(
    image: Image.Image,
    macro_grid: int,
    micro_grid: int,
    rng: random.Random,
) -> Image.Image:
    """只在每个粗区域内部打乱细块，尽量保留粗布局并破坏纹理组织。"""

    result = image.copy()
    width, height = image.size
    region_width, region_height = width // macro_grid, height // macro_grid
    if min(region_width, region_height) <= 0:
        return result
    for row in range(macro_grid):
        for column in range(macro_grid):
            left, top = column * region_width, row * region_height
            box = (left, top, left + region_width, top + region_height)
            result.paste(shuffle_tiles(image.crop(box), micro_grid, rng), (left, top))
    return result


@dataclass(frozen=True)
class PerturbationSpec:
    """一个受控破坏条件。"""

    name: str = "original"
    value: float = 0.0
    grid: int = 3
    micro_grid: int = 4
    repeat: int = 0


def apply_perturbation(
    image: Image.Image,
    spec: PerturbationSpec,
    sample_key: str,
    seed: int,
) -> Image.Image:
    """施加确定性破坏；同一路径、条件、重复编号的结果恒定。"""

    name = spec.name.lower()
    if name in {"none", "original"}:
        return image.copy()
    rng = random.Random(stable_seed(f"{sample_key}:{spec.repeat}:{name}", seed))
    if name == "grayscale":
        return TF.to_grayscale(image, num_output_channels=3)
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=spec.value))
    if name == "patch_shuffle":
        return shuffle_tiles(image, spec.grid, rng)
    if name == "texture_shuffle":
        return texture_shuffle(image, spec.grid, spec.micro_grid, rng)
    if name == "color_jitter":
        strength = spec.value
        changed = TF.adjust_brightness(image, rng.uniform(1 - strength, 1 + strength))
        changed = TF.adjust_contrast(changed, rng.uniform(1 - strength, 1 + strength))
        changed = TF.adjust_saturation(changed, rng.uniform(1 - strength, 1 + strength))
        hue = min(0.1, strength / 4)
        return TF.adjust_hue(changed, rng.uniform(-hue, hue))
    if name == "occlusion":
        if not 0 < spec.value < 1:
            raise ValueError(f"遮挡比例必须在 (0, 1)，实际为 {spec.value}。")
        width, height = image.size
        box_width = min(width, max(1, round(width * math.sqrt(spec.value))))
        box_height = min(height, max(1, round(height * math.sqrt(spec.value))))
        left = rng.randint(0, max(0, width - box_width))
        top = rng.randint(0, max(0, height - box_height))
        mean_color = tuple(int(value) for value in np.asarray(image).mean(axis=(0, 1)))
        result = image.copy()
        result.paste(mean_color, (left, top, left + box_width, top + box_height))
        return result
    raise ValueError(f"未知扰动：{spec.name}")


class DiagnosticTransform:
    """统一实现物理视野裁剪、模型 resize 与受控扰动。"""

    def __init__(
        self,
        crop_size: int,
        input_size: int = 224,
        training: bool = False,
        view_count: int = 1,
        perturbation: PerturbationSpec | None = None,
        seed: int = 2026,
        full_image: bool = False,
    ) -> None:
        self.crop_size = int(crop_size)
        self.input_size = int(input_size)
        self.training = bool(training)
        self.view_count = int(view_count)
        self.perturbation = perturbation or PerturbationSpec()
        self.seed = int(seed)
        self.full_image = bool(full_image)
        if min(self.crop_size, self.input_size, self.view_count) <= 0:
            raise ValueError("crop_size、input_size、view_count 必须为正数。")
        if self.training and self.view_count != 1:
            raise ValueError("训练阶段只允许一个随机视图。")

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        image = TF.resize(image, [self.input_size, self.input_size], antialias=True)
        return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, image: Image.Image, sample_key: str) -> torch.Tensor:
        if self.full_image:
            # Global 条件保留整张原图内容，只统一 resize 到模型输入尺寸。
            if self.training:
                if torch.rand(()) < 0.5:
                    image = TF.hflip(image)
                if torch.rand(()) < 0.5:
                    image = TF.vflip(image)
            # 诊断扰动统一在模型实际接收的 input_size 画布上发生，
            # 避免 Global 条件先在超大原图上模糊/打乱、再缩小导致扰动强度不可比。
            image = TF.resize(image, [self.input_size, self.input_size], antialias=True)
            image = apply_perturbation(image, self.perturbation, sample_key, self.seed)
            return self._tensor(image)

        image = _resize_short_side(image, self.crop_size)
        if self.training:
            top, left, height, width = RandomCrop.get_params(
                image,
                (self.crop_size, self.crop_size),
            )
            crop = TF.crop(image, top, left, height, width)
            if torch.rand(()) < 0.5:
                crop = TF.hflip(crop)
            if torch.rand(()) < 0.5:
                crop = TF.vflip(crop)
            crop = apply_perturbation(crop, self.perturbation, sample_key, self.seed)
            return self._tensor(crop)

        width, height = image.size
        views = []
        for index, (top, left) in enumerate(
            fixed_positions(width, height, self.crop_size, self.view_count)
        ):
            crop = TF.crop(image, top, left, self.crop_size, self.crop_size)
            crop = TF.resize(crop, [self.input_size, self.input_size], antialias=True)
            crop = apply_perturbation(
                crop,
                self.perturbation,
                f"{sample_key}:view{index}",
                self.seed,
            )
            views.append(self._tensor(crop))
        return views[0] if len(views) == 1 else torch.stack(views)


class DiagnosticImageDataset(Dataset):
    """读取 root/split/class 图片，同时记录每个 patch 对应的原图。"""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: DiagnosticTransform,
        class_names: Sequence[str] = DEFAULT_CLASSES,
        max_samples_per_class: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.classes = [str(name) for name in class_names]
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"数据目录不存在：{split_dir}")
        self.samples: list[tuple[Path, int]] = []
        self.parent_ids: list[str] = []
        for label, class_name in enumerate(self.classes):
            class_dir = split_dir / class_name
            paths = sorted(
                path.resolve()
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if max_samples_per_class is not None:
                paths = paths[:max_samples_per_class]
            if not paths:
                raise ValueError(f"类别没有图片：{class_dir}")
            self.samples.extend((path, label) for path in paths)
            self.parent_ids.extend(infer_parent_id(path, class_name) for path in paths)
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        return self.transform(image, str(path)), label, index


class RepeatedDataset(Dataset):
    """每轮从每张原图随机取多个 crop，不在磁盘复制图片。"""

    def __init__(self, dataset: Dataset, repeats: int) -> None:
        self.dataset, self.repeats = dataset, int(repeats)
        if self.repeats <= 0:
            raise ValueError("repeats 必须为正数。")

    def __len__(self) -> int:
        return len(self.dataset) * self.repeats

    def __getitem__(self, index: int):
        return self.dataset[index % len(self.dataset)]


def _seed_worker(worker_id: int) -> None:
    """固定 DataLoader 子进程随机源。"""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """构建可复现的数据加载器。"""

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def create_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """用 timm 创建分类模型，默认使用 MixNet-S。"""

    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("诊断实验需要 timm。") from exc
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def forward_probabilities(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """普通输入直接推理，多视图输入在概率空间取平均。"""

    if images.ndim == 4:
        return F.softmax(model(images), dim=1)
    if images.ndim != 5:
        raise ValueError(f"输入应为 4D/5D，实际为 {tuple(images.shape)}。")
    batch, views, channels, height, width = images.shape
    probabilities = F.softmax(model(images.reshape(-1, channels, height, width)), dim=1)
    return probabilities.view(batch, views, -1).mean(dim=1)


def _macro_f1(labels: np.ndarray, predictions: np.ndarray, classes: int) -> float:
    values = []
    for index in range(classes):
        tp = int(((labels == index) & (predictions == index)).sum())
        fp = int(((labels != index) & (predictions == index)).sum())
        fn = int(((labels == index) & (predictions != index)).sum())
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    return float(np.mean(values))


def _qwk(labels: np.ndarray, predictions: np.ndarray, classes: int) -> float:
    matrix = np.zeros((classes, classes), dtype=np.float64)
    for label, prediction in zip(labels, predictions):
        matrix[label, prediction] += 1
    total = matrix.sum()
    if total <= 0 or classes <= 1:
        return 0.0
    expected = np.outer(matrix.sum(1), matrix.sum(0)) / total
    axis = np.arange(classes)
    weights = (axis[:, None] - axis[None, :]) ** 2 / (classes - 1) ** 2
    observed_score = float((weights * matrix).sum() / total)
    expected_score = float((weights * expected).sum() / total)
    if expected_score <= 1e-12:
        return 1.0 if observed_score <= 1e-12 else 0.0
    return 1 - observed_score / expected_score


def _wilson_interval(correct: int, total: int) -> tuple[float, float]:
    """计算准确率 Wilson 95% 区间，提醒使用者独立原图样本量有限。"""

    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """计算准确率、F1、有序指标和混淆矩阵。"""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(1)
    errors = np.abs(labels - predictions)
    classes = len(class_names)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        matrix[label, prediction] += 1
    wrong_count = int((errors > 0).sum())
    correct_count = int((errors == 0).sum())
    ci_low, ci_high = _wilson_interval(correct_count, len(labels))
    return {
        "sample_count": len(labels),
        "accuracy": float((errors == 0).mean()),
        "accuracy_wilson95_low": ci_low,
        "accuracy_wilson95_high": ci_high,
        "macro_f1": _macro_f1(labels, predictions, classes),
        "mae": float(errors.mean()),
        "qwk": _qwk(labels, predictions, classes),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "mean_confidence": float(probabilities.max(1).mean()),
        "error_count": wrong_count,
        "adjacent_error_fraction": float((errors == 1).sum() / wrong_count) if wrong_count else 0.0,
        "far_error_fraction": float((errors > 1).sum() / wrong_count) if wrong_count else 0.0,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(class_names),
        "per_class_recall": {
            name: float(matrix[index, index] / max(1, matrix[index].sum()))
            for index, name in enumerate(class_names)
        },
    }


def aggregate_by_parent(
    dataset: DiagnosticImageDataset,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """同一原图的 patch 概率取平均，形成真正独立的原图级样本。"""

    groups: dict[str, list[int]] = defaultdict(list)
    for index, parent_id in enumerate(dataset.parent_ids):
        groups[parent_id].append(index)
    parent_labels, parent_probabilities, rows = [], [], []
    for parent_id in sorted(groups):
        indices = groups[parent_id]
        if len(set(labels[indices].tolist())) != 1:
            raise ValueError(f"原图标签不一致：{parent_id}")
        label = int(labels[indices[0]])
        probability = probabilities[indices].mean(0)
        parent_labels.append(label)
        parent_probabilities.append(probability)
        row = {
            "parent_id": parent_id,
            "label": label,
            "prediction": int(probability.argmax()),
            "correct": int(label == int(probability.argmax())),
            "patch_count": len(indices),
            "confidence": float(probability.max()),
        }
        row.update({f"prob_{name}": float(probability[i]) for i, name in enumerate(dataset.classes)})
        rows.append(row)
    return np.asarray(parent_labels), np.vstack(parent_probabilities), rows


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    dataset: DiagnosticImageDataset,
    device: torch.device,
) -> dict[str, Any]:
    """评估并返回 patch/原图两级指标和逐样本预测。"""

    model.eval()
    labels_list, probabilities_list, indices_list = [], [], []
    for images, labels, indices in loader:
        probabilities = forward_probabilities(model, images.to(device, non_blocking=True))
        labels_list.append(labels.numpy())
        probabilities_list.append(probabilities.cpu().numpy())
        indices_list.append(indices.numpy())
    labels = np.concatenate(labels_list)
    probabilities = np.concatenate(probabilities_list)
    order = np.argsort(np.concatenate(indices_list))
    labels, probabilities = labels[order], probabilities[order]
    predictions = probabilities.argmax(1)
    sample_rows = []
    for index, ((path, _), label, prediction, probability) in enumerate(
        zip(dataset.samples, labels, predictions, probabilities)
    ):
        row = {
            "path": str(path),
            "parent_id": dataset.parent_ids[index],
            "label": int(label),
            "prediction": int(prediction),
            "confidence": float(probability.max()),
        }
        row.update({f"prob_{name}": float(probability[i]) for i, name in enumerate(dataset.classes)})
        sample_rows.append(row)
    parent_labels, parent_probabilities, parent_rows = aggregate_by_parent(
        dataset,
        labels,
        probabilities,
    )
    return {
        "sample": classification_metrics(labels, probabilities, dataset.classes),
        "parent": classification_metrics(parent_labels, parent_probabilities, dataset.classes),
        "sample_predictions": sample_rows,
        "parent_predictions": parent_rows,
    }


@dataclass
class TrainingOptions:
    """单尺度训练参数。"""

    epochs: int = 150
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    patience: int = 30
    label_smoothing: float = 0.0
    amp: bool = True


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_dataset: DiagnosticImageDataset,
    device: torch.device,
    output_dir: Path,
    options: TrainingOptions,
    metadata: dict[str, Any],
) -> Path:
    """训练模型并用验证集原图聚合 accuracy 早停和选模。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, options.epochs),
        eta_min=min(1e-6, options.learning_rate),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=options.label_smoothing)
    amp_enabled = options.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_path = output_dir / "best.pth"
    best_accuracy, stale_epochs, history = -math.inf, 0, []
    for epoch in range(1, options.epochs + 1):
        model.train()
        loss_sum = correct = count = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with context:
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            count += len(labels)
        scheduler.step()
        validation = evaluate_model(model, val_loader, val_dataset, device)
        # 默认数据一行就是一张原图；若换成派生 patch 数据，这里仍按原图聚合选模，
        # 避免拥有更多相关 patch 的原图在 early stopping 中被重复加权。
        accuracy = validation["parent"]["accuracy"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / count,
                "train_accuracy": correct / count,
                "val_sample_accuracy": validation["sample"]["accuracy"],
                "val_parent_accuracy": accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={loss_sum / count:.4f} "
            f"train_acc={correct / count:.4f} "
            f"val_patch_acc={validation['sample']['accuracy']:.4f} "
            f"val_parent_acc={accuracy:.4f}"
        )
        if accuracy > best_accuracy:
            best_accuracy, stale_epochs = accuracy, 0
            torch.save(
                {"model_state_dict": model.state_dict(), "metadata": metadata, "epoch": epoch},
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= options.patience:
                print(f"连续 {stale_epochs} 轮未提升，提前停止。")
                break
    write_csv(output_dir / "history.csv", history)
    return best_path


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    model_name: str | None = None,
    num_classes: int | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """加载本工具 checkpoint；旧权重可显式补充模型名和类别数。"""

    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    metadata = dict(checkpoint.get("metadata") or {})
    resolved_name = model_name or metadata.get("model_name")
    resolved_classes = num_classes or metadata.get("num_classes")
    if not resolved_name or not resolved_classes:
        raise ValueError("checkpoint 缺模型元数据，请传 --model-name 和 --num-classes。")
    model = create_model(resolved_name, int(resolved_classes), pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), metadata


def write_json(path: str | Path, value: Any) -> None:
    """保存 UTF-8 JSON。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """保存 UTF-8 BOM CSV，便于 Excel 直接查看中文。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metrics_row(condition: str, result: dict[str, Any]) -> dict[str, Any]:
    """将两级指标压成汇总表的一行。"""

    row: dict[str, Any] = {"condition": condition}
    keys = (
        "sample_count",
        "accuracy",
        "accuracy_wilson95_low",
        "accuracy_wilson95_high",
        "macro_f1",
        "mae",
        "qwk",
        "nll",
        "mean_confidence",
        "adjacent_error_fraction",
        "far_error_fraction",
    )
    for level in ("sample", "parent"):
        row.update({f"{level}_{key}": result[level][key] for key in keys})
    return row
