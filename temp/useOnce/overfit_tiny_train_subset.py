"""在 datasets_01234 的训练集中抽取少量图片，做“小样本过拟合”诊断。

这个脚本的目标不是验证泛化能力，而是确认模型、标签、损失函数和训练循环是否正常：
如果只给模型 20~40 张训练图，并且关闭大部分正则化，训练准确率应该能接近 100%，
训练损失也应该明显接近 0。若做不到，通常说明数据标签、类别映射、模型输出维度、
loss 使用方式或优化流程里存在更基础的问题。

使用方式：
1. 直接在 PyCharm 中右键运行本文件。
2. 如需改样本数、学习率、模型或 epoch，只修改下面“可直接修改的配置区”。
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# =============================================================================
# 可直接修改的配置区：你之后基本只需要改这里。
# =============================================================================

# 项目根目录会通过脚本位置自动推断；数据集路径保持相对路径，方便本机和服务器同步使用。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path("datasets_01234")
TRAIN_SPLIT_NAME = "train"

# 从训练集中抽多少张图做过拟合测试；建议保持在 20~40，默认使用 32 张。
SUBSET_SIZE = 32

# 是否按类别尽量均衡抽样；五分类时 32 张大约会抽成 7/7/6/6/6。
BALANCED_SAMPLING = True

# 训练基础设置：这是“小样本背题”模式，不是正式泛化实验。
MODEL_NAME = "efficientnet_v2_s"  # 可选：resnet18 / efficientnet_v2_s / mobilenet_v3_large / convnext_tiny
USE_IMAGENET_PRETRAINED = True
NUM_CLASSES = 5
IMAGE_SIZE = 224
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 3e-4  # 如果不稳定，可以改成 1e-4。
WEIGHT_DECAY = 0.0
DROP_RATE = 0.0
SEED = 2026

# Windows/PyCharm 下 num_workers=0 最稳，避免多进程反复导入脚本。
NUM_WORKERS = 0

# auto 表示有 CUDA 就用 CUDA，没有就用 CPU；服务器 4090 上会自动使用 CUDA。
DEVICE_NAME = "auto"  # 可选：auto / cuda / cpu

# 只做训练集观察；达到连续若干轮几乎背下来的条件后，可提前停止节省时间。
STOP_WHEN_MEMORIZED = True
MEMORIZED_ACC = 0.995
MEMORIZED_LOSS = 0.02
MEMORIZED_PATIENCE = 5

# 输出目录同样使用相对路径；每次运行会自动创建时间戳子文件夹。
OUTPUT_ROOT = Path("temp/useOnce/overfit_tiny_subset_runs")

# 这个诊断通常不需要保存模型权重，默认关闭，避免额外占用硬盘。
SAVE_LAST_MODEL = False


# =============================================================================
# 数据集与工具函数。
# =============================================================================

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def resolve_project_path(relative_path: Path) -> Path:
    """把相对路径解析到项目根目录下，避免 PyCharm 工作目录不同导致找不到文件。"""
    return relative_path if relative_path.is_absolute() else PROJECT_ROOT / relative_path


def set_random_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机种子，让抽样和训练尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str) -> torch.device:
    """根据配置选择训练设备；auto 会优先使用 CUDA。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用 cuda，但当前环境没有可用 CUDA。")
    return torch.device(device_name)


def list_train_images(train_root: Path) -> Tuple[List[str], List[Tuple[Path, int, str]]]:
    """扫描 train/类别名 目录，返回类别列表和所有图片路径。"""
    if not train_root.is_dir():
        raise FileNotFoundError(f"训练集目录不存在：{train_root}")

    class_names = [path.name for path in sorted(train_root.iterdir()) if path.is_dir()]
    if not class_names:
        raise RuntimeError(f"训练集目录下没有类别子目录：{train_root}")

    samples: List[Tuple[Path, int, str]] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = train_root / class_name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((image_path, class_index, class_name))

    if not samples:
        raise RuntimeError(f"训练集中没有找到图片文件：{train_root}")
    return class_names, samples


def balanced_subset(
    samples: Sequence[Tuple[Path, int, str]],
    subset_size: int,
    seed: int,
) -> List[Tuple[Path, int, str]]:
    """按类别尽量均衡地抽取少量图片，避免 32 张刚好偏到某几个类别。"""
    rng = random.Random(seed)
    by_class: Dict[int, List[Tuple[Path, int, str]]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    for class_samples in by_class.values():
        rng.shuffle(class_samples)

    class_ids = sorted(by_class)
    selected: List[Tuple[Path, int, str]] = []
    cursor = {class_id: 0 for class_id in class_ids}

    while len(selected) < subset_size:
        progressed = False
        for class_id in class_ids:
            if len(selected) >= subset_size:
                break
            index = cursor[class_id]
            if index < len(by_class[class_id]):
                selected.append(by_class[class_id][index])
                cursor[class_id] += 1
                progressed = True
        if not progressed:
            break

    rng.shuffle(selected)
    return selected


def random_subset(
    samples: Sequence[Tuple[Path, int, str]],
    subset_size: int,
    seed: int,
) -> List[Tuple[Path, int, str]]:
    """从全部训练样本里随机抽取少量图片。"""
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    return shuffled[:subset_size]


class TinyImageDataset(Dataset):
    """只包含几十张图片的小数据集，用于测试模型是否能过拟合训练样本。"""

    def __init__(
        self,
        samples: Sequence[Tuple[Path, int, str]],
        transform: transforms.Compose,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        """返回小数据集中的图片数量。"""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """读取单张图片并返回 tensor 和整数标签。"""
        image_path, label, _class_name = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def build_no_augmentation_transform(image_size: int) -> transforms.Compose:
    """构建无随机增强的 transform，只做尺寸统一、张量转换和 ImageNet 标准化。"""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# =============================================================================
# 模型构建函数。
# =============================================================================


def get_torchvision_weights(model_name: str, use_pretrained: bool):
    """根据模型名返回 torchvision 的 ImageNet 预训练权重；关闭预训练时返回 None。"""
    if not use_pretrained:
        return None
    if model_name == "resnet18":
        return models.ResNet18_Weights.DEFAULT
    if model_name == "efficientnet_v2_s":
        return models.EfficientNet_V2_S_Weights.DEFAULT
    if model_name == "mobilenet_v3_large":
        return models.MobileNet_V3_Large_Weights.DEFAULT
    if model_name == "convnext_tiny":
        return models.ConvNeXt_Tiny_Weights.DEFAULT
    raise ValueError(f"暂不支持的模型：{model_name}")


def build_model(model_name: str, num_classes: int, use_pretrained: bool) -> nn.Module:
    """创建 torchvision 分类模型，并把最后分类层替换成当前五分类输出。"""
    try:
        weights = get_torchvision_weights(model_name, use_pretrained)
        print(f"使用模型：{model_name}, pretrained={use_pretrained}")
    except Exception as error:
        raise RuntimeError(f"获取预训练权重配置失败：{error}") from error

    try:
        if model_name == "resnet18":
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(nn.Dropout(p=DROP_RATE), nn.Linear(in_features, num_classes))
        elif model_name == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif model_name == "convnext_tiny":
            model = models.convnext_tiny(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"暂不支持的模型：{model_name}")
    except Exception as error:
        if use_pretrained:
            print(f"加载预训练权重失败，将自动退回随机初始化。原因：{error}")
            return build_model(model_name, num_classes, use_pretrained=False)
        raise

    disable_dropout(model)
    return model


def disable_dropout(model: nn.Module) -> None:
    """把模型中所有 Dropout 的概率设为 0，减少正则化对背题测试的干扰。"""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = DROP_RATE


# =============================================================================
# 训练、记录与输出。
# =============================================================================


def make_run_directory() -> Path:
    """为本次小样本过拟合测试创建一个带时间戳的输出目录。"""
    output_root = resolve_project_path(OUTPUT_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{MODEL_NAME}_subset{SUBSET_SIZE}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_subset_manifest(
    run_dir: Path,
    class_names: Sequence[str],
    selected_samples: Sequence[Tuple[Path, int, str]],
) -> None:
    """保存本次抽到的图片清单，方便复查到底让模型背了哪些图。"""
    manifest_path = run_dir / "selected_samples.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label_index", "class_name", "relative_path"])
        for index, (image_path, label, class_name) in enumerate(selected_samples):
            writer.writerow([index, label, class_name, image_path.relative_to(PROJECT_ROOT).as_posix()])

    meta = {
        "class_names": list(class_names),
        "subset_size": len(selected_samples),
        "class_counts": dict(Counter(class_name for _path, _label, class_name in selected_samples)),
        "model_name": MODEL_NAME,
        "use_imagenet_pretrained": USE_IMAGENET_PRETRAINED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "drop_rate": DROP_RATE,
        "seed": SEED,
        "save_last_model": SAVE_LAST_MODEL,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_history_csv(run_dir: Path, history: Sequence[Dict[str, float]]) -> None:
    """保存每个 epoch 的训练 loss 和 accuracy。"""
    history_path = run_dir / "train_history.csv"
    with history_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "train_acc", "seconds"])
        writer.writeheader()
        writer.writerows(history)


def save_training_curve(run_dir: Path, history: Sequence[Dict[str, float]]) -> None:
    """把训练损失和训练准确率画成 PNG，便于直观看是否背下来了。"""
    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]
    accs = [row["train_acc"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].plot(epochs, losses, marker="o", linewidth=1.5)
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, accs, marker="o", linewidth=1.5)
    axes[1].set_title("Train Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(run_dir / "overfit_training_curve.png")
    plt.close(fig)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """训练一个 epoch，并只统计训练集 loss 和 accuracy。"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    """执行小样本过拟合诊断。"""
    set_random_seed(SEED)
    device = resolve_device(DEVICE_NAME)
    train_root = resolve_project_path(DATASET_ROOT) / TRAIN_SPLIT_NAME
    run_dir = make_run_directory()

    class_names, all_samples = list_train_images(train_root)
    if len(class_names) != NUM_CLASSES:
        raise RuntimeError(f"类别数量不匹配：配置 NUM_CLASSES={NUM_CLASSES}，实际类别={class_names}")

    if BALANCED_SAMPLING:
        selected_samples = balanced_subset(all_samples, SUBSET_SIZE, SEED)
    else:
        selected_samples = random_subset(all_samples, SUBSET_SIZE, SEED)

    if len(selected_samples) < SUBSET_SIZE:
        raise RuntimeError(f"可用样本不足：期望 {SUBSET_SIZE}，实际 {len(selected_samples)}")

    transform = build_no_augmentation_transform(IMAGE_SIZE)
    dataset = TinyImageDataset(selected_samples, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(MODEL_NAME, NUM_CLASSES, USE_IMAGENET_PRETRAINED).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    save_subset_manifest(run_dir, class_names, selected_samples)


    print("=" * 80)
    print("小样本过拟合诊断开始")
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"训练集目录：{train_root}")
    print(f"输出目录：{run_dir}")
    print(f"设备：{device}")
    print(f"类别：{class_names}")
    print(f"抽样数量：{len(selected_samples)}")
    print(f"抽样类别分布：{dict(Counter(class_name for _path, _label, class_name in selected_samples))}")
    print("目标：训练准确率接近 100%，训练损失接近 0。")
    print("=" * 80)

    history: List[Dict[str, float]] = []
    memorized_streak = 0

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, loader, criterion, optimizer, device)
        seconds = time.time() - start_time

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "seconds": seconds,
            }
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"train_acc={train_acc:.4f} | "
            f"time={seconds:.1f}s"
        )

        if train_acc >= MEMORIZED_ACC and train_loss <= MEMORIZED_LOSS:
            memorized_streak += 1
        else:
            memorized_streak = 0

        if STOP_WHEN_MEMORIZED and memorized_streak >= MEMORIZED_PATIENCE:
            print(
                f"连续 {MEMORIZED_PATIENCE} 个 epoch 达到背题标准，提前停止。"
            )
            break

    save_history_csv(run_dir, history)
    save_training_curve(run_dir, history)
    if SAVE_LAST_MODEL:
        torch.save(model.state_dict(), run_dir / "overfit_last_model.pth")

    final = history[-1]
    print("=" * 80)
    print("小样本过拟合诊断结束")
    print(f"最终训练 loss：{final['train_loss']:.6f}")
    print(f"最终训练 acc ：{final['train_acc']:.4f}")
    print(f"训练日志：{run_dir / 'train_history.csv'}")
    print(f"训练曲线：{run_dir / 'overfit_training_curve.png'}")
    print(f"样本清单：{run_dir / 'selected_samples.csv'}")
    print(f"是否保存权重：{SAVE_LAST_MODEL}")
    print("判断：如果 acc 长期上不去或 loss 不明显下降，优先检查标签、类别映射和模型输出维度。")
    print("=" * 80)


if __name__ == "__main__":
    main()
