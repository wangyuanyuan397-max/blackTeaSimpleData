"""一次性检查 datasets_01234 的输入管线、Resize/RGB、张量范围和模型更新。

这个脚本对应下面几类常见训练故障：
1. 408x408 -> 224x224 缩放后颜色/纹理是否异常。
2. 是否存在 OpenCV BGR 与 PIL RGB 的通道顺序错误。
3. 一个 batch 进入模型前的 shape、dtype、数值范围、归一化是否正常。
4. 反归一化后，模型实际看到的图是否和肉眼看到的图一致。
5. logits、labels、CrossEntropyLoss 的使用是否正确。
6. 反向传播后，模型权重是否真的发生更新。

使用方式：直接在 PyCharm 右键运行本文件即可。脚本只做诊断，不会训练模型。
"""

from __future__ import annotations

import csv
import json
import random
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
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# =============================================================================
# 可直接修改的配置区。
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path("datasets_01234")
TRAIN_SPLIT_NAME = "train"
OUTPUT_ROOT = Path("temp/useOnce/input_pipeline_sanity_runs")

NUM_CLASSES = 5
IMAGE_SIZE = 224
BATCH_SIZE = 16
UPDATE_CHECK_BATCH_SIZE = 5
SEED = 2026

# 这里只检查一次前向/反向，不需要加载预训练权重，避免无网络环境下载失败。
MODEL_NAME = "efficientnet_v2_s"  # 可选：resnet18 / efficientnet_v2_s / mobilenet_v3_large / convnext_tiny
USE_PRETRAINED_FOR_UPDATE_CHECK = False
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
DEVICE_NAME = "auto"

# 保存多少张视觉核查图。
NUM_RESIZE_PREVIEW = 6
NUM_BATCH_PREVIEW = 16


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def resolve_project_path(path: Path) -> Path:
    """把相对路径解析到项目根目录下。"""
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_random_seed(seed: int) -> None:
    """固定随机种子，让抽样和检查结果稳定。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """选择检查设备；auto 会优先使用 CUDA。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用 cuda，但当前环境没有可用 CUDA。")
    return torch.device(device_name)


def build_eval_transform() -> transforms.Compose:
    """正式进入模型前的确定性 transform：Resize、ToTensor、ImageNet Normalize。"""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.flatten().tolist(), std=IMAGENET_STD.flatten().tolist()),
        ]
    )


class SimpleImageDataset(Dataset):
    """用于诊断的简化 ImageFolder 数据集，返回 image、label、path。"""

    def __init__(self, samples: Sequence[Tuple[Path, int, str]], transform: transforms.Compose) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        """返回样本数量。"""
        return len(self.samples)

    def __getitem__(self, index: int):
        """用 PIL 按 RGB 读取图像，然后应用 transform。"""
        image_path, label, _class_name = self.samples[index]
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        return self.transform(image), int(label), str(image_path)


def list_train_samples() -> Tuple[List[str], List[Tuple[Path, int, str]]]:
    """扫描 datasets_01234/train，得到类别名和图片路径。"""
    train_root = resolve_project_path(DATASET_ROOT) / TRAIN_SPLIT_NAME
    if not train_root.is_dir():
        raise FileNotFoundError(f"训练集目录不存在：{train_root}")

    class_names = [path.name for path in sorted(train_root.iterdir()) if path.is_dir()]
    samples: List[Tuple[Path, int, str]] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = train_root / class_name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((image_path, class_index, class_name))
    if not samples:
        raise RuntimeError(f"没有找到训练图片：{train_root}")
    return class_names, samples


def choose_balanced_samples(samples: Sequence[Tuple[Path, int, str]], count: int) -> List[Tuple[Path, int, str]]:
    """从每个类别轮流抽样，得到一个相对均衡的 batch。"""
    rng = random.Random(SEED)
    by_class: Dict[int, List[Tuple[Path, int, str]]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)
    for class_samples in by_class.values():
        rng.shuffle(class_samples)

    selected: List[Tuple[Path, int, str]] = []
    cursor = {class_id: 0 for class_id in by_class}
    while len(selected) < count:
        progressed = False
        for class_id in sorted(by_class):
            if len(selected) >= count:
                break
            index = cursor[class_id]
            if index < len(by_class[class_id]):
                selected.append(by_class[class_id][index])
                cursor[class_id] += 1
                progressed = True
        if not progressed:
            break
    return selected


def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """把 Normalize 后的 [3,H,W] tensor 反归一化成 RGB uint8 图像。"""
    cpu_tensor = tensor.detach().cpu()
    image = cpu_tensor * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def make_run_dir() -> Path:
    """创建本次诊断输出目录。"""
    output_root = resolve_project_path(OUTPUT_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"check_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def read_manifest_rows() -> Tuple[Path | None, List[dict[str, str]]]:
    """读取 datasets_01234 生成时保存的裁剪 manifest，用于复原 408->224 过程。"""
    dataset_root = resolve_project_path(DATASET_ROOT)
    manifest_path = dataset_root / "random_crop_manifest.csv"
    summary_path = dataset_root / "split_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return None, []

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_root = Path(summary["source_root"])
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return source_root, rows


def save_resize_rgb_preview(run_dir: Path) -> List[dict[str, float | str]]:
    """保存 408 crop、224 resize、已保存 jpg、反归一化 tensor 的对照图，并计算 RGB/BGR 差异。"""
    source_root, rows = read_manifest_rows()
    if source_root is None or not rows:
        print("未找到 random_crop_manifest.csv 或 split_summary.json，跳过 Resize/RGB 复原图。")
        return []

    preview_rows = [row for row in rows if row["split"] == TRAIN_SPLIT_NAME][:NUM_RESIZE_PREVIEW]
    fig, axes = plt.subplots(len(preview_rows), 4, figsize=(12, 3 * len(preview_rows)), dpi=150)
    if len(preview_rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    metrics: List[dict[str, float | str]] = []
    transform = build_eval_transform()
    dataset_root = resolve_project_path(DATASET_ROOT)

    for row_index, row in enumerate(preview_rows):
        source_path = source_root / row["source_relpath"]
        saved_patch_path = dataset_root / row["target_relpath"]
        left, top, right, bottom = (int(row[key]) for key in ("left", "top", "right", "bottom"))

        with Image.open(source_path) as opened_source:
            source_image = ImageOps.exif_transpose(opened_source).convert("RGB")
        crop_408 = source_image.crop((left, top, right, bottom))
        resized_224 = crop_408.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        with Image.open(saved_patch_path) as opened_patch:
            saved_224 = opened_patch.convert("RGB")

        tensor_seen = transform(saved_224)
        unnormalized = Image.fromarray(unnormalize_image(tensor_seen))

        rgb_array = np.asarray(resized_224).astype(np.float32)
        saved_array = np.asarray(saved_224).astype(np.float32)
        rgb_diff = float(np.abs(rgb_array - saved_array).mean())
        bgr_diff = float(np.abs(rgb_array[..., ::-1] - saved_array).mean())

        metrics.append(
            {
                "target_relpath": row["target_relpath"],
                "source_relpath": row["source_relpath"],
                "crop_size": f"{crop_408.size[0]}x{crop_408.size[1]}",
                "saved_size": f"{saved_224.size[0]}x{saved_224.size[1]}",
                "mean_abs_diff_rgb_pipeline": rgb_diff,
                "mean_abs_diff_if_bgr_swapped": bgr_diff,
                "rgb_channel_mean_saved": [float(x) for x in saved_array.mean(axis=(0, 1))],
            }
        )

        images = [crop_408, resized_224, saved_224, unnormalized]
        titles = ["408 crop", "PIL RGB resize", "saved jpg", "model sees"]
        for col_index, (image, title) in enumerate(zip(images, titles)):
            axes[row_index, col_index].imshow(image)
            axes[row_index, col_index].set_title(title)
            axes[row_index, col_index].axis("off")
        axes[row_index, 0].set_ylabel(row["target_relpath"], fontsize=8)

    fig.tight_layout()
    fig.savefig(run_dir / "resize_rgb_preview.png")
    plt.close(fig)
    return metrics


def save_batch_preview(run_dir: Path, images: torch.Tensor, labels: torch.Tensor, class_names: Sequence[str]) -> None:
    """把一个 batch 反归一化后保存成预览图。"""
    show_count = min(NUM_BATCH_PREVIEW, images.size(0))
    cols = 4
    rows = int(np.ceil(show_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=150)
    axes = np.asarray(axes).reshape(rows, cols)
    for index in range(rows * cols):
        ax = axes[index // cols, index % cols]
        ax.axis("off")
        if index < show_count:
            ax.imshow(unnormalize_image(images[index]))
            label_index = int(labels[index].item())
            ax.set_title(f"{label_index}: {class_names[label_index]}")
    fig.tight_layout()
    fig.savefig(run_dir / "batch_unnormalized_preview.png")
    plt.close(fig)


def get_torchvision_weights(model_name: str, use_pretrained: bool):
    """按模型名返回 torchvision 权重；诊断默认不加载预训练权重。"""
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
    raise ValueError(f"不支持的模型：{model_name}")


def disable_dropout(model: nn.Module) -> None:
    """诊断时关闭 Dropout，避免随机性干扰一次性权重更新检查。"""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = 0.0


def build_model() -> nn.Module:
    """构建一个五分类模型，用于检查 logits、loss、梯度和权重更新。"""
    weights = get_torchvision_weights(MODEL_NAME, USE_PRETRAINED_FOR_UPDATE_CHECK)
    if MODEL_NAME == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif MODEL_NAME == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    elif MODEL_NAME == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    elif MODEL_NAME == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    else:
        raise ValueError(f"不支持的模型：{MODEL_NAME}")
    disable_dropout(model)
    return model


def find_probe_parameter(model: nn.Module) -> Tuple[str, nn.Parameter]:
    """优先找分类头权重；找不到时退回最后一个可训练二维参数。"""
    named_params = list(model.named_parameters())
    for keyword in ("classifier", "fc", "head"):
        for name, param in reversed(named_params):
            if keyword in name and param.requires_grad and param.ndim >= 2:
                return name, param
    for name, param in reversed(named_params):
        if param.requires_grad and param.ndim >= 2:
            return name, param
    raise RuntimeError("没有找到可用于权重更新检查的可训练参数。")


def check_model_update(images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, float | int | str | list[str]]:
    """执行一次 forward/backward/step，确认 logits、labels、loss 和权重更新正常。"""
    model = build_model().to(device)
    model.train()
    training_flag_after_train = bool(model.training)

    images = images.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    probe_name, probe_param = find_probe_parameter(model)
    before = probe_param.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    predictions = logits.argmax(dim=1)
    loss.backward()

    no_grad_names = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    grad_mean = float(probe_param.grad.detach().abs().mean().item()) if probe_param.grad is not None else 0.0
    optimizer.step()
    after = probe_param.detach().clone()
    update_mean_abs = float((after - before).abs().mean().item())

    model.eval()
    training_flag_after_eval = bool(model.training)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())

    return {
        "model_name": MODEL_NAME,
        "probe_parameter": probe_name,
        "logits_shape": list(logits.shape),
        "labels_shape": list(labels.shape),
        "labels_dtype": str(labels.dtype),
        "labels_min": int(labels.min().item()),
        "labels_max": int(labels.max().item()),
        "loss": float(loss.item()),
        "predictions_first_20": [int(x) for x in predictions[:20].detach().cpu().tolist()],
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_ratio": float(trainable_params / total_params),
        "probe_grad_abs_mean": grad_mean,
        "probe_update_abs_mean": update_mean_abs,
        "no_grad_param_count": len(no_grad_names),
        "no_grad_param_examples": no_grad_names[:20],
        "model_training_after_train_call": training_flag_after_train,
        "model_training_after_eval_call": training_flag_after_eval,
    }


def main() -> None:
    """执行完整输入管线诊断。"""
    set_random_seed(SEED)
    run_dir = make_run_dir()
    device = resolve_device(DEVICE_NAME)

    class_names, all_samples = list_train_samples()
    selected_samples = choose_balanced_samples(all_samples, BATCH_SIZE)
    dataset = SimpleImageDataset(selected_samples, build_eval_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    images, labels, paths = next(iter(loader))

    batch_stats = {
        "images_shape": list(images.shape),
        "images_dtype": str(images.dtype),
        "images_min": float(images.min().item()),
        "images_max": float(images.max().item()),
        "images_mean": float(images.mean().item()),
        "images_std": float(images.std().item()),
        "labels_first_20": [int(x) for x in labels[:20].tolist()],
        "labels_dtype": str(labels.dtype),
        "label_bincount": [int(x) for x in torch.bincount(labels, minlength=NUM_CLASSES).tolist()],
        "class_names": list(class_names),
        "selected_class_counts": dict(Counter(sample[2] for sample in selected_samples)),
        "sample_paths_first_10": [str(Path(path).relative_to(PROJECT_ROOT).as_posix()) for path in paths[:10]],
    }

    save_batch_preview(run_dir, images, labels, class_names)
    resize_metrics = save_resize_rgb_preview(run_dir)
    update_batch_size = min(UPDATE_CHECK_BATCH_SIZE, images.size(0))
    update_metrics = check_model_update(images[:update_batch_size], labels[:update_batch_size], device)

    summary = {
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(resolve_project_path(DATASET_ROOT)),
        "run_dir": str(run_dir),
        "batch_stats": batch_stats,
        "resize_rgb_metrics": resize_metrics,
        "model_update_metrics": update_metrics,
        "checks": {
            "is_bchw": images.ndim == 4 and images.shape[1] == 3,
            "is_float_tensor": images.dtype == torch.float32,
            "labels_are_long": labels.dtype == torch.long,
            "labels_in_range": int(labels.min()) >= 0 and int(labels.max()) < NUM_CLASSES,
            "logits_are_batch_by_classes": update_metrics["logits_shape"] == [update_batch_size, NUM_CLASSES],
            "weights_updated": update_metrics["probe_update_abs_mean"] > 0.0,
            "probe_has_gradient": update_metrics["probe_grad_abs_mean"] > 0.0,
            "train_eval_flags_ok": update_metrics["model_training_after_train_call"] is True
            and update_metrics["model_training_after_eval_call"] is False,
        },
    }
    (run_dir / "input_pipeline_sanity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("datasets_01234 输入管线诊断完成")
    print(f"输出目录：{run_dir}")
    print("一个 batch 的张量信息：")
    for key, value in batch_stats.items():
        print(f"  {key}: {value}")
    print("模型更新检查：")
    for key, value in update_metrics.items():
        print(f"  {key}: {value}")
    print("最终检查项：")
    for key, value in summary["checks"].items():
        print(f"  {key}: {value}")
    print(f"反归一化 batch 预览：{run_dir / 'batch_unnormalized_preview.png'}")
    print(f"Resize/RGB 对照预览：{run_dir / 'resize_rgb_preview.png'}")
    print(f"完整 JSON：{run_dir / 'input_pipeline_sanity_summary.json'}")
    print("=" * 80)

    failed_checks = [name for name, passed in summary["checks"].items() if not passed]
    if failed_checks:
        raise SystemExit(f"存在未通过检查项：{failed_checks}")


if __name__ == "__main__":
    main()
