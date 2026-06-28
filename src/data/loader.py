"""直接读取 root/train|val|test/class_name 目录的数据适配器。"""

from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from ..utils.registry import DATASETS, TRANSFORMS


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@TRANSFORMS.register("patch_train_224")
def build_patch_train_transform(image_size: int = 224) -> T.Compose:
    """为已经裁好的 patch 构建轻量训练增强与 ImageNet 标准化。"""
    return T.Compose(
        [
            T.Resize((image_size, image_size), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@TRANSFORMS.register("patch_eval_224")
def build_patch_eval_transform(image_size: int = 224) -> T.Compose:
    """为验证集和测试集构建无随机增强的确定性变换。"""
    return T.Compose(
        [
            T.Resize((image_size, image_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@DATASETS.register("image_folder")
class ImageFolderWithPaths(Dataset):
    """读取类别子目录中的图片，并返回 image、label、path 三元组。"""

    def __init__(
        self,
        root: str | Path,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"数据目录不存在：{self.root}")

        available_classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if not available_classes:
            raise ValueError(f"数据目录中没有类别文件夹：{self.root}")

        if class_to_idx:
            normalized_mapping = {str(name): int(index) for name, index in class_to_idx.items()}
            expected_indices = list(range(len(normalized_mapping)))
            actual_indices = sorted(normalized_mapping.values())
            if actual_indices != expected_indices:
                raise ValueError(
                    f"class_to_idx 的编号必须从 0 连续递增；当前编号为 {actual_indices}。"
                )
            missing_classes = sorted(set(normalized_mapping) - set(available_classes))
            unexpected_classes = sorted(set(available_classes) - set(normalized_mapping))
            if missing_classes or unexpected_classes:
                raise ValueError(
                    "类别目录与 class_to_idx 不一致："
                    f"缺少={missing_classes}，多出={unexpected_classes}。"
                )
            self.class_to_idx = normalized_mapping
            self.classes = [
                name for name, _ in sorted(normalized_mapping.items(), key=lambda item: item[1])
            ]
        else:
            self.classes = available_classes
            self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_directory = self.root / class_name
            class_samples = sorted(
                path.resolve()
                for path in class_directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not class_samples:
                raise ValueError(f"类别目录中没有图片：{class_directory}")
            label = self.class_to_idx[class_name]
            self.samples.extend((path, label) for path in class_samples)

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


def _build_weighted_sampler(targets: Sequence[int]) -> WeightedRandomSampler:
    """依据类别频数为少数类设置更高采样权重。"""
    target_tensor = torch.as_tensor(list(targets), dtype=torch.long)
    class_counts = torch.bincount(target_tensor)
    if torch.any(class_counts == 0):
        raise ValueError("加权采样器检测到没有样本的类别。")
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[target_tensor]
    return WeightedRandomSampler(sample_weights, num_samples=len(targets), replacement=True)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    is_train: bool,
    num_workers: int = 0,
    enable_weighted_sampler: bool = False,
) -> DataLoader:
    """根据现有 ComponentBuilder 接口创建训练、验证或测试 DataLoader。"""
    sampler = None
    if is_train and enable_weighted_sampler:
        targets = getattr(dataset, "targets", None)
        if targets is None:
            raise ValueError("当前数据集不提供 targets，无法启用 weighted_sampler。")
        sampler = _build_weighted_sampler(targets)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(is_train and sampler is None),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
    )
