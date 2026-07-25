"""MixNet-S 在线数据增强强度对比实验入口。

这个临时脚本只用于排查 01234 五分类任务中的过拟合问题：
1. 固定模型为 fixed_timm_mixnet_s.yaml。
2. 固定公共训练配置为 fixed_split_01234_grid30_408_train.yaml。
3. 只替换训练集 data.train_transform，验证集/测试集保持确定性 transform。
4. 本机只建议 dry-run 或导出增强预览，完整训练放到服务器执行。
"""

from __future__ import annotations

import argparse
import copy
import csv
import html
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import yaml


# 项目根目录。脚本位于 temp/augmentation_overfit_diagnostics/，向上两级回到项目根。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 当前临时实验目录，以及所有训练结果的保存位置。
TEMP_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = TEMP_ROOT / "results"

# 固定公共训练配置：数据集、epoch、batch size、optimizer、scheduler、loss 等都从这里继承。
BASE_COMMON_CONFIG = Path("configs/fixed_split_01234_grid30_408_train.yaml")

# 固定模型配置：本轮只比较 transform，不改 MixNet-S 模型结构。
BASE_MODEL_CONFIG = Path("configs/fixed_split_01234_models/fixed_timm_mixnet_s.yaml")

# 输出 run 名前缀，最终格式为 mixnet_s__{transform_name}__seed{seed}。
BASE_RUN_NAME = "mixnet_s"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_BATCH_PATH = PROJECT_ROOT / "tools" / "train_batch.py"
spec = importlib.util.spec_from_file_location("train_batch_base", TRAIN_BATCH_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"无法加载训练辅助脚本：{TRAIN_BATCH_PATH}")
train_batch_base = importlib.util.module_from_spec(spec)
sys.modules["train_batch_base"] = train_batch_base
spec.loader.exec_module(train_batch_base)

from src.data.loader import (  # noqa: E402
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.utils import TRANSFORMS  # noqa: E402


class AddGaussianNoise:
    """在 ToTensor 之后、Normalize 之前给图像张量添加高斯噪音。"""

    def __init__(self, probability: float = 0.0, std: float = 0.0) -> None:
        self.probability = float(probability)
        self.std = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.probability <= 0.0 or self.std <= 0.0:
            return tensor
        if float(torch.rand(()).item()) >= self.probability:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def build_patch_train_aug_sweep(
    image_size: int = 408,
    crop_scale_min: float = 1.0,
    crop_ratio_min: float = 1.0,
    crop_ratio_max: float = 1.0,
    rotate_degrees: float = 0.0,
    translate: float = 0.0,
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
    noise_p: float = 0.0,
    noise_std: float = 0.0,
) -> T.Compose:
    """为已经裁好的 408x408 patch 构建一套在线训练增强。

    关键参数含义：
    - crop_scale_min：随机裁剪保留面积下限，越小裁剪越强。
    - rotate_degrees：随机旋转角度范围。
    - translate：平移比例，例如 0.05 表示宽高方向最多平移 5%。
    - noise_p/noise_std：添加高斯噪音的概率和强度。
    """

    transforms: list[Any] = []
    if float(crop_scale_min) < 1.0 or float(crop_ratio_min) != 1.0 or float(crop_ratio_max) != 1.0:
        transforms.append(
            T.RandomResizedCrop(
                size=(int(image_size), int(image_size)),
                scale=(float(crop_scale_min), 1.0),
                ratio=(float(crop_ratio_min), float(crop_ratio_max)),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        )
    else:
        transforms.append(T.Resize((int(image_size), int(image_size)), antialias=True))

    if float(rotate_degrees) > 0.0 or float(translate) > 0.0:
        transforms.append(
            T.RandomAffine(
                degrees=float(rotate_degrees),
                translate=(float(translate), float(translate)) if float(translate) > 0.0 else None,
                interpolation=InterpolationMode.BILINEAR,
                fill=(124, 116, 104),
            )
        )

    if float(hflip_p) > 0.0:
        transforms.append(T.RandomHorizontalFlip(p=float(hflip_p)))
    if float(vflip_p) > 0.0:
        transforms.append(T.RandomVerticalFlip(p=float(vflip_p)))

    transforms.extend(
        [
            T.ToTensor(),
            AddGaussianNoise(probability=float(noise_p), std=float(noise_std)),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return T.Compose(transforms)


# 临时注册一个 transform builder，训练配置中的 type=patch_train_aug_sweep 会走到这里。
# 这里不改 src/data/loader.py，是为了让本轮探索保持在 temp 目录内。
TRANSFORMS.module_dict["patch_train_aug_sweep"] = build_patch_train_aug_sweep


@dataclass(frozen=True)
class TransformExperiment:
    name: str
    description: str
    train_transform: dict[str, Any]
    random_seed: int = 2026


def aug_transform(**kwargs: Any) -> dict[str, Any]:
    """生成一组增强参数。

    默认值等价于当前基线训练增强：Resize + 水平翻转 + 垂直翻转。
    后面的实验只覆盖需要变化的参数，方便直接比较单类/组合增强。
    """
    config: dict[str, Any] = {
        "type": "patch_train_aug_sweep",
        "image_size": 408,
        # 随机裁剪参数；1.0 表示不裁剪，只做 Resize。
        "crop_scale_min": 1.0,
        "crop_ratio_min": 1.0,
        "crop_ratio_max": 1.0,
        # 仿射增强参数；0.0 表示不旋转、不平移。
        "rotate_degrees": 0.0,
        "translate": 0.0,
        # 当前基线中已经包含水平/垂直翻转，所以默认保留。
        "hflip_p": 0.5,
        "vflip_p": 0.5,
        # 噪音增强参数；noise_p=0 表示不添加噪音。
        "noise_p": 0.0,
        "noise_std": 0.0,
    }
    config.update(kwargs)
    return config


# 核心实验列表：服务器会按这里的顺序逐个训练并汇总结果。
# 想新增一组增强，只需要复制一个 TransformExperiment 并修改 name/description/train_transform。
TRANSFORM_EXPERIMENT_LIST: list[TransformExperiment] = [
    TransformExperiment(
        name="baseline_current_flip",
        description="当前基线训练增强：Resize + 水平翻转 + 垂直翻转。",
        train_transform=aug_transform(),
    ),
    TransformExperiment(
        name="crop_light",
        description="在基线翻转基础上加入轻度随机裁剪，观察裁剪本身是否缓解过拟合。",
        train_transform=aug_transform(crop_scale_min=0.92, crop_ratio_min=0.98, crop_ratio_max=1.02),
    ),
    TransformExperiment(
        name="affine_light",
        description="在基线翻转基础上加入轻度旋转和平移，不加裁剪、不加噪音。",
        train_transform=aug_transform(rotate_degrees=5.0, translate=0.03),
    ),
    TransformExperiment(
        name="noise_light",
        description="在基线翻转基础上只加入轻度高斯噪音，用于判断纹理扰动是否有效。",
        train_transform=aug_transform(noise_p=0.5, noise_std=0.015),
    ),
    TransformExperiment(
        name="geom_medium",
        description="中等几何增强：随机裁剪 + 旋转 + 平移 + 翻转。",
        train_transform=aug_transform(
            crop_scale_min=0.85,
            crop_ratio_min=0.95,
            crop_ratio_max=1.05,
            rotate_degrees=10.0,
            translate=0.05,
        ),
    ),
    TransformExperiment(
        name="geom_noise_medium",
        description="中等几何增强 + 中等高斯噪音，作为本轮主力候选增强。",
        train_transform=aug_transform(
            crop_scale_min=0.85,
            crop_ratio_min=0.95,
            crop_ratio_max=1.05,
            rotate_degrees=10.0,
            translate=0.05,
            noise_p=0.5,
            noise_std=0.02,
        ),
    ),
    TransformExperiment(
        name="geom_strong",
        description="强几何增强：更强随机裁剪、旋转和平移，用于测试增强过强的边界。",
        train_transform=aug_transform(
            crop_scale_min=0.75,
            crop_ratio_min=0.90,
            crop_ratio_max=1.10,
            rotate_degrees=18.0,
            translate=0.08,
        ),
    ),
    TransformExperiment(
        name="geom_noise_strong",
        description="强几何增强 + 较强高斯噪音，用于检查过强扰动是否伤害泛化。",
        train_transform=aug_transform(
            crop_scale_min=0.75,
            crop_ratio_min=0.90,
            crop_ratio_max=1.10,
            rotate_degrees=18.0,
            translate=0.08,
            noise_p=0.7,
            noise_std=0.035,
        ),
    ),
]


def load_yaml_mapping(relative_path: Path, label: str) -> dict[str, Any]:
    absolute_path = PROJECT_ROOT / relative_path
    if not absolute_path.is_file():
        raise FileNotFoundError(f"{label} 不存在：{absolute_path}")
    with absolute_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{label} 顶层必须是 YAML 字典：{absolute_path}")
    return data


def experiment_by_name() -> dict[str, TransformExperiment]:
    return {experiment.name: experiment for experiment in TRANSFORM_EXPERIMENT_LIST}


def select_experiments(requested: list[str] | None) -> list[TransformExperiment]:
    if not requested:
        return list(TRANSFORM_EXPERIMENT_LIST)
    available = experiment_by_name()
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(f"未知 transform 实验：{unknown}。可选项：{sorted(available)}")
    if len(set(requested)) != len(requested):
        raise ValueError("--experiments 不能包含重复名称。")
    return [available[name] for name in requested]


def print_experiments(experiments: list[TransformExperiment]) -> None:
    print("数据增强实验列表:")
    for experiment in experiments:
        params = {k: v for k, v in experiment.train_transform.items() if k != "type"}
        print(f"  - {experiment.name}: {experiment.description}")
        print(f"    {params}")


def build_common_config_for_experiment(
    base_common: dict[str, Any],
    experiment: TransformExperiment,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """复制公共训练配置，并只替换本轮实验的 train_transform。"""
    common = copy.deepcopy(base_common)
    common["runs_root"] = str(RESULTS_ROOT.relative_to(PROJECT_ROOT))
    common["augmentation_experiment"] = {
        "name": experiment.name,
        "description": experiment.description,
        "train_transform": copy.deepcopy(experiment.train_transform),
    }
    common.setdefault("data", {})
    common["data"]["train_transform"] = copy.deepcopy(experiment.train_transform)
    common["data"]["eval_transform"] = {"type": "patch_eval_224", "image_size": 408}
    common["data"]["test_transform"] = {"type": "patch_eval_224", "image_size": 408}
    common.setdefault("train", {})
    common["train"]["num_workers"] = int(args.num_workers)
    common["train"]["keep_pth_files"] = bool(args.keep_pth_files)
    return common


def build_model_config_for_experiment(
    base_model: dict[str, Any],
    experiment: TransformExperiment,
    seed: int,
) -> dict[str, Any]:
    """复制 MixNet-S 模型配置，并为当前 transform/seed 生成独立 run 名。"""
    model_config = copy.deepcopy(base_model)
    model_config["name"] = f"{BASE_RUN_NAME}__{experiment.name}__seed{int(seed)}"
    model_config["random_seed"] = int(seed)
    return model_config


def build_preview_config(
    base_common: dict[str, Any],
    base_model: dict[str, Any],
    experiment: TransformExperiment,
    seed: int,
    args: argparse.Namespace,
    dataset_root: Path,
    device: torch.device,
):
    common = build_common_config_for_experiment(base_common, experiment, args)
    model = build_model_config_for_experiment(base_model, experiment, seed)
    preview_directory = RESULTS_ROOT / "__dry_run__"
    return train_batch_base.build_training_config_from_file(
        common,
        model,
        BASE_MODEL_CONFIG,
        dataset_root,
        preview_directory,
        device,
    )


def read_history(run_directory: str | Path | None) -> dict[str, list[Any]]:
    if not run_directory:
        return {}
    history_path = Path(run_directory) / "history.json"
    if not history_path.is_file():
        return {}
    with history_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def list_value(values: Any, index: int) -> Any:
    if not isinstance(values, list) or not values:
        return None
    if index < 0 or index >= len(values):
        return None
    return values[index]


def numeric_gap(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def result_to_transform_row(
    result: dict[str, Any],
    experiment: TransformExperiment,
    seed: int,
    batch_timestamp: str,
) -> dict[str, Any]:
    """把单次训练结果整理成 transform 横向比较表的一行。"""
    metrics = result.get("metrics") or {}
    history = read_history(result.get("run_directory"))
    best_epoch = metrics.get("best_epoch")
    best_index = int(best_epoch) - 1 if best_epoch is not None else -1
    best_train_acc = list_value(history.get("train_acc"), best_index)
    best_val_acc = list_value(history.get("val_acc"), best_index)
    final_train_acc = list_value(history.get("train_acc"), len(history.get("train_acc", [])) - 1)
    final_val_acc = list_value(history.get("val_acc"), len(history.get("val_acc", [])) - 1)
    val_values = [float(value) for value in history.get("val_acc", []) if value is not None]
    final_val_float = float(final_val_acc) if final_val_acc is not None else None
    val_peak_to_final_drop = (
        max(val_values) - final_val_float
        if val_values and final_val_float is not None
        else None
    )
    return {
        "batch_timestamp": batch_timestamp,
        "transform_name": experiment.name,
        "seed": int(seed),
        "run_name": result.get("model_name"),
        "status": result.get("status"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "mae": metrics.get("mae"),
        "qwk": metrics.get("qwk"),
        "plus_minus_one_accuracy": metrics.get("plus_minus_one_accuracy"),
        "best_epoch": best_epoch,
        "train_acc_at_best": best_train_acc,
        "val_acc_at_best": best_val_acc,
        "train_val_gap_at_best": numeric_gap(best_train_acc, best_val_acc),
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "final_train_val_gap": numeric_gap(final_train_acc, final_val_acc),
        "val_peak_to_final_drop": val_peak_to_final_drop,
        "training_time_seconds": metrics.get("training_time_seconds"),
        "run_directory": result.get("run_directory"),
        "report_path": result.get("report_path"),
        "description": experiment.description,
        "train_transform": json.dumps(experiment.train_transform, ensure_ascii=True, sort_keys=True),
        "error": result.get("error"),
    }


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_transform_summary_html(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "transform_name",
        "seed",
        "status",
        "accuracy",
        "macro_f1",
        "mae",
        "qwk",
        "best_epoch",
        "train_acc_at_best",
        "val_acc_at_best",
        "train_val_gap_at_best",
        "final_train_val_gap",
        "report_path",
    ]
    table_rows = []
    for row in rows:
        cells = []
        for key in preferred:
            value = row.get(key)
            if key == "report_path" and value:
                try:
                    link = Path(value).relative_to(path.parent).as_posix()
                except ValueError:
                    link = str(value)
                value_text = f"<a href='{html.escape(link)}'>报告</a>"
            else:
                value_text = html.escape(format_cell(value))
            cells.append(f"<td>{value_text}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")
    headers = "".join(f"<th>{html.escape(key)}</th>" for key in preferred)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Transform 增强对比汇总</title>
  <style>
    body {{ max-width: 1180px; margin: 32px auto; padding: 0 18px; font: 14px/1.5 system-ui, sans-serif; color: #172033; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #dfe5ef; padding: 8px; text-align: center; }}
    th {{ background: #f0f4fa; position: sticky; top: 0; }}
    a {{ color: #315efb; }}
  </style>
</head>
<body>
  <h1>Transform 增强对比汇总</h1>
  <p>每一行对应一个 transform + seed。理想信号是测试指标稳定或提升，同时 train/val gap 下降。</p>
  <table>
    <thead><tr>{headers}</tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def find_preview_image_paths(dataset_root: Path, limit: int) -> list[Path]:
    """从训练集中每个类别取少量样本，用于导出增强预览图。"""
    train_root = dataset_root / "train"
    paths: list[Path] = []
    for class_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(image_path)
                break
        if len(paths) >= limit:
            break
    return paths


def unnormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """把 Normalize 后的 tensor 还原到 0-1 范围，方便保存为预览图。"""
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return torch.clamp(tensor.detach().cpu() * std + mean, 0.0, 1.0)


def export_transform_previews(
    experiments: list[TransformExperiment],
    dataset_root: Path,
    sample_count: int,
    variant_count: int,
) -> Path:
    """导出每个 transform 的少量增强结果，便于人工检查是否破坏图像语义。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_root = TEMP_ROOT / "preview_transforms" / timestamp
    preview_root.mkdir(parents=True, exist_ok=False)
    source_paths = find_preview_image_paths(dataset_root, sample_count)
    if not source_paths:
        raise FileNotFoundError(f"未在训练集目录中找到可预览图片：{dataset_root / 'train'}")
    to_pil = T.ToPILImage()
    for experiment in experiments:
        transform_kwargs = {k: v for k, v in experiment.train_transform.items() if k != "type"}
        transform = build_patch_train_aug_sweep(**transform_kwargs)
        experiment_dir = preview_root / experiment.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for sample_index, image_path in enumerate(source_paths, start=1):
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            for variant_index in range(1, variant_count + 1):
                tensor = transform(image)
                preview = to_pil(unnormalize_tensor(tensor))
                preview.save(
                    experiment_dir / f"sample{sample_index:02d}_variant{variant_index:02d}.jpg",
                    quality=95,
                )
    return preview_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 MixNet-S 01234 在线数据增强对比实验。"
    )
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--experiments", nargs="+", help="只运行指定 transform 名称。")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="覆盖所有 transform 的随机种子；不传时使用每个 list item 自带的 random_seed。",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。服务器建议 4，本机 dry-run 建议 0。")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置、数据集和 transform，不训练模型。")
    parser.add_argument("--fail-fast", action="store_true", help="任一实验失败后立即停止后续实验。")
    parser.add_argument("--preview-transforms", action="store_true", help="导出少量增强预览图，便于人工检查增强强度。")
    parser.add_argument("--preview-samples", type=int, default=3, help="每次预览选取多少张源图。")
    parser.add_argument("--preview-variants", type=int, default=3, help="每张源图为每个 transform 导出多少个随机版本。")
    pth_group = parser.add_mutually_exclusive_group()
    pth_group.add_argument("--keep-pth", dest="keep_pth_files", action="store_true", help="保留每个实验的 best_model.pth。")
    pth_group.add_argument("--discard-pth", dest="keep_pth_files", action="store_false", help="评估完成后删除 pth，节省服务器空间。")
    parser.set_defaults(keep_pth_files=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = select_experiments(args.experiments)
    seed_plan = {
        experiment.name: [int(seed) for seed in args.seeds]
        if args.seeds
        else [int(experiment.random_seed)]
        for experiment in experiments
    }

    if args.list_experiments:
        print_experiments(experiments)
        return

    base_common = load_yaml_mapping(BASE_COMMON_CONFIG, "公共训练配置")
    base_model = load_yaml_mapping(BASE_MODEL_CONFIG, "模型配置")
    device = train_batch_base.resolve_device(args.device)
    dataset_root = train_batch_base.resolve_project_path(
        base_common.get("dataset_root", "datasets_01234_grid30_408")
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_root}")

    if args.preview_transforms:
        preview_root = export_transform_previews(
            experiments,
            dataset_root,
            sample_count=int(args.preview_samples),
            variant_count=int(args.preview_variants),
        )
        print(f"增强预览图已保存到: {preview_root}")

    first_seed = seed_plan[experiments[0].name][0]
    print("数据增强对比实验设置:")
    print(f"  公共训练配置: {BASE_COMMON_CONFIG.as_posix()}")
    print(f"  固定模型配置: {BASE_MODEL_CONFIG.as_posix()}")
    print(f"  数据集目录: {dataset_root}")
    print(f"  结果目录: {RESULTS_ROOT}")
    print(f"  训练设备: {device}")
    print(f"  随机种子计划: {seed_plan}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  是否保留 pth: {bool(args.keep_pth_files)}")
    print_experiments(experiments)

    if args.dry_run:
        print("\ndry-run：逐个检查 transform 对应的 dataloader。")
        for experiment in experiments:
            dry_run_seed = seed_plan[experiment.name][0]
            config = build_preview_config(
                base_common,
                base_model,
                experiment,
                dry_run_seed,
                args,
                dataset_root,
                device,
            )
            summary = train_batch_base.validate_fixed_dataset(config, device)
            train_shape = summary["train"]["sample_shape"]
            print(f"  - {experiment.name}: 训练样本张量形状={train_shape}")
        print("dry-run 完成：未创建正式训练 run。")
        return

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    should_stop = False

    for experiment in experiments:
        for seed in seed_plan[experiment.name]:
            common = build_common_config_for_experiment(base_common, experiment, args)
            model_config = build_model_config_for_experiment(base_model, experiment, int(seed))
            result = train_batch_base.run_config_file(
                common,
                BASE_MODEL_CONFIG,
                model_config,
                dataset_root,
                RESULTS_ROOT,
                device,
            )
            result["transform_name"] = experiment.name
            result["transform_description"] = experiment.description
            result["seed"] = int(seed)
            results.append(result)
            summary_rows.append(result_to_transform_row(result, experiment, int(seed), batch_timestamp))
            write_csv_rows(RESULTS_ROOT / f"transform_sweep_{batch_timestamp}_summary.csv", summary_rows)
            write_transform_summary_html(
                RESULTS_ROOT / f"transform_sweep_{batch_timestamp}_summary.html",
                summary_rows,
            )
            if result.get("status") == "failed" and args.fail_fast:
                should_stop = True
                break
        if should_stop:
            break

    batch_summary = train_batch_base.write_batch_summary(RESULTS_ROOT, batch_timestamp, results)
    transform_csv = RESULTS_ROOT / f"transform_sweep_{batch_timestamp}_summary.csv"
    transform_html = RESULTS_ROOT / f"transform_sweep_{batch_timestamp}_summary.html"
    print(f"\n批量总览报告: {batch_summary}")
    print(f"数据增强对比 CSV: {transform_csv}")
    print(f"数据增强对比 HTML: {transform_html}")
    failed = [result.get("model_name") for result in results if result.get("status") == "failed"]
    if failed:
        print(f"失败的实验: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
