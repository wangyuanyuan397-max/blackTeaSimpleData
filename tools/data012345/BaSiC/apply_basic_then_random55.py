"""对原图做 BaSiC 明度照明矫正，然后随机裁剪 55 个 408x408 patch。

严格防止数据泄露：
1. 只使用 datasets_01234_original_split/train 下的完整原图拟合 BaSiC 照明场；
2. val/test 不参与 fit；
3. 同一个由 train 拟合出的明度照明场应用到 train/val/test；
4. 只矫正明度：用单通道 flatfield 生成 gain，并把同一个 gain 同时乘到 R/G/B 三通道。

默认输出：datasets_01234_BaSic。
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - 运行时给出更友好的安装提示
    cv2 = None

try:
    from basicpy import BaSiC
except ImportError:  # pragma: no cover - 运行时给出更友好的安装提示
    BaSiC = None


# =========================
# 直接右键运行时，只改这里
# =========================

# 当前脚本位于 tools/data012345/BaSiC，因此 parents[3] 是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 输入：已经按原图级别划分好的 train/val/test，不裁剪、不缩放的原图数据集。
SOURCE_ROOT = PROJECT_ROOT / "datasets_01234_original_split"

# 输出：BaSiC 明度矫正后再随机裁剪得到的数据集。
OUTPUT_ROOT = PROJECT_ROOT / "datasets_01234_BaSic"

TIME_CODES = ("00", "10", "20", "30", "40")
SPLITS = ("train", "val", "test")
EXPECTED_SPLIT_COUNTS = {"train": 15, "val": 4, "test": 5}

# BaSiC 只用 train 原图拟合照明场，禁止改成 val/test/all。
FIT_SPLIT = "train"

# cv2.resize 的 size 顺序是 (width, height)。这里用于低分辨率拟合照明场。
FIT_SIZE = (306, 256)

# 每张 BaSiC 矫正后的完整原图随机裁剪 55 个 patch。
CROPS_PER_SOURCE = 55
CROP_SIZE = 408

# 默认不 resize，直接保存 408x408；如果后续要 224 版本，可改成 True。
ENABLE_RESIZE_AFTER_CROP = False
RESIZE_SIZE = 224

RANDOM_SEED = 2026
JPEG_QUALITY = 95

# 只矫正明度的增益限制，避免暗角被异常放大。
GAIN_MIN = 0.7
GAIN_MAX = 1.4

# 输出目录非空时默认停止，避免旧结果和新结果混在一起。
ALLOW_NONEMPTY_OUTPUT = False

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def ensure_dependencies() -> None:
    """检查运行依赖，缺失时给出明确安装命令。"""
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if BaSiC is None:
        missing.append("basicpy")
    if missing:
        raise ImportError(
            "缺少依赖："
            + ", ".join(missing)
            + "。请先安装：pip install basicpy opencv-python"
        )


def ensure_clean_output_root(output_root: Path) -> None:
    """创建输出目录，并在目录非空时按配置决定是否停止。"""
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()) and not ALLOW_NONEMPTY_OUTPUT:
        raise FileExistsError(
            f"输出目录已存在且非空：{output_root}\n"
            "为避免混入旧结果，脚本已停止。请先手动清空该目录，"
            "或者把脚本顶部 ALLOW_NONEMPTY_OUTPUT 改为 True。"
        )


def list_split_images(split_name: str, time_code: str) -> list[Path]:
    """列出某个 split/time_code 下的原图，并检查数量。"""
    folder = SOURCE_ROOT / split_name / time_code
    if not folder.is_dir():
        raise FileNotFoundError(f"原图目录不存在：{folder}")

    images = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    expected_count = EXPECTED_SPLIT_COUNTS[split_name]
    if len(images) != expected_count:
        raise ValueError(
            f"{split_name}/{time_code} 应有 {expected_count} 张原图，实际为 {len(images)}。"
        )
    return images


def build_source_rows() -> list[dict[str, str]]:
    """构建 train/val/test 全部原图清单。"""
    rows: list[dict[str, str]] = []
    for split_name in SPLITS:
        for time_code in TIME_CODES:
            for image_path in list_split_images(split_name, time_code):
                rows.append(
                    {
                        "split": split_name,
                        "time_code": time_code,
                        "source_image_id": f"t{time_code}__{image_path.stem}",
                        "source_stem": image_path.stem,
                        "source_suffix": image_path.suffix,
                        "source_relpath": image_path.relative_to(SOURCE_ROOT).as_posix(),
                        "source_path": str(image_path.resolve()),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_bgr(path: Path) -> np.ndarray:
    """用 OpenCV 读取 BGR 图像，并在读取失败时报错。"""
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    return image_bgr


def load_luminance_stack(image_paths: list[Path], fit_size: tuple[int, int]) -> np.ndarray:
    """只读取训练集完整原图，构建用于 BaSiC fit 的亮度堆栈 [T,Y,X]。"""
    stack = []
    for path in image_paths:
        image_bgr = read_bgr(path)

        # 明确转换为 RGB 后再转灰度，避免 BGR/RGB 通道语义混乱。
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        luminance = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 低分辨率拟合照明场；照明场是大尺度缓慢变化，不需要保留茶叶细纹理。
        luminance = cv2.resize(luminance, fit_size, interpolation=cv2.INTER_AREA)
        stack.append(luminance.astype(np.float32))

    if not stack:
        raise RuntimeError("用于 BaSiC fit 的训练集图像列表为空。")
    return np.stack(stack, axis=0)


def fit_luminance_basic(train_rows: list[dict[str, str]]) -> np.ndarray:
    """只用 train split 的完整原图拟合单通道明度 flatfield。"""
    fit_rows = [row for row in train_rows if row["split"] == FIT_SPLIT]
    if any(row["split"] != "train" for row in fit_rows):
        raise RuntimeError("数据泄露风险：BaSiC fit 列表中出现了非 train 图像。")

    train_paths = [Path(row["source_path"]) for row in fit_rows]
    images = load_luminance_stack(train_paths, FIT_SIZE)
    basic = BaSiC(get_darkfield=False, fitting_mode="approximate")
    basic.fit(images)

    flatfield = np.asarray(basic.flatfield, dtype=np.float32).squeeze()
    if flatfield.ndim != 2:
        raise RuntimeError(f"BaSiC flatfield 应为二维数组，实际 shape={flatfield.shape}")
    if not np.all(np.isfinite(flatfield)):
        raise RuntimeError("BaSiC flatfield 中存在 NaN 或 Inf。")
    return flatfield


def compute_gain(flatfield: np.ndarray, gain_min: float = GAIN_MIN, gain_max: float = GAIN_MAX) -> np.ndarray:
    """由单通道 flatfield 计算明度校正 gain。"""
    flatfield = np.maximum(flatfield.astype(np.float32), 1e-6)
    target = float(np.median(flatfield))
    gain = target / flatfield
    return np.clip(gain, gain_min, gain_max).astype(np.float32)


def apply_luminance_flatfield(image_bgr: np.ndarray, flatfield_small: np.ndarray) -> np.ndarray:
    """只矫正明度：同一个单通道 gain 同时乘到 B/G/R 三个通道。

    注意：这里不分别拟合 RGB，也不做白平衡或颜色直方图均衡。
    OpenCV 内部数组是 BGR 顺序，但同一个 gain 乘到三个通道，所以颜色比例保持不变。
    """
    height, width = image_bgr.shape[:2]
    flatfield = cv2.resize(flatfield_small, (width, height), interpolation=cv2.INTER_CUBIC)
    gain = compute_gain(flatfield)

    corrected = image_bgr.astype(np.float32) * gain[..., None]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def save_flatfield_preview(flatfield_small: np.ndarray, output_dir: Path) -> None:
    """保存 flatfield 和 gain 的可视化图，方便肉眼检查照明场。"""
    gain_small = compute_gain(flatfield_small)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    im0 = axes[0].imshow(flatfield_small, cmap="viridis")
    axes[0].set_title("BaSiC flatfield from train luminance")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(gain_small, cmap="magma")
    axes[1].set_title(f"clipped gain [{GAIN_MIN}, {GAIN_MAX}]")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "flatfield_preview.png")
    plt.close(fig)


def save_correction_preview(source_rows: list[dict[str, str]], flatfield_small: np.ndarray, output_dir: Path) -> None:
    """保存若干原图矫正前/后的对照图。"""
    preview_rows = source_rows[: min(6, len(source_rows))]
    if not preview_rows:
        return

    fig, axes = plt.subplots(len(preview_rows), 2, figsize=(8, 3 * len(preview_rows)), dpi=150)
    if len(preview_rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, row in enumerate(preview_rows):
        source_path = Path(row["source_path"])
        original_bgr = read_bgr(source_path)
        corrected_bgr = apply_luminance_flatfield(original_bgr, flatfield_small)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)

        # 原图太大，预览图只缩小显示，不影响实际处理。
        axes[row_index, 0].imshow(original_rgb)
        axes[row_index, 0].set_title(f"original {row['split']}/{row['time_code']}")
        axes[row_index, 0].axis("off")
        axes[row_index, 1].imshow(corrected_rgb)
        axes[row_index, 1].set_title("BaSiC luminance corrected")
        axes[row_index, 1].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "basic_correction_preview.png")
    plt.close(fig)


def save_patch(path: Path, patch_bgr: np.ndarray) -> None:
    """用 OpenCV 保存 BGR patch，并设置 JPEG 质量。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), patch_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    if not ok:
        raise RuntimeError(f"保存 patch 失败：{path}")


def crop_corrected_source(
    row: dict[str, str],
    corrected_bgr: np.ndarray,
    crop_manifest_rows: list[dict[str, Any]],
    counts: dict[str, dict[str, int]],
) -> None:
    """对一张已经完成 BaSiC 明度矫正的完整原图随机裁剪 55 个 patch。"""
    split_name = row["split"]
    time_code = row["time_code"]
    source_image_id = row["source_image_id"]
    output_dir = OUTPUT_ROOT / split_name / time_code

    height, width = corrected_bgr.shape[:2]
    if width < CROP_SIZE or height < CROP_SIZE:
        raise ValueError(f"原图 {row['source_path']} 尺寸为 {width}x{height}，小于裁剪尺寸 {CROP_SIZE}。")

    # 每张父图独立随机流，保证与旧随机裁剪脚本一样稳定、可复现。
    rng = random.Random(f"{RANDOM_SEED}_{source_image_id}")
    for crop_index in range(1, CROPS_PER_SOURCE + 1):
        left = rng.randint(0, width - CROP_SIZE)
        top = rng.randint(0, height - CROP_SIZE)
        right = left + CROP_SIZE
        bottom = top + CROP_SIZE

        patch_bgr = corrected_bgr[top:bottom, left:right]
        if ENABLE_RESIZE_AFTER_CROP:
            patch_bgr = cv2.resize(patch_bgr, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_CUBIC)

        save_name = f"{source_image_id}__random55_{crop_index:03d}.jpg"
        save_path = output_dir / save_name
        save_patch(save_path, patch_bgr)
        counts[split_name][time_code] += 1
        crop_manifest_rows.append(
            {
                "split": split_name,
                "time_code": time_code,
                "source_image_id": source_image_id,
                "source_relpath": row["source_relpath"],
                "crop_index": crop_index,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "crop_size": CROP_SIZE,
                "preprocess": "BaSiC_luminance_only",
                "fit_split": FIT_SPLIT,
                "gain_min": GAIN_MIN,
                "gain_max": GAIN_MAX,
                "resize_enabled": ENABLE_RESIZE_AFTER_CROP,
                "resize_size": RESIZE_SIZE if ENABLE_RESIZE_AFTER_CROP else "",
                "output_size": RESIZE_SIZE if ENABLE_RESIZE_AFTER_CROP else CROP_SIZE,
                "target_relpath": save_path.relative_to(OUTPUT_ROOT).as_posix(),
            }
        )


def apply_basic_and_crop_all(source_rows: list[dict[str, str]], flatfield_small: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    """把 train 拟合得到的同一个明度 flatfield 应用到 train/val/test，并裁剪 patch。"""
    apply_manifest_rows: list[dict[str, Any]] = []
    crop_manifest_rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in source_rows:
        source_path = Path(row["source_path"])
        original_bgr = read_bgr(source_path)
        corrected_bgr = apply_luminance_flatfield(original_bgr, flatfield_small)
        crop_corrected_source(row, corrected_bgr, crop_manifest_rows, counts)

        apply_row = dict(row)
        apply_row.update(
            {
                "used_for_basic_fit": row["split"] == FIT_SPLIT,
                "basic_fit_split": FIT_SPLIT,
                "correction_type": "luminance_only_same_gain_for_bgr_channels",
                "gain_min": GAIN_MIN,
                "gain_max": GAIN_MAX,
            }
        )
        apply_manifest_rows.append(apply_row)

    return apply_manifest_rows, crop_manifest_rows, counts


def main() -> None:
    ensure_dependencies()
    ensure_clean_output_root(OUTPUT_ROOT)
    source_rows = build_source_rows()
    fit_rows = [row for row in source_rows if row["split"] == FIT_SPLIT]

    print("=" * 100)
    print("BaSiC 明度矫正 + 随机裁剪开始")
    print(f"输入原图级数据集：{SOURCE_ROOT}")
    print(f"输出 patch 数据集：{OUTPUT_ROOT}")
    print(f"BaSiC fit 只使用：{FIT_SPLIT}，图像数量：{len(fit_rows)}")
    print(f"fit_size(width,height)：{FIT_SIZE}")
    print("矫正方式：只矫正明度，同一个 gain 同时乘到 B/G/R 三个通道。")
    print("=" * 100)

    flatfield_small = fit_luminance_basic(source_rows)
    np.save(OUTPUT_ROOT / "flatfield_small.npy", flatfield_small)
    save_flatfield_preview(flatfield_small, OUTPUT_ROOT)
    save_correction_preview(fit_rows, flatfield_small, OUTPUT_ROOT)

    apply_manifest_rows, crop_manifest_rows, counts = apply_basic_and_crop_all(source_rows, flatfield_small)
    write_csv(OUTPUT_ROOT / "basic_fit_manifest.csv", fit_rows)
    write_csv(OUTPUT_ROOT / "basic_apply_manifest.csv", apply_manifest_rows)
    write_csv(OUTPUT_ROOT / "random_crop_manifest.csv", crop_manifest_rows)

    summary = {
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "time_codes": list(TIME_CODES),
        "splits": list(SPLITS),
        "expected_split_counts_per_time": EXPECTED_SPLIT_COUNTS,
        "random_seed": RANDOM_SEED,
        "basic_fit_split": FIT_SPLIT,
        "basic_fit_image_count": len(fit_rows),
        "basic_fit_size_width_height": list(FIT_SIZE),
        "correction_type": "BaSiC_luminance_only_same_gain_for_bgr_channels",
        "no_data_leakage_rule": "BaSiC.fit uses train split only; fitted flatfield is applied to train/val/test.",
        "gain_min": GAIN_MIN,
        "gain_max": GAIN_MAX,
        "crops_per_source": CROPS_PER_SOURCE,
        "crop_size": CROP_SIZE,
        "resize_after_crop": ENABLE_RESIZE_AFTER_CROP,
        "resize_size": RESIZE_SIZE if ENABLE_RESIZE_AFTER_CROP else None,
        "output_size": RESIZE_SIZE if ENABLE_RESIZE_AFTER_CROP else CROP_SIZE,
        "total_source_count": len(source_rows),
        "total_crop_count": len(crop_manifest_rows),
        "crop_counts": {split: dict(time_counts) for split, time_counts in counts.items()},
    }
    with (OUTPUT_ROOT / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("BaSiC 明度矫正和随机裁剪完成。")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"flatfield：{OUTPUT_ROOT / 'flatfield_small.npy'}")
    print(f"预览图：{OUTPUT_ROOT / 'flatfield_preview.png'}")
    print(f"矫正对照：{OUTPUT_ROOT / 'basic_correction_preview.png'}")
    print(f"随机裁剪图数量：{len(crop_manifest_rows)}")
    for split_name in SPLITS:
        print(f"{split_name}: {dict(counts[split_name])}")


if __name__ == "__main__":
    main()


