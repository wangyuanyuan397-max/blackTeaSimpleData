"""对原图做 CLAHE-L 明度增强，然后随机裁剪 55 个 408x408 patch。

CLAHE 处理原则：
1. 每张图独立处理，不拟合跨数据集模型；
2. 只处理 Lab 颜色空间中的 L 明度通道；
3. a/b 色度通道保持不变；
4. 不做 RGB 三通道独立均衡、不做白平衡、不做颜色直方图均衡；
5. 处理后的完整原图再随机裁剪 55 个 408x408 patch。

默认输出：datasets_01234_CLAHE_L_clip1p5_grid8。
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


# =========================
# 直接右键运行时，只改这里
# =========================

# 当前脚本位于 tools/data012345/CLAHE，因此 parents[3] 是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 输入：已经按原图级别划分好的 train/val/test，不裁剪、不缩放的原图数据集。
SOURCE_ROOT = PROJECT_ROOT / "datasets_01234_original_split"

# 输出：CLAHE-L 处理后再随机裁剪得到的数据集。
OUTPUT_ROOT = PROJECT_ROOT / "datasets_01234_CLAHE_L_clip1p5_grid8"

TIME_CODES = ("00", "10", "20", "30", "40")
SPLITS = ("train", "val", "test")
EXPECTED_SPLIT_COUNTS = {"train": 15, "val": 4, "test": 5}

# 第一轮建议参数：保守增强，避免过度强化噪声和局部伪纹理。
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)

# 每张 CLAHE 处理后的完整原图随机裁剪 55 个 patch。
CROPS_PER_SOURCE = 55
CROP_SIZE = 408

# 默认不 resize，直接保存 408x408；如果后续要 224 版本，可改成 True。
ENABLE_RESIZE_AFTER_CROP = False
RESIZE_SIZE = 224

RANDOM_SEED = 2026
JPEG_QUALITY = 95

# 输出目录非空时默认停止，避免旧结果和新结果混在一起。
ALLOW_NONEMPTY_OUTPUT = False

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def ensure_dependencies() -> None:
    """检查 OpenCV 依赖，缺失时给出明确安装命令。"""
    if cv2 is None:
        raise ImportError("缺少依赖：opencv-python。请先安装：pip install opencv-python")


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


def apply_clahe_to_luminance(
    image_bgr: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """只对 Lab 颜色空间中的 L 明度通道应用 CLAHE。"""
    if image_bgr is None:
        raise ValueError("输入图像为空。")

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    l_clahe = clahe.apply(l_channel)

    # 只替换 L；a/b 色度保持不变，尽量保护茶叶本身颜色。
    lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """把 OpenCV 使用的 BGR 顺序转换成 matplotlib/PIL 常用的 RGB 顺序。"""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_clahe_preview(source_rows: list[dict[str, str]], output_root: Path) -> Path:
    """保存少量原图与 CLAHE-L 结果的对比图，方便先肉眼检查处理效果。"""
    preview_dir = output_root / "_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = source_rows[: min(6, len(source_rows))]
    if not selected_rows:
        raise RuntimeError("没有可用于生成预览图的原图。")

    fig, axes = plt.subplots(
        nrows=len(selected_rows),
        ncols=2,
        figsize=(9, 3 * len(selected_rows)),
        squeeze=False,
    )

    for row_index, row in enumerate(selected_rows):
        source_path = Path(row["source_path"])
        original_bgr = read_bgr(source_path)
        clahe_bgr = apply_clahe_to_luminance(original_bgr)

        axes[row_index][0].imshow(bgr_to_rgb(original_bgr))
        axes[row_index][0].set_title(f"Original: {row['split']}/{row['time_code']}")
        axes[row_index][0].axis("off")

        axes[row_index][1].imshow(bgr_to_rgb(clahe_bgr))
        axes[row_index][1].set_title("CLAHE on Lab-L only")
        axes[row_index][1].axis("off")

    fig.tight_layout()
    preview_path = preview_dir / "clahe_l_preview.png"
    fig.savefig(preview_path, dpi=160)
    plt.close(fig)
    return preview_path


def save_patch(output_path: Path, patch_bgr: np.ndarray) -> None:
    """把裁剪得到的 patch 保存到硬盘，并检查 OpenCV 是否保存成功。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        success = cv2.imwrite(
            str(output_path),
            patch_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
        )
    else:
        success = cv2.imwrite(str(output_path), patch_bgr)

    if not success:
        raise IOError(f"patch 保存失败：{output_path}")


def crop_processed_source(
    row: dict[str, str],
    processed_bgr: np.ndarray,
    crop_manifest_rows: list[dict[str, Any]],
    counts: dict[tuple[str, str], int],
) -> None:
    """对一张已经做过 CLAHE-L 的完整原图随机裁剪 55 个 patch。"""
    split_name = row["split"]
    time_code = row["time_code"]
    source_image_id = row["source_image_id"]
    height, width = processed_bgr.shape[:2]

    if width < CROP_SIZE or height < CROP_SIZE:
        raise ValueError(
            f"图像尺寸小于裁剪尺寸：{row['source_path']}，"
            f"图像={width}x{height}，crop={CROP_SIZE}x{CROP_SIZE}。"
        )

    # 每张父图使用独立但可复现的随机数，避免脚本运行顺序影响裁剪坐标。
    rng = random.Random(f"{RANDOM_SEED}_{split_name}_{time_code}_{source_image_id}")
    max_left = width - CROP_SIZE
    max_top = height - CROP_SIZE

    for crop_index in range(1, CROPS_PER_SOURCE + 1):
        left = rng.randint(0, max_left)
        top = rng.randint(0, max_top)
        right = left + CROP_SIZE
        bottom = top + CROP_SIZE

        patch_bgr = processed_bgr[top:bottom, left:right]

        if ENABLE_RESIZE_AFTER_CROP:
            interpolation = cv2.INTER_AREA if RESIZE_SIZE < CROP_SIZE else cv2.INTER_CUBIC
            patch_bgr = cv2.resize(
                patch_bgr,
                (RESIZE_SIZE, RESIZE_SIZE),
                interpolation=interpolation,
            )

        save_name = f"{source_image_id}__random55_{crop_index:03d}.jpg"
        output_path = OUTPUT_ROOT / split_name / time_code / save_name
        save_patch(output_path, patch_bgr)

        counts[(split_name, time_code)] += 1
        crop_manifest_rows.append(
            {
                "split": split_name,
                "time_code": time_code,
                "source_image_id": source_image_id,
                "source_relpath": row["source_relpath"],
                "source_width": width,
                "source_height": height,
                "crop_index": crop_index,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "preprocess": "CLAHE_L_only",
                "clip_limit": CLAHE_CLIP_LIMIT,
                "tile_grid_size": f"{CLAHE_TILE_GRID_SIZE[0]}x{CLAHE_TILE_GRID_SIZE[1]}",
                "resize_enabled": ENABLE_RESIZE_AFTER_CROP,
                "saved_width": int(patch_bgr.shape[1]),
                "saved_height": int(patch_bgr.shape[0]),
                "target_relpath": output_path.relative_to(OUTPUT_ROOT).as_posix(),
                "target_path": str(output_path.resolve()),
            }
        )


def process_all_sources(
    source_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], int]]:
    """依次处理全部原图：CLAHE-L 明度增强，然后随机裁剪 patch。"""
    apply_manifest_rows: list[dict[str, Any]] = []
    crop_manifest_rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for image_index, row in enumerate(source_rows, start=1):
        source_path = Path(row["source_path"])
        original_bgr = read_bgr(source_path)
        processed_bgr = apply_clahe_to_luminance(original_bgr)

        crop_processed_source(
            row=row,
            processed_bgr=processed_bgr,
            crop_manifest_rows=crop_manifest_rows,
            counts=counts,
        )

        height, width = original_bgr.shape[:2]
        apply_manifest_rows.append(
            {
                "split": row["split"],
                "time_code": row["time_code"],
                "source_image_id": row["source_image_id"],
                "source_relpath": row["source_relpath"],
                "source_width": width,
                "source_height": height,
                "preprocess": "CLAHE_L_only",
                "color_space": "BGR_to_Lab_to_BGR",
                "changed_channel": "L",
                "unchanged_channels": "a,b",
                "clip_limit": CLAHE_CLIP_LIMIT,
                "tile_grid_size": f"{CLAHE_TILE_GRID_SIZE[0]}x{CLAHE_TILE_GRID_SIZE[1]}",
                "crops_per_source": CROPS_PER_SOURCE,
                "crop_size": CROP_SIZE,
                "resize_enabled": ENABLE_RESIZE_AFTER_CROP,
                "resize_size": RESIZE_SIZE if ENABLE_RESIZE_AFTER_CROP else "",
            }
        )

        print(
            f"[{image_index:03d}/{len(source_rows):03d}] "
            f"{row['source_relpath']} -> {CROPS_PER_SOURCE} patches"
        )

    return apply_manifest_rows, crop_manifest_rows, counts


def build_summary(
    source_rows: list[dict[str, str]],
    crop_manifest_rows: list[dict[str, Any]],
    counts: dict[tuple[str, str], int],
    preview_path: Path,
) -> dict[str, Any]:
    """整理本次处理的参数、数量和关键注意事项，写入 JSON 便于追溯。"""
    source_counts = {
        split_name: {
            time_code: sum(
                1
                for row in source_rows
                if row["split"] == split_name and row["time_code"] == time_code
            )
            for time_code in TIME_CODES
        }
        for split_name in SPLITS
    }

    patch_counts = {
        split_name: {
            time_code: counts[(split_name, time_code)]
            for time_code in TIME_CODES
        }
        for split_name in SPLITS
    }

    return {
        "task": "Apply CLAHE to Lab L channel, then random-crop 55 patches per original image.",
        "source_root": str(SOURCE_ROOT.resolve()),
        "output_root": str(OUTPUT_ROOT.resolve()),
        "preprocess": {
            "method": "CLAHE_L_only",
            "color_space": "Lab",
            "changed_channel": "L",
            "unchanged_channels": ["a", "b"],
            "clip_limit": CLAHE_CLIP_LIMIT,
            "tile_grid_size": list(CLAHE_TILE_GRID_SIZE),
            "note": "CLAHE is applied independently to each image, so there is no train/val/test fitting step.",
        },
        "crop": {
            "random_seed": RANDOM_SEED,
            "crops_per_source": CROPS_PER_SOURCE,
            "crop_size": CROP_SIZE,
            "resize_enabled": ENABLE_RESIZE_AFTER_CROP,
            "resize_size": RESIZE_SIZE if ENABLE_RESIZE_AFTER_CROP else None,
            "saved_format": "jpg",
            "jpeg_quality": JPEG_QUALITY,
        },
        "source_counts": source_counts,
        "patch_counts": patch_counts,
        "total_source_images": len(source_rows),
        "total_patches": len(crop_manifest_rows),
        "preview_path": str(preview_path.resolve()),
        "manifests": {
            "clahe_apply_manifest": str((OUTPUT_ROOT / "clahe_apply_manifest.csv").resolve()),
            "random_crop_manifest": str((OUTPUT_ROOT / "random_crop_manifest.csv").resolve()),
        },
    }


def main() -> None:
    """脚本主入口：检查依赖、检查数据、生成预览、批量处理并写出记录。"""
    ensure_dependencies()
    ensure_clean_output_root(OUTPUT_ROOT)

    source_rows = build_source_rows()
    print(f"输入原图数量：{len(source_rows)}")
    print(f"输入目录：{SOURCE_ROOT}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(
        "CLAHE 参数："
        f"clipLimit={CLAHE_CLIP_LIMIT}, "
        f"tileGridSize={CLAHE_TILE_GRID_SIZE}"
    )
    print(
        "裁剪参数："
        f"每张原图 {CROPS_PER_SOURCE} 个，"
        f"crop={CROP_SIZE}x{CROP_SIZE}，"
        f"resize={ENABLE_RESIZE_AFTER_CROP}"
    )

    preview_path = save_clahe_preview(source_rows, OUTPUT_ROOT)
    print(f"CLAHE 效果预览图：{preview_path}")

    apply_manifest_rows, crop_manifest_rows, counts = process_all_sources(source_rows)

    write_csv(OUTPUT_ROOT / "clahe_apply_manifest.csv", apply_manifest_rows)
    write_csv(OUTPUT_ROOT / "random_crop_manifest.csv", crop_manifest_rows)

    summary = build_summary(
        source_rows=source_rows,
        crop_manifest_rows=crop_manifest_rows,
        counts=counts,
        preview_path=preview_path,
    )
    (OUTPUT_ROOT / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n处理完成，patch 数量如下：")
    for split_name in SPLITS:
        split_total = 0
        for time_code in TIME_CODES:
            value = counts[(split_name, time_code)]
            split_total += value
            print(f"  {split_name}/{time_code}: {value}")
        print(f"  {split_name}/total: {split_total}")
    print(f"\n总 patch 数：{len(crop_manifest_rows)}")
    print(f"汇总文件：{OUTPUT_ROOT / 'split_summary.json'}")


if __name__ == "__main__":
    main()

