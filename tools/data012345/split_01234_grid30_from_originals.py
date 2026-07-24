"""按原图级别划分 00/10/20/30/40，然后按固定网格裁剪 30 个 patch。

核心逻辑：
1. 每个时间点 24 张原图；
2. 先按原图级别随机划分 train/val/test = 15/4/5；
3. 每张原图按照 6 列 x 5 行固定网格裁剪 30 个 408x408 patch；
4. 横向完整使用 2448 像素，纵向使用 2040 像素，底部剩余 8 像素直接丢弃；
5. 默认把 408x408 patch 缩放到 224x224 保存，和 tools/data/crop_resize_30_patches.py 的裁剪方式一致。
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


# =========================
# 直接右键运行时，只改这里
# =========================

# 输入目录：按时间点保存的完整原始大图目录，下面应有 00/10/20/30/40 五个文件夹。
SOURCE_ROOT = Path(r"E:\workspaces\python\BlackTeaSimpleData\datas_test_point")

# 输出目录：会生成 train/val/test，每个 split 下再放 00/10/20/30/40 类别文件夹。
OUTPUT_ROOT = Path(r"E:\workspaces\python\BlackTeaSimpleData\datasets_01234_grid30_224")

# 只处理这 5 个时间点。ImageFolder 会按 00、10、20、30、40 排序，对应 0、1、2、3、4 类。
TIME_CODES = ("00", "10", "20", "30", "40")

# 每个时间点 24 张原图；按原图级别随机划分为 train/val/test = 15/4/5。
SPLIT_COUNTS = {"train": 15, "val": 4, "test": 5}

# 固定随机种子，保证每次运行得到同一套原图划分。
RANDOM_SEED = 2026

# 原图尺寸要求：固定网格裁剪默认针对 2448x2048 原图设计。
EXPECTED_WIDTH = 2448
EXPECTED_HEIGHT = 2048

# 固定网格裁剪参数：6 列 x 5 行，每个 patch 为 408x408。
PATCH_SIZE = 408
GRID_COLUMNS = 6
GRID_ROWS = 5
PATCHES_PER_SOURCE = GRID_COLUMNS * GRID_ROWS

# 实际参与裁剪的区域：2448x2040，底部 8 像素不进入任何 patch。
USED_WIDTH = PATCH_SIZE * GRID_COLUMNS
USED_HEIGHT = PATCH_SIZE * GRID_ROWS
DISCARDED_BOTTOM_PIXELS = EXPECTED_HEIGHT - USED_HEIGHT

# True：裁剪后缩放成 RESIZE_SIZE x RESIZE_SIZE；False：直接保存 408x408。
ENABLE_RESIZE = True
RESIZE_SIZE = 224

# 输出 jpg 质量。
JPEG_QUALITY = 95

# 如果输出目录已经存在且非空，默认直接报错，避免旧文件和新结果混在一起。
ALLOW_NONEMPTY_OUTPUT = False

# 支持的图像后缀。
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def ensure_clean_output_root(output_root: Path) -> None:
    """创建输出目录，并在目录非空时按配置决定是否停止。"""
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()) and not ALLOW_NONEMPTY_OUTPUT:
        raise FileExistsError(
            f"输出目录已存在且非空：{output_root}\n"
            "为避免混入旧结果，脚本已停止。请先手动清空该目录，"
            "或者把脚本顶部 ALLOW_NONEMPTY_OUTPUT 改为 True。"
        )


def list_source_images(time_code: str) -> list[Path]:
    """列出某个时间点目录下的原图，并检查数量是否为 24。"""
    folder = SOURCE_ROOT / time_code
    if not folder.is_dir():
        raise FileNotFoundError(f"时间点目录不存在：{folder}")

    images = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    expected_count = sum(SPLIT_COUNTS.values())
    if len(images) != expected_count:
        raise ValueError(f"时间点 {time_code} 应有 {expected_count} 张原图，实际为 {len(images)}。")
    return images


def build_source_split() -> list[dict[str, str]]:
    """按每个时间点的 24 张原图随机划分 15/4/5。"""
    rng = random.Random(RANDOM_SEED)
    rows: list[dict[str, str]] = []

    for time_code in TIME_CODES:
        images = list_source_images(time_code)
        rng.shuffle(images)
        start = 0

        for split_name, count in SPLIT_COUNTS.items():
            selected_images = images[start:start + count]
            for image_path in selected_images:
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
            start += count

    return rows


def validate_image_size(image_path: Path, width: int, height: int) -> None:
    """检查原图尺寸是否符合固定网格裁剪要求。"""
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise ValueError(
            f"原图尺寸异常：{image_path}\n"
            f"要求尺寸：{EXPECTED_WIDTH}x{EXPECTED_HEIGHT}，实际尺寸：{width}x{height}。\n"
            "固定 30 patch 裁剪依赖 2448x2048 原图，否则裁剪坐标含义会改变。"
        )

    if USED_WIDTH > width or USED_HEIGHT > height:
        raise ValueError(
            f"裁剪网格超过图像边界：{image_path}，"
            f"网格使用 {USED_WIDTH}x{USED_HEIGHT}，图像为 {width}x{height}。"
        )


def iter_grid_boxes() -> list[dict[str, int]]:
    """按行优先顺序生成 30 个固定网格裁剪框。"""
    boxes: list[dict[str, int]] = []
    patch_index = 1
    for row_index in range(GRID_ROWS):
        top = row_index * PATCH_SIZE
        for column_index in range(GRID_COLUMNS):
            left = column_index * PATCH_SIZE
            boxes.append(
                {
                    "patch_index": patch_index,
                    "row_index": row_index + 1,
                    "column_index": column_index + 1,
                    "left": left,
                    "top": top,
                    "right": left + PATCH_SIZE,
                    "bottom": top + PATCH_SIZE,
                }
            )
            patch_index += 1
    return boxes


def save_patch(patch: Image.Image, save_path: Path) -> None:
    """保存单个 patch，并统一使用较高质量的 JPEG 设置。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    patch.save(
        save_path,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=0,
        optimize=True,
    )


def crop_one_source(
    row: dict[str, str],
    crop_manifest_rows: list[dict[str, Any]],
    counts: dict[str, dict[str, int]],
) -> None:
    """对一张原图按固定 6x5 网格裁剪 30 个 patch，并保存到对应 split/time_code 目录。"""
    split_name = row["split"]
    time_code = row["time_code"]
    source_image_id = row["source_image_id"]
    source_path = Path(row["source_path"])
    output_dir = OUTPUT_ROOT / split_name / time_code

    with Image.open(source_path) as opened_image:
        image = ImageOps.exif_transpose(opened_image).convert("RGB")
        width, height = image.size
        validate_image_size(source_path, width, height)

        for box in iter_grid_boxes():
            patch = image.crop((box["left"], box["top"], box["right"], box["bottom"]))
            if ENABLE_RESIZE:
                patch = patch.resize((RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.LANCZOS)

            save_name = f"{source_image_id}__grid30_{box['patch_index']:02d}.jpg"
            save_path = output_dir / save_name
            save_patch(patch, save_path)

            counts[split_name][time_code] += 1
            crop_manifest_rows.append(
                {
                    "split": split_name,
                    "time_code": time_code,
                    "source_image_id": source_image_id,
                    "source_relpath": row["source_relpath"],
                    "patch_index": box["patch_index"],
                    "row_index": box["row_index"],
                    "column_index": box["column_index"],
                    "left": box["left"],
                    "top": box["top"],
                    "right": box["right"],
                    "bottom": box["bottom"],
                    "crop_size": PATCH_SIZE,
                    "resize_enabled": ENABLE_RESIZE,
                    "resize_size": RESIZE_SIZE if ENABLE_RESIZE else "",
                    "output_size": RESIZE_SIZE if ENABLE_RESIZE else PATCH_SIZE,
                    "target_relpath": save_path.relative_to(OUTPUT_ROOT).as_posix(),
                    "target_path": str(save_path.resolve()),
                }
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 CSV；rows 不能为空。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    source_split_rows: list[dict[str, str]],
    crop_manifest_rows: list[dict[str, Any]],
    counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """构建处理摘要，方便之后确认数据集来源和裁剪规则。"""
    return {
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "time_codes": list(TIME_CODES),
        "random_seed": RANDOM_SEED,
        "source_split_counts_per_time": SPLIT_COUNTS,
        "operation": "source_level_split_then_fixed_grid30_crop",
        "expected_source_size": [EXPECTED_WIDTH, EXPECTED_HEIGHT],
        "grid_columns": GRID_COLUMNS,
        "grid_rows": GRID_ROWS,
        "patches_per_source": PATCHES_PER_SOURCE,
        "crop_size": PATCH_SIZE,
        "used_width": USED_WIDTH,
        "used_height": USED_HEIGHT,
        "discarded_bottom_pixels": DISCARDED_BOTTOM_PIXELS,
        "resize_enabled": ENABLE_RESIZE,
        "resize_size": RESIZE_SIZE if ENABLE_RESIZE else None,
        "output_size": RESIZE_SIZE if ENABLE_RESIZE else PATCH_SIZE,
        "jpeg_quality": JPEG_QUALITY,
        "total_source_count": len(source_split_rows),
        "total_crop_count": len(crop_manifest_rows),
        "crop_counts": {split: dict(time_counts) for split, time_counts in counts.items()},
    }


def main() -> None:
    """执行原图级别 train/val/test 划分，并对每张原图做固定 30 patch 裁剪。"""
    ensure_clean_output_root(OUTPUT_ROOT)
    source_split_rows = build_source_split()
    crop_manifest_rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for image_index, row in enumerate(source_split_rows, start=1):
        crop_one_source(row, crop_manifest_rows, counts)
        print(
            f"[{image_index:03d}/{len(source_split_rows):03d}] "
            f"{row['source_relpath']} -> {PATCHES_PER_SOURCE} 个 grid patch"
        )

    write_csv(OUTPUT_ROOT / "source_split_manifest.csv", source_split_rows)
    write_csv(OUTPUT_ROOT / "grid30_crop_manifest.csv", crop_manifest_rows)

    summary = build_summary(source_split_rows, crop_manifest_rows, counts)
    with (OUTPUT_ROOT / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n原图级别划分 + 固定 30 patch 裁剪完成。")
    print(f"输入目录：{SOURCE_ROOT}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"裁剪方式：{GRID_COLUMNS}列 x {GRID_ROWS}行 = {PATCHES_PER_SOURCE} 张")
    print("裁剪坐标：x=0,408,816,1224,1632,2040；y=0,408,816,1224,1632")
    print(f"底部丢弃：{DISCARDED_BOTTOM_PIXELS} 像素")
    print(f"是否 resize：{ENABLE_RESIZE}")
    print(f"输出图片尺寸：{RESIZE_SIZE if ENABLE_RESIZE else PATCH_SIZE}x{RESIZE_SIZE if ENABLE_RESIZE else PATCH_SIZE}")
    print(f"原图数量：{len(source_split_rows)}")
    print(f"patch 数量：{len(crop_manifest_rows)}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name}: {dict(counts[split_name])}")


if __name__ == "__main__":
    main()
