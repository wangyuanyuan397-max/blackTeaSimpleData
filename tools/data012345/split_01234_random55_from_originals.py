"""按原图级别划分 00/10/20/30/40，并随机裁剪 55 个 408x408 patch 后缩放到 224x224。"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps


# =========================
# 直接右键运行时，只改这里
# =========================

# 输入目录：按时间点保存的原始大图目录。
SOURCE_ROOT = Path(r"E:\workspaces\python\BlackTeaSimpleData\datas_test_point")

# 输出目录：会生成 train/val/test，每个 split 下再放 00/10/20/30/40 类别文件夹。
OUTPUT_ROOT = Path(r"E:\workspaces\python\BlackTeaSimpleData\datasets_01234")

# 只处理这 5 个时间点。ImageFolder 会按 00、10、20、30、40 排序，对应 0、1、2、3、4 类。
TIME_CODES = ("00", "10", "20", "30", "40")

# 每个时间点 24 张原图；按原图级别随机划分为 15/4/5。
SPLIT_COUNTS = {"train": 15, "val": 4, "test": 5}

# 每张原图随机裁剪 55 个 patch。
CROPS_PER_SOURCE = 55

# 先从原图随机裁 408x408，再缩放成 224x224 保存。
CROP_SIZE = 408
RESIZE_SIZE = 224

# 固定随机种子，保证每次运行得到同一套原图划分和同一组随机裁剪坐标。
RANDOM_SEED = 2026

# 输出 jpg 质量。
JPEG_QUALITY = 95

# 如果输出目录已存在且非空，默认直接报错，避免旧文件和新结果混在一起。
ALLOW_NONEMPTY_OUTPUT = False

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
    """列出某个时间点目录下的原图。"""
    folder = SOURCE_ROOT / time_code
    if not folder.is_dir():
        raise FileNotFoundError(f"时间点目录不存在：{folder}")
    images = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
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
            for image_path in images[start:start + count]:
                source_image_id = f"t{time_code}__{image_path.stem}"
                rows.append(
                    {
                        "split": split_name,
                        "time_code": time_code,
                        "source_image_id": source_image_id,
                        "source_stem": image_path.stem,
                        "source_relpath": image_path.relative_to(SOURCE_ROOT).as_posix(),
                        "source_path": str(image_path.resolve()),
                    }
                )
            start += count
    return rows


def crop_one_source(row: dict[str, str], crop_manifest_rows: list[dict[str, str]], counts: dict[str, dict[str, int]]) -> None:
    """对一张原图随机裁剪 55 次，并保存到对应 split/time_code 目录。"""
    split_name = row["split"]
    time_code = row["time_code"]
    source_image_id = row["source_image_id"]
    source_path = Path(row["source_path"])
    output_dir = OUTPUT_ROOT / split_name / time_code
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(source_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width < CROP_SIZE or height < CROP_SIZE:
        raise ValueError(f"原图 {source_path} 尺寸为 {width}x{height}，小于裁剪尺寸 {CROP_SIZE}。")

    # 每张原图使用独立随机流，保证以后即使处理顺序变化，同一原图的裁剪坐标仍稳定。
    source_seed = f"{RANDOM_SEED}_{source_image_id}"
    rng = random.Random(source_seed)

    for crop_index in range(1, CROPS_PER_SOURCE + 1):
        left = rng.randint(0, width - CROP_SIZE)
        top = rng.randint(0, height - CROP_SIZE)
        right = left + CROP_SIZE
        bottom = top + CROP_SIZE
        patch = image.crop((left, top, right, bottom)).resize((RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.BICUBIC)
        save_name = f"{source_image_id}__random55_{crop_index:03d}.jpg"
        save_path = output_dir / save_name
        patch.save(save_path, quality=JPEG_QUALITY)
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
                "resize_size": RESIZE_SIZE,
                "target_relpath": save_path.relative_to(OUTPUT_ROOT).as_posix(),
            }
        )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """写出 CSV；rows 不能为空。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_clean_output_root(OUTPUT_ROOT)
    source_split_rows = build_source_split()
    crop_manifest_rows: list[dict[str, str]] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in source_split_rows:
        crop_one_source(row, crop_manifest_rows, counts)

    write_csv(OUTPUT_ROOT / "source_split_manifest.csv", source_split_rows)
    write_csv(OUTPUT_ROOT / "random_crop_manifest.csv", crop_manifest_rows)
    summary = {
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "time_codes": list(TIME_CODES),
        "random_seed": RANDOM_SEED,
        "source_split_counts_per_time": SPLIT_COUNTS,
        "crops_per_source": CROPS_PER_SOURCE,
        "crop_size": CROP_SIZE,
        "resize_size": RESIZE_SIZE,
        "total_source_count": len(source_split_rows),
        "total_crop_count": len(crop_manifest_rows),
        "crop_counts": {split: dict(time_counts) for split, time_counts in counts.items()},
    }
    with (OUTPUT_ROOT / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("划分和随机裁剪完成。")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"原图数量：{len(source_split_rows)}")
    print(f"随机裁剪图数量：{len(crop_manifest_rows)}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name}: {dict(counts[split_name])}")


if __name__ == "__main__":
    main()
