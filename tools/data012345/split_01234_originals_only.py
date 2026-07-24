"""只按原图级别划分 00/10/20/30/40，不裁剪、不缩放。

这个脚本和 split_01234_random55_from_originals.py 使用同一套核心划分逻辑：
每个时间点 24 张原图，按原图级别随机划分为 train/val/test = 15/4/5。

区别是：
1. 本脚本只复制原图到 train/val/test 目录；
2. 不做 408x408 随机裁剪；
3. 不做 224x224 resize；
4. 不改变图像像素内容。
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


# =========================
# 直接右键运行时，只改这里
# =========================

# 输入目录：按时间点保存原始大图，下面应有 00/10/20/30/40 五个文件夹。
SOURCE_ROOT = Path(r"E:\workspaces\python\BlackTeaSimpleData\datas_test_point")

# 输出目录：会生成 train/val/test，每个 split 下再放 00/10/20/30/40 类别文件夹。
OUTPUT_ROOT = Path(r"E:\workspaces\python\BlackTeaSimpleData\datasets_01234_original_split")

# 只处理这 5 个时间点。ImageFolder 会按 00、10、20、30、40 排序，对应 0、1、2、3、4 类。
TIME_CODES = ("00", "10", "20", "30", "40")

# 每个时间点 24 张原图；按原图级别随机划分为 train/val/test = 15/4/5。
SPLIT_COUNTS = {"train": 15, "val": 4, "test": 5}

# 固定随机种子，保证每次运行得到同一套原图划分。
RANDOM_SEED = 2026

# 如果输出目录已经存在且非空，默认直接报错，避免旧结果和新结果混在一起。
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
        path for path in folder.iterdir()
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


def copy_one_source(row: dict[str, str], copied_rows: list[dict[str, str]], counts: dict[str, dict[str, int]]) -> None:
    """把一张原图复制到对应 split/time_code 目录，不做任何图像处理。"""
    split_name = row["split"]
    time_code = row["time_code"]
    source_path = Path(row["source_path"])
    output_dir = OUTPUT_ROOT / split_name / time_code
    output_dir.mkdir(parents=True, exist_ok=True)

    target_path = output_dir / source_path.name
    if target_path.exists():
        raise FileExistsError(f"目标文件已存在，可能存在重名原图：{target_path}")

    shutil.copy2(source_path, target_path)
    counts[split_name][time_code] += 1

    copied_row = dict(row)
    copied_row["target_relpath"] = target_path.relative_to(OUTPUT_ROOT).as_posix()
    copied_row["target_path"] = str(target_path.resolve())
    copied_rows.append(copied_row)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """写出 CSV，rows 不能为空。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """执行原图级别 train/val/test 划分，并复制原图。"""
    ensure_clean_output_root(OUTPUT_ROOT)
    source_split_rows = build_source_split()
    copied_rows: list[dict[str, str]] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in source_split_rows:
        copy_one_source(row, copied_rows, counts)

    write_csv(OUTPUT_ROOT / "source_split_manifest.csv", source_split_rows)
    write_csv(OUTPUT_ROOT / "copied_originals_manifest.csv", copied_rows)

    summary = {
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "time_codes": list(TIME_CODES),
        "random_seed": RANDOM_SEED,
        "source_split_counts_per_time": SPLIT_COUNTS,
        "operation": "copy_originals_only_no_crop_no_resize",
        "total_source_count": len(source_split_rows),
        "total_copied_count": len(copied_rows),
        "copy_counts": {split: dict(time_counts) for split, time_counts in counts.items()},
    }
    with (OUTPUT_ROOT / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("原图级别划分完成：未裁剪，未缩放。")
    print(f"输入目录：{SOURCE_ROOT}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"原图数量：{len(source_split_rows)}")
    print(f"复制数量：{len(copied_rows)}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name}: {dict(counts[split_name])}")


if __name__ == "__main__":
    main()
