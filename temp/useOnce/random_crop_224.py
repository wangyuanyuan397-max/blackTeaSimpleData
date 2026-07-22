"""从单张图片中随机裁剪 224x224 小图。"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent


# =========================
# 直接右键运行时，只改这里
# =========================

# 输入图片路径：把这里改成你要裁剪的那一张图片。
INPUT_IMAGE_PATH = r"D:\desktop\blackTea\0520\00-1.bmp"

# 输出文件夹：60 张随机裁剪图和 crop_metadata.csv 会保存到这里。
OUTPUT_DIR = r"E:\workspaces\python\BlackTeaSimpleData\temp\useOnce\out2"

# 随机裁剪数量。
NUM_CROPS = 55

# 每张小图的裁剪尺寸。
CROP_SIZE = 408

# 随机种子：固定后每次裁剪位置可复现；想每次都不一样可以换一个数字。
RANDOM_SEED = 2026

# 输出文件名前缀；None 表示自动使用原图文件名。
OUTPUT_PREFIX = None

# 输出图片格式，可选："jpg"、"png"、"bmp"。
OUTPUT_FORMAT = "jpg"

# 保存 jpg 时的质量。
JPG_QUALITY = 95


def make_default_output_dir() -> Path:
    """创建默认输出目录，避免多次运行互相覆盖。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / f"random_crops_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从一张图片中随机裁剪若干张 224x224 patch。")
    parser.add_argument("--image", default=INPUT_IMAGE_PATH, help="输入图片路径；默认读取代码顶部 INPUT_IMAGE_PATH。")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出文件夹；默认读取代码顶部 OUTPUT_DIR。")
    parser.add_argument("--num-crops", type=int, default=NUM_CROPS, help="随机裁剪数量。")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE, help="裁剪边长。")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子。")
    parser.add_argument("--prefix", default=OUTPUT_PREFIX, help="输出文件名前缀；None 表示使用原图文件名。")
    parser.add_argument("--format", choices=("jpg", "png", "bmp"), default=OUTPUT_FORMAT, help="输出图片格式。")
    parser.add_argument("--jpg-quality", type=int, default=JPG_QUALITY, help="保存 jpg 时的质量。")
    return parser.parse_args()


def random_crop_once(image: Image.Image, crop_size: int) -> tuple[Image.Image, int, int, int, int]:
    """随机裁剪一次，并返回裁剪图和坐标。"""
    width, height = image.size
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom)), left, top, right, bottom


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"输入图片不存在：{image_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else make_default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    crop_size = int(args.crop_size)

    if width < crop_size or height < crop_size:
        raise ValueError(
            f"图片尺寸为 {width}x{height}，小于裁剪尺寸 {crop_size}x{crop_size}，无法直接随机裁剪。"
        )

    prefix = args.prefix or image_path.stem
    extension = args.format.lower()
    metadata_rows = []

    for index in range(1, int(args.num_crops) + 1):
        patch, left, top, right, bottom = random_crop_once(image, crop_size)
        save_name = f"{prefix}_random_crop_{index:03d}.{extension}"
        save_path = output_dir / save_name
        if extension == "jpg":
            patch.save(save_path, quality=int(args.jpg_quality))
        else:
            patch.save(save_path)
        metadata_rows.append(
            {
                "index": index,
                "file_name": save_name,
                "source_image": str(image_path),
                "source_width": width,
                "source_height": height,
                "crop_size": crop_size,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            }
        )

    metadata_path = output_dir / "crop_metadata.csv"
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"输入图片：{image_path}")
    print(f"原图尺寸：{width}x{height}")
    print(f"裁剪数量：{args.num_crops}")
    print(f"裁剪尺寸：{crop_size}x{crop_size}")
    print(f"输出目录：{output_dir}")
    print(f"坐标记录：{metadata_path}")


if __name__ == "__main__":
    main()
