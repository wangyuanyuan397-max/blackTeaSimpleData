"""使用 512×512 窗口和 256 像素步长批量裁剪图片。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from PIL import Image


# ---------------------------------------------------------------------------
# PyCharm 右键运行配置
# ---------------------------------------------------------------------------
# 可以填写单张图片路径，也可以填写包含多张图片的文件夹路径。
# 示例：Path(r"E:\workspaces\python\BlackTeaSimpleData\example.bmp")
# 示例：Path(r"E:\workspaces\python\BlackTeaSimpleData\datasets\train")
INPUT_PATH = Path(r"D:\desktop\blackTea\00\2-1.bmp")

# 所有 patch 都保存在这个目录下；可以根据需要自行修改。
OUTPUT_ROOT = Path(
    r"D:\desktop\咩咩文档\绘图资料\data-spli"
)

# 是否递归搜索输入文件夹下面的多级子文件夹。
RECURSIVE = True

# 输出扩展名使用 PNG，能够无损保存 patch；如需 BMP 可改为 ".bmp"。
OUTPUT_EXTENSION = ".png"

# True 表示重复运行时覆盖同名 patch；False 表示跳过已经存在的 patch。
OVERWRITE_EXISTING = True

# 固定滑动窗口参数。
WINDOW_WIDTH = 512
WINDOW_HEIGHT = 512
STRIDE_X = 256
STRIDE_Y = 256

# 严格步长模式下，边缘不足 512 像素的区域会被舍弃。
# 如果改成 True，会额外添加一个贴住右边缘/下边缘的完整窗口；
# 但最后一步的实际位移可能小于 256，因此默认保持 False。
INCLUDE_EDGE_ALIGNED_WINDOW = False

# 支持读取的图片格式。
SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def normalize_output_extension(extension: str) -> str:
    """把 png 或 .png 统一规范为小写的 .png。"""
    normalized = str(extension).strip().lower()
    if not normalized:
        raise ValueError("输出扩展名不能为空。")
    return normalized if normalized.startswith(".") else f".{normalized}"


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    """收集单张输入图片，或按稳定顺序收集目录中的所有图片。"""
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的图片格式：{input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(
            f"输入路径不存在：{input_path}\n"
            "请修改脚本顶部 INPUT_PATH，或者通过 --input 指定路径。"
        )

    iterator: Iterable[Path]
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise ValueError(f"输入文件夹中没有找到支持的图片：{input_path}")
    return images


def generate_start_positions(
    image_length: int,
    window_length: int,
    stride: int,
    include_edge_aligned_window: bool,
) -> list[int]:
    """生成某一方向的窗口起点；默认只接受严格固定步长的完整窗口。"""
    if image_length < window_length:
        return []
    positions = list(range(0, image_length - window_length + 1, stride))
    edge_position = image_length - window_length
    if (
        include_edge_aligned_window
        and positions
        and positions[-1] != edge_position
    ):
        positions.append(edge_position)
    return positions


def resolve_image_output_directory(
    image_path: Path,
    input_path: Path,
    output_root: Path,
) -> Path:
    """为每张原图建立独立目录，并保留输入目录的相对层级以避免重名。"""
    if input_path.is_file():
        relative_parent = Path()
    else:
        relative_parent = image_path.parent.relative_to(input_path)
    return output_root / relative_parent / image_path.stem


def save_patch(
    patch: Image.Image,
    save_path: Path,
    overwrite_existing: bool,
) -> bool:
    """保存一个 patch；返回 True 表示本次实际写入了文件。"""
    if save_path.exists() and not overwrite_existing:
        return False
    save_path.parent.mkdir(parents=True, exist_ok=True)
    patch.save(save_path)
    return True


def split_one_image(
    image_path: Path,
    input_path: Path,
    output_root: Path,
    output_extension: str,
    include_edge_aligned_window: bool,
    overwrite_existing: bool,
    dry_run: bool,
) -> tuple[list[dict[str, object]], int]:
    """裁剪一张图片，返回 patch 清单以及本次实际写入数量。"""
    with Image.open(image_path) as opened_image:
        # 统一转成 RGB，避免调色板、灰度或透明通道导致不同格式保存行为不一致。
        image = opened_image.convert("RGB")
        image_width, image_height = image.size

        x_positions = generate_start_positions(
            image_width,
            WINDOW_WIDTH,
            STRIDE_X,
            include_edge_aligned_window,
        )
        y_positions = generate_start_positions(
            image_height,
            WINDOW_HEIGHT,
            STRIDE_Y,
            include_edge_aligned_window,
        )
        if not x_positions or not y_positions:
            print(
                f"跳过尺寸不足的图片：{image_path}，"
                f"尺寸={image_width}×{image_height}"
            )
            return [], 0

        image_output_directory = resolve_image_output_directory(
            image_path,
            input_path,
            output_root,
        )
        manifest_rows: list[dict[str, object]] = []
        written_count = 0
        patch_index = 0

        # 先按纵向逐行移动，再在每行内从左向右移动。
        for row_index, top in enumerate(y_positions, start=1):
            for column_index, left in enumerate(x_positions, start=1):
                patch_index += 1
                right = left + WINDOW_WIDTH
                bottom = top + WINDOW_HEIGHT
                patch_filename = (
                    f"{image_path.stem}_patch_{patch_index:03d}_"
                    f"r{row_index:02d}_c{column_index:02d}_"
                    f"x{left}_y{top}{output_extension}"
                )
                patch_path = image_output_directory / patch_filename

                if not dry_run:
                    patch = image.crop((left, top, right, bottom))
                    if save_patch(patch, patch_path, overwrite_existing):
                        written_count += 1

                manifest_rows.append(
                    {
                        "source_image": str(image_path),
                        "patch_path": str(patch_path),
                        "patch_index": patch_index,
                        "row": row_index,
                        "column": column_index,
                        "left_x": left,
                        "top_y": top,
                        "right_x": right,
                        "bottom_y": bottom,
                        "window_width": WINDOW_WIDTH,
                        "window_height": WINDOW_HEIGHT,
                        "stride_x": STRIDE_X,
                        "stride_y": STRIDE_Y,
                        "source_width": image_width,
                        "source_height": image_height,
                    }
                )

    return manifest_rows, written_count


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> Path:
    """保存所有 patch 的来源、文件名与裁剪坐标，方便后续核查。"""
    manifest_path = output_root / "patch_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 512×512 窗口、256 步长裁剪并保存所有完整 patch。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="单张图片或图片文件夹；默认读取脚本顶部 INPUT_PATH。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT,
        help="patch 输出根目录。",
    )
    parser.add_argument(
        "--output-extension",
        default=OUTPUT_EXTENSION,
        help="输出格式，例如 .png 或 .bmp。",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="输入为文件夹时只搜索当前层，不搜索子文件夹。",
    )
    parser.add_argument(
        "--include-edge-window",
        action="store_true",
        default=INCLUDE_EDGE_ALIGNED_WINDOW,
        help="额外加入贴住右/下边缘的完整窗口；最后一步可能不足 256。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=not OVERWRITE_EXISTING,
        help="跳过已存在的同名 patch，不进行覆盖。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算 patch 数量和坐标，不创建任何输出文件。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_path = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    output_extension = normalize_output_extension(args.output_extension)
    recursive = bool(RECURSIVE and not args.no_recursive)
    image_paths = collect_images(input_path, recursive=recursive)

    print(f"输入路径：{input_path}")
    print(f"输出路径：{output_root}")
    print(
        f"窗口：{WINDOW_WIDTH}×{WINDOW_HEIGHT}，"
        f"步长：{STRIDE_X}×{STRIDE_Y}"
    )
    print(f"待处理图片：{len(image_paths)} 张")

    all_rows: list[dict[str, object]] = []
    total_written = 0
    for image_index, image_path in enumerate(image_paths, start=1):
        rows, written_count = split_one_image(
            image_path=image_path,
            input_path=input_path,
            output_root=output_root,
            output_extension=output_extension,
            include_edge_aligned_window=bool(args.include_edge_window),
            overwrite_existing=not bool(args.skip_existing),
            dry_run=bool(args.dry_run),
        )
        all_rows.extend(rows)
        total_written += written_count
        print(
            f"[{image_index}/{len(image_paths)}] {image_path.name}: "
            f"patch={len(rows)}"
        )

    if not all_rows:
        raise RuntimeError("没有生成任何 patch；请检查图片尺寸是否至少为 512×512。")

    if args.dry_run:
        print(f"dry-run 完成：预计生成 {len(all_rows)} 个 patch，未写入文件。")
        return

    manifest_path = write_manifest(output_root, all_rows)
    skipped_count = len(all_rows) - total_written
    print(f"处理完成：共规划 {len(all_rows)} 个 patch。")
    print(f"本次实际写入：{total_written} 个；跳过已有：{skipped_count} 个。")
    print(f"坐标清单：{manifest_path}")


if __name__ == "__main__":
    main()
