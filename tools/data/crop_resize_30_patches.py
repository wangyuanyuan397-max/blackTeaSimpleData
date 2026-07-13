#!/usr/bin/env python3  # 使用当前 Python 环境执行本脚本。
"""将 datasets_split 中的每张图片裁成 30 个 patch，并缩放后保存到独立目录。"""  # 概括脚本的主要功能。

import argparse  # 用于解析来源目录、输出目录、JPEG 质量和预览模式等命令行参数。
from collections import Counter  # 用于统计各数据子集和类别中生成的 patch 数量。
from pathlib import Path  # 用于安全、清晰地处理 Windows 文件和目录路径。
from typing import Dict, List, Sequence, Tuple  # 用于明确标注函数参数和返回值中的集合类型。

from PIL import Image  # 用于读取原图、按坐标裁剪 patch、缩放并保存为 JPEG 图片。


EXPECTED_WIDTH = 2448  # 定义本任务要求的原始图片宽度，单位为像素。
EXPECTED_HEIGHT = 2048  # 定义本任务要求的原始图片高度，单位为像素。
PATCH_SIZE = 408  # 定义每个正方形裁剪区域的边长，单位为像素。
GRID_COLUMNS = 6  # 定义横向从左到右完整裁剪的 patch 列数。
GRID_ROWS = 5  # 定义纵向从上到下完整裁剪的 patch 行数。
DEFAULT_RESIZE_SIZE = 224  # 定义每个 patch 缩放后的默认宽度和高度。
PATCHES_PER_IMAGE = GRID_COLUMNS * GRID_ROWS  # 计算每张原图最终产生的 patch 数量，即 6×5=30。
USED_WIDTH = PATCH_SIZE * GRID_COLUMNS  # 计算横向裁剪实际使用的宽度，即 408×6=2448 像素。
USED_HEIGHT = PATCH_SIZE * GRID_ROWS  # 计算纵向裁剪实际使用的高度，即 408×5=2040 像素。
DISCARDED_BOTTOM_PIXELS = EXPECTED_HEIGHT - USED_HEIGHT  # 计算并记录原图底部被丢弃的 8 个像素。
SUPPORTED_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}  # 定义脚本能够读取的常见图片扩展名。


def parse_arguments() -> argparse.Namespace:  # 创建并解析本脚本支持的全部命令行参数。
    project_root = Path(__file__).resolve().parents[2]  # 根据 tools/data/脚本名.py 的位置自动确定项目根目录。
    default_source = project_root / "datasets_split"  # 默认读取项目根目录下已经划分好的 datasets_split 数据集。
    default_output = project_root / "datasets_split_patches"  # 默认把全部 patch 保存到独立的 datasets_split_patches 目录。
    parser = argparse.ArgumentParser(  # 创建带有中文功能说明的命令行参数解析器。
        description="把每张 2448×2048 图片裁成 6×5 个 408×408 patch，并缩放后保存。"  # 在 --help 输出中说明脚本用途。
    )  # 结束命令行参数解析器的创建过程。
    parser.add_argument(  # 添加用于指定待处理数据集根目录的参数。
        "--source-root",  # 定义覆盖默认来源目录时使用的参数名称。
        type=Path,  # 把用户输入的路径字符串自动转换成 Path 对象。
        default=default_source,  # 未指定该参数时读取项目中的 datasets_split 文件夹。
        help=f"待裁剪数据集目录（默认：{default_source}）。",  # 在帮助信息中展示默认来源路径。
    )  # 结束来源目录参数的定义。
    parser.add_argument(  # 添加用于指定新数据集保存位置的参数。
        "--output-root",  # 定义覆盖默认输出目录时使用的参数名称。
        type=Path,  # 把用户输入的输出路径字符串自动转换成 Path 对象。
        default=default_output,  # 未指定该参数时输出到项目中的 datasets_split_patches 文件夹。
        help=f"裁剪缩放后数据集的输出目录（默认：{default_output}）。",  # 在帮助信息中展示默认输出路径。
    )  # 结束输出目录参数的定义。
    parser.add_argument(  # 添加用于调整每个 patch 最终缩放尺寸的参数。
        "--resize-size",  # 定义在命令行中修改输出边长时使用的参数名称。
        type=int,  # 要求缩放后的正方形边长必须是整数。
        default=DEFAULT_RESIZE_SIZE,  # 默认把每个 408×408 patch 缩放成 224×224。
        help=f"输出 patch 的正方形边长（默认：{DEFAULT_RESIZE_SIZE}）。",  # 在帮助信息中说明默认输出尺寸。
    )  # 结束缩放尺寸参数的定义。
    parser.add_argument(  # 添加用于调整输出 JPEG 图片质量的参数。
        "--jpeg-quality",  # 定义在命令行中修改 JPEG 保存质量时使用的参数名称。
        type=int,  # 要求 JPEG 质量必须使用整数表示。
        default=95,  # 默认使用 95，在图像质量和文件大小之间取得较好平衡。
        help="输出 JPEG 质量，范围为 1～100（默认：95）。",  # 在帮助信息中说明有效范围和默认值。
    )  # 结束 JPEG 质量参数的定义。
    parser.add_argument(  # 添加允许替换已有目标 patch 的显式开关。
        "--overwrite",  # 定义允许覆盖已有输出文件时使用的命令行开关。
        action="store_true",  # 当命令中出现 --overwrite 时把参数值设置为 True。
        help="允许覆盖已存在的目标 patch；默认遇到已有文件就停止。",  # 在帮助信息中说明默认的文件保护行为。
    )  # 结束覆盖模式参数的定义。
    parser.add_argument(  # 添加只检查和统计而不真正生成 patch 的安全预览开关。
        "--dry-run",  # 定义启用预览模式时使用的命令行开关。
        action="store_true",  # 当命令中出现 --dry-run 时把参数值设置为 True。
        help="仅检查图片尺寸并显示统计，不创建目录或保存 patch。",  # 在帮助信息中说明预览模式不会修改文件。
    )  # 结束预览模式参数的定义。
    return parser.parse_args()  # 解析当前命令行并返回包含全部参数值的对象。


def collect_image_paths(source_root: Path) -> List[Path]:  # 递归收集来源数据集中的全部受支持图片。
    image_paths = sorted(  # 按稳定顺序创建完整的来源图片路径列表。
        path.resolve()  # 将每张图片的路径转换成规范化的绝对路径。
        for path in source_root.rglob("*")  # 递归遍历来源根目录下的全部文件和文件夹。
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS  # 只保留扩展名受支持的普通图片文件。
    )  # 结束来源图片路径列表的构建。
    if not image_paths:  # 确认来源目录中至少找到了一张受支持的图片。
        raise ValueError(f"来源目录中没有找到受支持的图片：{source_root}")  # 没有输入图片时使用明确错误停止运行。
    return image_paths  # 返回按路径排序后的全部来源图片列表。


def build_output_paths(  # 为一张来源图片构建 30 个目标 patch 路径。
    image_path: Path,  # 接收当前待裁剪图片的绝对路径。
    source_root: Path,  # 接收用于计算相对目录结构的来源根目录。
    output_root: Path,  # 接收独立的新数据集输出根目录。
) -> List[Path]:  # 返回当前图片对应的 30 个 JPEG 目标路径。
    relative_parent = image_path.relative_to(source_root).parent  # 保留 train/类别、val/类别或 test/类别等相对目录结构。
    destination_directory = output_root / relative_parent  # 把原相对目录结构拼接到新的输出根目录下。
    return [  # 按 patch 编号从 01 到 30 构建全部目标文件路径。
        destination_directory / f"{image_path.stem}_patch_{patch_id:02d}.jpg"  # 使用“原文件名_patch_两位编号.jpg”作为输出文件名。
        for patch_id in range(1, PATCHES_PER_IMAGE + 1)  # 依次生成 1～30 的 patch 编号。
    ]  # 结束当前来源图片目标路径列表的构建。


def validate_images_and_destinations(  # 在写入任何文件之前检查原图尺寸、命名冲突和已有目标文件。
    image_paths: Sequence[Path],  # 接收全部待处理来源图片路径。
    source_root: Path,  # 接收来源数据集根目录以计算相对路径。
    output_root: Path,  # 接收新的 patch 数据集输出根目录。
    overwrite: bool,  # 接收是否明确允许覆盖已有目标文件的设置。
) -> None:  # 本函数只负责验证，成功时不需要返回额外数据。
    destination_to_source: Dict[Path, Path] = {}  # 记录每个目标路径对应的来源图片，用于检测命名冲突。
    invalid_sizes: List[Tuple[Path, Tuple[int, int]]] = []  # 保存尺寸不是 2448×2048 的来源图片及其实际尺寸。
    existing_destinations: List[Path] = []  # 保存默认保护模式下已经存在的目标 patch 路径。
    for image_path in image_paths:  # 依次检查每一张待处理的来源图片。
        with Image.open(image_path) as image:  # 以自动关闭文件句柄的方式读取当前图片元数据。
            actual_size = image.size  # 按照 PIL 的“宽度、高度”顺序取得当前图片尺寸。
        if actual_size != (EXPECTED_WIDTH, EXPECTED_HEIGHT):  # 判断当前图片尺寸是否严格等于要求的 2448×2048。
            invalid_sizes.append((image_path, actual_size))  # 记录不符合裁剪规则的图片，稍后统一报告。
        for destination_path in build_output_paths(image_path, source_root, output_root):  # 检查当前图片的全部 30 个目标路径。
            previous_source = destination_to_source.get(destination_path)  # 查询是否已有另一张来源图片占用了相同目标路径。
            if previous_source is not None and previous_source != image_path:  # 判断两个不同来源是否会产生同名目标 patch。
                raise ValueError(f"目标文件名冲突：{previous_source} 和 {image_path} -> {destination_path}")  # 在写入前停止以防静默覆盖。
            destination_to_source[destination_path] = image_path  # 为当前来源图片登记并保留该目标 patch 路径。
            if destination_path.exists() and not overwrite:  # 默认检查并保护以前已经生成的同名 patch。
                existing_destinations.append(destination_path)  # 记录已有目标文件并推迟到完整检查后统一报错。
    if invalid_sizes:  # 判断是否发现任何不满足 2448×2048 要求的来源图片。
        preview = "\n".join(f"{path}：{size[0]}×{size[1]}" for path, size in invalid_sizes[:10])  # 最多展示前十张异常图片和尺寸。
        raise ValueError(f"发现 {len(invalid_sizes)} 张尺寸异常图片，要求为 2448×2048：\n{preview}")  # 停止运行以避免裁剪越界或产生黑边。
    if existing_destinations:  # 判断默认保护模式下是否发现任何已有目标 patch。
        preview = "\n".join(str(path) for path in existing_destinations[:10])  # 最多展示前十个已有文件路径以控制错误长度。
        raise FileExistsError(  # 停止运行而不是覆盖可能属于此前处理结果的图片。
            f"输出目录中已有 {len(existing_destinations)} 个目标 patch，请更换输出目录或使用 --overwrite。\n{preview}"  # 给出冲突数量和解决方法。
        )  # 结束已有目标文件异常的构建。


def print_plan(image_paths: Sequence[Path], source_root: Path, resize_size: int) -> None:  # 输出来源数据和预计生成结果的统计信息。
    relative_group_counts = Counter(  # 统计 train/pre、val/slight 等各相对父目录中的原图数量。
        str(image_path.relative_to(source_root).parent)  # 使用当前图片相对于来源根目录的父目录作为统计键。
        for image_path in image_paths  # 依次读取每张来源图片以生成分组统计。
    )  # 结束相对目录原图数量统计的构建。
    print("\n处理计划")  # 使用中文标题将计划统计与其他日志清晰分隔。
    print(f"原图数量：{len(image_paths)}")  # 显示本次将要处理的来源图片总数。
    print(f"单图裁剪：{GRID_COLUMNS} 列 × {GRID_ROWS} 行 = {PATCHES_PER_IMAGE} 个 patch")  # 显示每张图片的裁剪网格和 patch 数量。
    print(f"裁剪尺寸：{PATCH_SIZE}×{PATCH_SIZE}")  # 显示每个原始 patch 的宽度和高度。
    print(f"缩放尺寸：{resize_size}×{resize_size}")  # 显示每个 patch 最终保存时的宽度和高度。
    print(f"底部丢弃：{DISCARDED_BOTTOM_PIXELS} 像素")  # 明确显示原图底部不会进入任何 patch 的像素行数。
    print(f"预计生成：{len(image_paths) * PATCHES_PER_IMAGE} 张 JPEG patch")  # 显示整个数据集预计产生的目标图片总数。
    print("各目录预计生成数量：")  # 输出保留原数据层级后的分组统计标题。
    for relative_group, source_count in sorted(relative_group_counts.items()):  # 按相对目录名称稳定输出每组统计结果。
        print(f"  {relative_group}：{source_count * PATCHES_PER_IMAGE}")  # 显示当前数据子集和类别将生成的 patch 数量。


def crop_resize_and_save(  # 将单张原图裁成 30 个 patch、缩放并保存为 JPEG。
    image_path: Path,  # 接收当前待处理来源图片的绝对路径。
    destination_paths: Sequence[Path],  # 接收与 30 个 patch 编号一一对应的目标文件路径。
    resize_size: int,  # 接收每个输出正方形 patch 的目标边长。
    jpeg_quality: int,  # 接收保存 JPEG 图片时使用的质量参数。
) -> None:  # 本函数通过写入图片完成工作，不需要返回额外数据。
    with Image.open(image_path) as opened_image:  # 以自动关闭文件句柄的方式打开当前来源图片。
        rgb_image = opened_image.convert("RGB")  # 统一转换成三通道 RGB，确保能够稳定保存为 JPEG。
        for row_index in range(GRID_ROWS):  # 按从上到下的顺序依次处理 5 行 patch。
            top = row_index * PATCH_SIZE  # 计算当前行裁剪框上边界的 y 坐标。
            for column_index in range(GRID_COLUMNS):  # 按从左到右的顺序依次处理当前行的 6 列 patch。
                left = column_index * PATCH_SIZE  # 计算当前列裁剪框左边界的 x 坐标。
                right = left + PATCH_SIZE  # 计算当前裁剪框右边界的 x 坐标。
                bottom = top + PATCH_SIZE  # 计算当前裁剪框下边界的 y 坐标。
                patch = rgb_image.crop((left, top, right, bottom))  # 使用 PIL 的左、上、右、下坐标裁出 408×408 区域。
                resized_patch = patch.resize(  # 使用高质量重采样算法把当前 patch 缩放到目标尺寸。
                    (resize_size, resize_size),  # 指定输出 patch 的宽度和高度相同。
                    resample=Image.Resampling.LANCZOS,  # 使用 LANCZOS 滤波尽量保留缩小后的纹理细节。
                )  # 结束当前 patch 的高质量缩放操作。
                patch_id = row_index * GRID_COLUMNS + column_index  # 计算当前 patch 在零起始目标路径列表中的位置。
                destination_path = destination_paths[patch_id]  # 根据行优先顺序取得当前 patch 对应的目标文件路径。
                destination_path.parent.mkdir(parents=True, exist_ok=True)  # 根据需要递归创建 train/类别等目标目录。
                resized_patch.save(  # 把完成裁剪和缩放的 RGB patch 保存成 JPEG 文件。
                    destination_path,  # 指定当前 patch 的完整目标文件路径。
                    format="JPEG",  # 显式指定使用 JPEG 编码格式保存图片。
                    quality=jpeg_quality,  # 使用用户设置的 JPEG 图片质量。
                    subsampling=0,  # 禁用色度下采样以尽量保留颜色和细节信息。
                    optimize=True,  # 优化 JPEG 编码以在不降低质量的前提下减小文件体积。
                )  # 结束当前 patch 的 JPEG 保存操作。


def process_dataset(  # 按顺序处理数据集中的全部来源图片并显示进度。
    image_paths: Sequence[Path],  # 接收已经完成尺寸和冲突检查的全部来源图片。
    source_root: Path,  # 接收来源数据集根目录以保留相对目录结构。
    output_root: Path,  # 接收新 patch 数据集的独立输出根目录。
    resize_size: int,  # 接收每个输出 patch 的正方形目标边长。
    jpeg_quality: int,  # 接收 JPEG 图片保存质量。
) -> None:  # 本函数通过批量写入图片完成处理，不需要返回额外数据。
    total_images = len(image_paths)  # 记录来源图片总数以便显示当前处理进度。
    for image_index, image_path in enumerate(image_paths, start=1):  # 从 1 开始依次处理并编号每张来源图片。
        destination_paths = build_output_paths(image_path, source_root, output_root)  # 计算当前图片对应的 30 个目标路径。
        crop_resize_and_save(image_path, destination_paths, resize_size, jpeg_quality)  # 完成当前图片的裁剪、缩放和保存。
        relative_path = image_path.relative_to(source_root)  # 计算适合在进度日志中展示的简短相对路径。
        print(f"[{image_index}/{total_images}] 已处理 {relative_path}，生成 {PATCHES_PER_IMAGE} 个 patch。")  # 显示当前进度和处理结果。


def main() -> None:  # 组织参数解析、输入验证、计划显示和批量处理的完整流程。
    args = parse_arguments()  # 从当前命令行读取目录、缩放尺寸、JPEG 质量和运行模式。
    source_root = args.source_root.expanduser().resolve()  # 展开并规范化来源数据集目录的绝对路径。
    output_root = args.output_root.expanduser().resolve()  # 展开并规范化新 patch 数据集目录的绝对路径。
    if not source_root.is_dir():  # 在递归搜索图片前确认来源根目录存在且为文件夹。
        raise FileNotFoundError(f"来源目录不存在：{source_root}")  # 使用明确错误指出无法找到的来源路径。
    if output_root == source_root or source_root in output_root.parents:  # 禁止把输出目录放进来源目录内部以免以后重复处理输出 patch。
        raise ValueError("输出目录不能等于来源目录，也不能位于来源目录内部。")  # 提醒用户使用默认的同级独立目录。
    if args.resize_size <= 0:  # 检查缩放后的 patch 边长是否为有效正整数。
        raise ValueError("--resize-size 必须是大于 0 的整数。")  # 对无效缩放尺寸给出直接且可操作的错误信息。
    if not 1 <= args.jpeg_quality <= 100:  # 检查 JPEG 质量是否位于 Pillow 接受的合理范围内。
        raise ValueError("--jpeg-quality 必须位于 1～100 之间。")  # 对无效 JPEG 质量给出明确范围提示。
    image_paths = collect_image_paths(source_root)  # 递归收集并排序来源数据集中的全部图片。
    validate_images_and_destinations(image_paths, source_root, output_root, args.overwrite)  # 在写入前完成尺寸、重名和覆盖检查。
    print_plan(image_paths, source_root, args.resize_size)  # 输出本次处理的输入数量、裁剪规则和预计结果。
    if args.dry_run:  # 判断用户是否只希望安全预览而不真正生成 patch。
        print("\n预览模式：未创建目录，也未保存任何 patch。")  # 明确告知本次运行没有修改文件系统。
        return  # 结束预览运行并跳过后续批量图片处理。
    process_dataset(image_paths, source_root, output_root, args.resize_size, args.jpeg_quality)  # 执行全部图片的裁剪、缩放和保存。
    print(f"\n处理完成，共生成 {len(image_paths) * PATCHES_PER_IMAGE} 张 patch。")  # 输出最终生成的 patch 总数量。
    print(f"新数据集目录：{output_root}")  # 输出新数据集所在的绝对目录路径。


if __name__ == "__main__":  # 仅当直接运行本文件时执行主流程，被其他模块导入时不自动处理图片。
    main()  # 启动完整的数据集裁剪、缩放和保存流程。
