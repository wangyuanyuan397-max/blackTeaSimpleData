#!/usr/bin/env python3
"""按时刻整理原始大图，复用既有逻辑裁成 30 块并生成可追溯清单。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

try:
    from .crop_resize_30_patches import (
        DISCARDED_BOTTOM_PIXELS,
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
        GRID_COLUMNS,
        GRID_ROWS,
        PATCHES_PER_IMAGE,
        PATCH_SIZE,
        crop_resize_and_save,
    )
except ImportError:
    from crop_resize_30_patches import (  # type: ignore[no-redef]
        DISCARDED_BOTTOM_PIXELS,
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
        GRID_COLUMNS,
        GRID_ROWS,
        PATCHES_PER_IMAGE,
        PATCH_SIZE,
        crop_resize_and_save,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "datas_test_point"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datas_test_point_30_patches"
DEFAULT_RESIZE_SIZE = 224
DEFAULT_JPEG_QUALITY = 95
CROP_VERSION = "timepoint_6x5_408_to_224_v1"

TIME_TO_SEVERITY = {
    "00": "pre",
    "05": "pre",
    "10": "pre",
    "15": "slight",
    "20": "slight",
    "25": "slight",
    "30": "moderate",
    "35": "moderate",
    "40": "moderate",
    "45": "moderate",
    "50": "over",
    "55": "over",
    "60": "over",
}
TIME_CODES = tuple(TIME_TO_SEVERITY)
EXPECTED_STEMS = tuple(
    f"{name_part_1}-{name_part_2}"
    for name_part_1 in range(1, 7)
    for name_part_2 in range(1, 5)
)
KNOWN_EXCLUSIONS = {
    ("55", "6-2"): "confirmed_anomalous_sample",
}
STEM_PATTERN = re.compile(r"^(?P<part1>[1-6])-(?P<part2>[1-4])$")
CONTROL_FILENAMES = {
    "source_manifest.csv",
    "patch_manifest.csv",
    "excluded_samples.csv",
    "preparation_summary.json",
}


@dataclass(frozen=True)
class SourceImage:
    path: Path
    source_relpath: str
    source_image_id: str
    source_stem: str
    name_part_1: int
    name_part_2: int
    time_code: str
    time_h: float
    severity: str
    width: int
    height: int
    mode: str


@dataclass(frozen=True)
class PatchPlan:
    source: SourceImage
    destination: Path
    patch_relpath: str
    patch_index: int
    grid_row: int
    grid_column: int
    left: int
    top: int
    right: int
    bottom: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "把 datas_test_point 中按时刻保存的 2448×2048 原图裁成 "
            "6×5 个 patch，并生成来源与坐标清单。"
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resize-size",
        type=int,
        default=DEFAULT_RESIZE_SIZE,
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖当前计划中的已有 patch 和清单；不会删除未知文件。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="完成全部输入、尺寸、命名和目标检查，但不创建任何文件。",
    )
    return parser.parse_args()


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def inspect_source_image(
    image_path: Path,
    source_root: Path,
    time_code: str,
) -> SourceImage:
    match = STEM_PATTERN.fullmatch(image_path.stem)
    if match is None:
        raise ValueError(
            f"文件名不符合 <1-6>-<1-4>.bmp：{image_path}"
        )
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        mode = str(image.mode)
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise ValueError(
            f"原图尺寸异常：{image_path}，实际={width}×{height}，"
            f"要求={EXPECTED_WIDTH}×{EXPECTED_HEIGHT}"
        )
    source_stem = image_path.stem
    return SourceImage(
        path=image_path.resolve(),
        source_relpath=relative_posix(image_path, source_root),
        source_image_id=f"t{time_code}__{source_stem}",
        source_stem=source_stem,
        name_part_1=int(match.group("part1")),
        name_part_2=int(match.group("part2")),
        time_code=time_code,
        time_h=int(time_code) / 10.0,
        severity=TIME_TO_SEVERITY[time_code],
        width=width,
        height=height,
        mode=mode,
    )


def collect_source_images(
    source_root: Path,
) -> tuple[list[SourceImage], list[dict[str, Any]]]:
    actual_directories = {
        path.name: path
        for path in source_root.iterdir()
        if path.is_dir()
    }
    missing_directories = sorted(set(TIME_CODES) - set(actual_directories))
    unexpected_directories = sorted(set(actual_directories) - set(TIME_CODES))
    if missing_directories or unexpected_directories:
        raise ValueError(
            "时刻目录不符合预期："
            f"缺少={missing_directories}，额外={unexpected_directories}"
        )
    root_files = sorted(path for path in source_root.iterdir() if path.is_file())
    if root_files:
        raise ValueError(f"来源根目录不应直接包含文件：{root_files[:10]}")

    actual_paths: dict[tuple[str, str], Path] = {}
    for time_code in TIME_CODES:
        time_directory = actual_directories[time_code]
        nested_directories = sorted(
            path for path in time_directory.iterdir() if path.is_dir()
        )
        if nested_directories:
            raise ValueError(
                f"时刻目录中不应再有子目录：{nested_directories[:10]}"
            )
        for image_path in sorted(time_directory.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() != ".bmp":
                raise ValueError(f"时刻目录中存在非 BMP 文件：{image_path}")
            if STEM_PATTERN.fullmatch(image_path.stem) is None:
                raise ValueError(
                    f"文件名不符合 <1-6>-<1-4>.bmp：{image_path}"
                )
            key = (time_code, image_path.stem)
            if key in actual_paths:
                raise ValueError(f"发现重复来源键：{key}")
            actual_paths[key] = image_path

    expected_keys = {
        (time_code, source_stem)
        for time_code in TIME_CODES
        for source_stem in EXPECTED_STEMS
    }
    allowed_keys = expected_keys - set(KNOWN_EXCLUSIONS)
    actual_keys = set(actual_paths)
    missing_keys = sorted(allowed_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "来源图片清单不符合预期："
            f"非排除项缺失={missing_keys[:20]}，额外={unexpected_keys[:20]}"
        )

    excluded_rows = []
    for (time_code, source_stem), reason in sorted(KNOWN_EXCLUSIONS.items()):
        excluded_path = actual_paths.get((time_code, source_stem))
        excluded_rows.append(
            {
                "source_relpath": f"{time_code}/{source_stem}.bmp",
                "time_code": time_code,
                "time_h": int(time_code) / 10.0,
                "source_stem": source_stem,
                "status": "excluded",
                "reason": reason,
                "present_in_source": excluded_path is not None,
            }
        )

    sources = [
        inspect_source_image(actual_paths[key], source_root, key[0])
        for key in sorted(
            allowed_keys,
            key=lambda item: (
                int(item[0]),
                int(item[1].split("-")[0]),
                int(item[1].split("-")[1]),
            ),
        )
    ]
    return sources, excluded_rows


def build_patch_plans(
    sources: Sequence[SourceImage],
    output_root: Path,
) -> list[PatchPlan]:
    plans: list[PatchPlan] = []
    seen_destinations: set[Path] = set()
    for source in sources:
        destination_directory = output_root / source.time_code
        for row_index in range(GRID_ROWS):
            top = row_index * PATCH_SIZE
            for column_index in range(GRID_COLUMNS):
                left = column_index * PATCH_SIZE
                patch_index = row_index * GRID_COLUMNS + column_index + 1
                destination = destination_directory / (
                    f"{source.source_image_id}__patch_{patch_index:02d}.jpg"
                )
                resolved_destination = destination.resolve()
                if resolved_destination in seen_destinations:
                    raise ValueError(f"目标 patch 路径冲突：{destination}")
                seen_destinations.add(resolved_destination)
                plans.append(
                    PatchPlan(
                        source=source,
                        destination=destination,
                        patch_relpath=relative_posix(destination, output_root),
                        patch_index=patch_index,
                        grid_row=row_index + 1,
                        grid_column=column_index + 1,
                        left=left,
                        top=top,
                        right=left + PATCH_SIZE,
                        bottom=top + PATCH_SIZE,
                    )
                )
    return plans


def validate_output_location(
    source_root: Path,
    output_root: Path,
    plans: Sequence[PatchPlan],
    overwrite: bool,
) -> None:
    if (
        output_root == source_root
        or source_root in output_root.parents
        or output_root in source_root.parents
    ):
        raise ValueError("输出目录必须与来源目录相互独立，不能互为父子目录。")
    if not output_root.exists():
        return

    expected_files = {plan.destination.resolve() for plan in plans}
    expected_files.update(
        (output_root / filename).resolve() for filename in CONTROL_FILENAMES
    )
    existing_files = {
        path.resolve() for path in output_root.rglob("*") if path.is_file()
    }
    unknown_files = sorted(existing_files - expected_files)
    if unknown_files:
        raise FileExistsError(
            "输出目录包含不属于当前计划的文件，拒绝继续：\n"
            + "\n".join(str(path) for path in unknown_files[:10])
        )
    if existing_files and not overwrite:
        raise FileExistsError(
            f"输出目录中已有 {len(existing_files)} 个计划文件；"
            "请使用新的输出目录，或明确添加 --overwrite。"
        )


def source_manifest_row(source: SourceImage) -> dict[str, Any]:
    return {
        "source_relpath": source.source_relpath,
        "source_image_id": source.source_image_id,
        "source_stem": source.source_stem,
        "name_part_1": source.name_part_1,
        "name_part_2": source.name_part_2,
        "time_code": source.time_code,
        "time_h": source.time_h,
        "severity": source.severity,
        "source_width": source.width,
        "source_height": source.height,
        "source_mode": source.mode,
        "expected_patch_count": PATCHES_PER_IMAGE,
        "split": "unassigned",
    }


def patch_manifest_row(
    plan: PatchPlan,
    resize_size: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    return {
        "patch_relpath": plan.patch_relpath,
        "source_relpath": plan.source.source_relpath,
        "source_image_id": plan.source.source_image_id,
        "source_stem": plan.source.source_stem,
        "name_part_1": plan.source.name_part_1,
        "name_part_2": plan.source.name_part_2,
        "time_code": plan.source.time_code,
        "time_h": plan.source.time_h,
        "severity": plan.source.severity,
        "split": "unassigned",
        "source_width": plan.source.width,
        "source_height": plan.source.height,
        "source_mode": plan.source.mode,
        "patch_index": plan.patch_index,
        "grid_row": plan.grid_row,
        "grid_column": plan.grid_column,
        "left": plan.left,
        "top": plan.top,
        "right": plan.right,
        "bottom": plan.bottom,
        "coordinate_convention": "0-based,left/top-inclusive,right/bottom-exclusive",
        "crop_width": PATCH_SIZE,
        "crop_height": PATCH_SIZE,
        "resize_width": resize_size,
        "resize_height": resize_size,
        "jpeg_quality": jpeg_quality,
        "crop_version": CROP_VERSION,
    }


def write_csv_atomic(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(f"不能写入空清单：{path}")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def print_plan(
    sources: Sequence[SourceImage],
    plans: Sequence[PatchPlan],
    excluded_rows: Sequence[dict[str, Any]],
    source_root: Path,
    output_root: Path,
    resize_size: int,
) -> None:
    time_counts = Counter(source.time_code for source in sources)
    severity_counts = Counter(source.severity for source in sources)
    print("\n时刻数据 30-Patch 准备计划")
    print(f"来源目录：{source_root}")
    print(f"输出目录：{output_root}")
    print(f"纳入原图：{len(sources)}")
    print(f"已登记排除：{len(excluded_rows)}")
    print(f"单图裁剪：{GRID_COLUMNS}×{GRID_ROWS}={PATCHES_PER_IMAGE}")
    print(f"裁剪尺寸：{PATCH_SIZE}×{PATCH_SIZE}")
    print(f"缩放尺寸：{resize_size}×{resize_size}")
    print(f"底部丢弃：{DISCARDED_BOTTOM_PIXELS} 像素")
    print(f"预计 Patch：{len(plans)}")
    print("各时刻原图：")
    for time_code in TIME_CODES:
        print(f"  {time_code} ({int(time_code) / 10.0:.1f} h)：{time_counts[time_code]}")
    print("各等级原图/Patch：")
    for severity in ("pre", "slight", "moderate", "over"):
        source_count = severity_counts[severity]
        print(f"  {severity}：{source_count}/{source_count * PATCHES_PER_IMAGE}")


def crop_all_sources(
    sources: Sequence[SourceImage],
    plans: Sequence[PatchPlan],
    resize_size: int,
    jpeg_quality: int,
) -> None:
    plans_by_source = {
        source.source_image_id: []
        for source in sources
    }
    for plan in plans:
        plans_by_source[plan.source.source_image_id].append(plan)

    for source_index, source in enumerate(sources, start=1):
        source_plans = plans_by_source[source.source_image_id]
        destination_paths = [plan.destination for plan in source_plans]
        crop_resize_and_save(
            source.path,
            destination_paths,
            resize_size,
            jpeg_quality,
        )
        print(
            f"[{source_index}/{len(sources)}] {source.source_relpath} -> "
            f"{len(destination_paths)} patches"
        )


def validate_written_outputs(
    output_root: Path,
    plans: Sequence[PatchPlan],
    resize_size: int,
) -> None:
    expected_paths = {plan.destination.resolve() for plan in plans}
    actual_paths = {
        path.resolve()
        for path in output_root.rglob("*.jpg")
        if path.is_file()
    }
    missing_paths = sorted(expected_paths - actual_paths)
    unexpected_paths = sorted(actual_paths - expected_paths)
    if missing_paths or unexpected_paths:
        raise RuntimeError(
            "输出文件集合与计划不一致："
            f"缺少={len(missing_paths)}，额外={len(unexpected_paths)}"
        )

    invalid_outputs = []
    for plan in plans:
        try:
            with Image.open(plan.destination) as image:
                image.load()
                if image.size != (resize_size, resize_size) or image.mode != "RGB":
                    invalid_outputs.append(
                        (plan.destination, image.size, image.mode)
                    )
        except Exception as error:
            invalid_outputs.append((plan.destination, "decode_error", str(error)))
        if len(invalid_outputs) >= 10:
            break
    if invalid_outputs:
        raise RuntimeError(f"输出图片 QA 失败：{invalid_outputs}")


def build_summary(
    sources: Sequence[SourceImage],
    plans: Sequence[PatchPlan],
    excluded_rows: Sequence[dict[str, Any]],
    source_root: Path,
    output_root: Path,
    resize_size: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    time_source_counts = Counter(source.time_code for source in sources)
    severity_source_counts = Counter(source.severity for source in sources)
    return {
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "crop_version": CROP_VERSION,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_count": len(sources),
        "patch_count": len(plans),
        "excluded_count": len(excluded_rows),
        "split_status": "unassigned",
        "time_source_counts": {
            time_code: time_source_counts[time_code]
            for time_code in TIME_CODES
        },
        "severity_source_counts": dict(sorted(severity_source_counts.items())),
        "severity_patch_counts": {
            severity: count * PATCHES_PER_IMAGE
            for severity, count in sorted(severity_source_counts.items())
        },
        "crop_parameters": {
            "source_width": EXPECTED_WIDTH,
            "source_height": EXPECTED_HEIGHT,
            "grid_columns": GRID_COLUMNS,
            "grid_rows": GRID_ROWS,
            "patch_size": PATCH_SIZE,
            "patches_per_image": PATCHES_PER_IMAGE,
            "discarded_bottom_pixels": DISCARDED_BOTTOM_PIXELS,
            "resize_size": resize_size,
            "resize_resampling": "LANCZOS",
            "output_format": "JPEG",
            "jpeg_quality": jpeg_quality,
            "jpeg_subsampling": 0,
            "jpeg_optimize": True,
        },
        "known_exclusions": list(excluded_rows),
    }


def main() -> None:
    args = parse_arguments()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"来源目录不存在：{source_root}")
    if args.resize_size <= 0:
        raise ValueError("--resize-size 必须是大于 0 的整数。")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality 必须位于 1～100。")

    sources, excluded_rows = collect_source_images(source_root)
    plans = build_patch_plans(sources, output_root)
    expected_source_count = len(TIME_CODES) * len(EXPECTED_STEMS) - len(
        KNOWN_EXCLUSIONS
    )
    expected_patch_count = expected_source_count * PATCHES_PER_IMAGE
    if len(sources) != expected_source_count or len(plans) != expected_patch_count:
        raise RuntimeError(
            "内部数量断言失败："
            f"sources={len(sources)}/{expected_source_count}，"
            f"patches={len(plans)}/{expected_patch_count}"
        )
    validate_output_location(
        source_root,
        output_root,
        plans,
        overwrite=bool(args.overwrite),
    )
    print_plan(
        sources,
        plans,
        excluded_rows,
        source_root,
        output_root,
        args.resize_size,
    )
    if args.dry_run:
        print("\ndry-run 完成：未创建目录、Patch 或清单。")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    crop_all_sources(
        sources,
        plans,
        args.resize_size,
        args.jpeg_quality,
    )
    validate_written_outputs(output_root, plans, args.resize_size)

    source_rows = [source_manifest_row(source) for source in sources]
    patch_rows = [
        patch_manifest_row(plan, args.resize_size, args.jpeg_quality)
        for plan in plans
    ]
    write_csv_atomic(output_root / "source_manifest.csv", source_rows)
    write_csv_atomic(output_root / "patch_manifest.csv", patch_rows)
    write_csv_atomic(output_root / "excluded_samples.csv", excluded_rows)
    summary = build_summary(
        sources,
        plans,
        excluded_rows,
        source_root,
        output_root,
        args.resize_size,
        args.jpeg_quality,
    )
    write_json_atomic(output_root / "preparation_summary.json", summary)
    print(f"\n处理完成：{len(sources)} 张原图 -> {len(plans)} 张 Patch")
    print(f"输出目录：{output_root}")
    print(f"Patch 清单：{output_root / 'patch_manifest.csv'}")


if __name__ == "__main__":
    main()
