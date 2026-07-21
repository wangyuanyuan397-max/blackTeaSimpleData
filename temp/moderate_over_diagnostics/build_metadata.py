"""Build source-level and patch-level metadata for Moderate-vs-Over diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    ORIGINAL_ROOT,
    PATCH_ROOT,
    TARGET_CODES,
    create_run_dir,
    ensure_dir,
    label_from_time_code,
    list_image_files,
    patch_metadata_path,
    save_json,
    source_metadata_path,
)


def build_source_metadata(original_root: Path) -> pd.DataFrame:
    rows = []
    for time_code in TARGET_CODES:
        folder = original_root / time_code
        if not folder.is_dir():
            continue
        label, label_name = label_from_time_code(time_code)
        for image_path in list_image_files(folder):
            source_stem = image_path.stem
            rows.append({
                "source_image_id": f"t{time_code}__{source_stem}",
                "source_stem": source_stem,
                "time_code": time_code,
                "time_h": int(time_code) / 10.0,
                "label": label,
                "label_name": label_name,
                "source_relpath": image_path.relative_to(original_root).as_posix(),
                "original_path": str(image_path.resolve()),
            })
    frame = pd.DataFrame(rows).sort_values(["time_code", "source_stem"]).reset_index(drop=True)
    if frame.empty:
        raise FileNotFoundError(f"没有在 {original_root} 中找到 Moderate/Over 原图。")
    return frame


def build_patch_metadata(patch_root: Path, source_metadata: pd.DataFrame) -> pd.DataFrame:
    manifest_path = patch_root / "patch_manifest.csv"
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path, dtype=str)
        manifest = manifest[manifest["time_code"].isin(TARGET_CODES)].copy()
        manifest["patch_path"] = manifest["patch_relpath"].map(lambda value: str((patch_root / value).resolve()))
        manifest["time_h"] = manifest["time_code"].astype(int) / 10.0
        manifest["label"] = manifest["time_code"].map(lambda code: label_from_time_code(code)[0])
        manifest["label_name"] = manifest["label"].map({0: "moderate", 1: "over"})
        keep_columns = [
            "patch_relpath", "patch_path", "source_relpath", "source_image_id", "source_stem",
            "time_code", "time_h", "label", "label_name", "patch_index", "grid_row", "grid_column",
        ]
        frame = manifest[[column for column in keep_columns if column in manifest.columns]].copy()
    else:
        rows = []
        for _, source in source_metadata.iterrows():
            time_code = str(source["time_code"]).zfill(2)
            pattern = f"t{time_code}__{source['source_stem']}__patch_*.jpg"
            for patch_path in sorted((patch_root / time_code).glob(pattern)):
                rows.append({
                    "patch_relpath": patch_path.relative_to(patch_root).as_posix(),
                    "patch_path": str(patch_path.resolve()),
                    "source_relpath": source["source_relpath"],
                    "source_image_id": source["source_image_id"],
                    "source_stem": source["source_stem"],
                    "time_code": time_code,
                    "time_h": source["time_h"],
                    "label": int(source["label"]),
                    "label_name": source["label_name"],
                    "patch_index": int(patch_path.stem.rsplit("_", 1)[-1]),
                })
        frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"没有在 {patch_root} 中找到 Moderate/Over patch。")
    frame["patch_exists"] = frame["patch_path"].map(lambda value: Path(value).is_file())
    return frame.sort_values(["time_code", "source_image_id", "patch_index"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Moderate/Over 诊断元数据。")
    parser.add_argument("--run-dir", default=None, help="输出运行目录；不填则自动创建 outputs/时间戳。")
    parser.add_argument("--original-root", default=str(ORIGINAL_ROOT), help="原图时间点目录。")
    parser.add_argument("--patch-root", default=str(PATCH_ROOT), help="30 patch 时间点目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = create_run_dir(args.run_dir)
    original_root = Path(args.original_root)
    patch_root = Path(args.patch_root)
    metadata_dir = ensure_dir(run_dir / "metadata")
    source_frame = build_source_metadata(original_root)
    patch_frame = build_patch_metadata(patch_root, source_frame)
    source_frame.to_csv(source_metadata_path(run_dir), index=False, encoding="utf-8-sig")
    patch_frame.to_csv(patch_metadata_path(run_dir), index=False, encoding="utf-8-sig")
    summary = {
        "run_dir": str(run_dir.resolve()),
        "source_count": int(len(source_frame)),
        "patch_count": int(len(patch_frame)),
        "source_by_time": source_frame.groupby("time_code").size().astype(int).to_dict(),
        "patch_by_time": patch_frame.groupby("time_code").size().astype(int).to_dict(),
        "missing_patch_count": int((~patch_frame["patch_exists"]).sum()),
    }
    save_json(metadata_dir / "metadata_summary.json", summary)
    print(f"metadata saved: {metadata_dir}")


if __name__ == "__main__":
    main()
