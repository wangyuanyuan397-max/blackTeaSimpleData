"""Create source-level train/val/test splits for Moderate-vs-Over diagnostics."""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from common import ensure_dir, save_json, source_metadata_path, source_splits_path


def split_source_frame(source_frame: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    source_frame = source_frame.copy().reset_index(drop=True)
    first_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val_index, test_index = next(first_split.split(source_frame, source_frame["time_code"].astype(str)))
    train_val = source_frame.iloc[train_val_index].reset_index(drop=True)
    test = source_frame.iloc[test_index].reset_index(drop=True)
    second_split = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + 1)
    train_index, val_index = next(second_split.split(train_val, train_val["time_code"].astype(str)))
    train = train_val.iloc[train_index].copy()
    val = train_val.iloc[val_index].copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return pd.concat([train, val, test], ignore_index=True).sort_values(["split", "time_code", "source_image_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按原图 source_image_id 创建 7:1:2 split。")
    parser.add_argument("--run-dir", required=True, help="build_metadata.py 生成的运行目录。")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_frame = pd.read_csv(source_metadata_path(args.run_dir), dtype={"time_code": str})
    split_frame = split_source_frame(source_frame, seed=args.seed)
    split_dir = ensure_dir(source_splits_path(args.run_dir).parent)
    split_frame.to_csv(source_splits_path(args.run_dir), index=False, encoding="utf-8-sig")
    count_table = split_frame.groupby(["split", "time_code", "label_name"]).size().reset_index(name="count")
    count_table.to_csv(split_dir / "split_counts.csv", index=False, encoding="utf-8-sig")
    save_json(split_dir / "split_summary.json", {
        "seed": args.seed,
        "total_sources": int(len(split_frame)),
        "counts_by_split": split_frame.groupby("split").size().astype(int).to_dict(),
        "counts_by_split_and_time": count_table.to_dict(orient="records"),
    })
    print(f"source splits saved: {split_dir}")


if __name__ == "__main__":
    main()
