"""Aggregate 30 patch predictions back to each source image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dir, save_json


def entropy_binary(prob_over: float) -> float:
    values = [1.0 - prob_over, prob_over]
    return float(-sum(value * np.log2(value) for value in values if value > 0))


def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_id, group in predictions.groupby("source_image_id"):
        total = int(len(group))
        moderate_count = int((group["pred_label"].astype(int) == 0).sum())
        over_count = int((group["pred_label"].astype(int) == 1).sum())
        over_ratio = over_count / total if total else 0.0
        true_label = int(group["label"].iloc[0])
        majority_pred = 1 if over_count > moderate_count else 0
        rows.append({
            "source_image_id": source_id,
            "source_stem": group["source_stem"].iloc[0],
            "time_code": str(group["time_code"].iloc[0]).zfill(2),
            "time_h": float(group["time_h"].iloc[0]),
            "true_label": true_label,
            "true_label_name": "over" if true_label == 1 else "moderate",
            "patch_count": total,
            "moderate_count": moderate_count,
            "over_count": over_count,
            "moderate_ratio": moderate_count / total if total else 0.0,
            "over_ratio": over_ratio,
            "mean_prob_over": float(group["prob_over"].mean()),
            "mixed_score": min(moderate_count, over_count) / total if total else 0.0,
            "entropy": entropy_binary(over_ratio),
            "majority_pred_label": majority_pred,
            "majority_correct": int(majority_pred == true_label),
        })
    return pd.DataFrame(rows).sort_values(["time_code", "source_image_id"]).reset_index(drop=True)


def plot_over_ratio_bar(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values(["time_code", "over_ratio", "source_image_id"]).reset_index(drop=True)
    colors = ["#4C78A8" if label == 0 else "#F58518" for label in ordered["true_label"]]
    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.08), 4.8))
    ax.bar(np.arange(len(ordered)), ordered["over_ratio"], color=colors, width=0.9)
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Over prediction ratio among 30 patches")
    ax.set_xlabel("Source images sorted by time and ratio")
    ax.set_title("Patch-level prediction consistency per source image")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_time_box(summary: pd.DataFrame, output_path: Path) -> None:
    time_codes = sorted(summary["time_code"].astype(str).unique())
    values = [summary.loc[summary["time_code"].astype(str) == code, "over_ratio"].to_numpy() for code in time_codes]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.boxplot(values, labels=time_codes, showmeans=True)
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time code")
    ax.set_ylabel("Over prediction ratio")
    ax.set_title("Patch prediction ratio by fermentation time")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_mixed_score(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for time_code, group in summary.groupby("time_code"):
        ax.scatter([str(time_code)] * len(group), group["mixed_score"], alpha=0.75)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Time code")
    ax.set_ylabel("min(moderate_count, over_count) / patch_count")
    ax.set_title("Within-source heterogeneity score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def diagnose(run_dir: Path, predictions_csv: Path | None = None, split: str = "test") -> Path:
    if predictions_csv is None:
        predictions_csv = run_dir / "cnn_patch_rgb" / f"{split}_predictions.csv"
    predictions = pd.read_csv(predictions_csv, dtype={"time_code": str})
    out_dir = ensure_dir(run_dir / "patch_consistency")
    summary = aggregate_predictions(predictions)
    summary.to_csv(out_dir / f"{split}_source_patch_consistency.csv", index=False, encoding="utf-8-sig")
    top_mixed = summary.sort_values("mixed_score", ascending=False).head(30)
    top_mixed.to_csv(out_dir / f"{split}_top_mixed_sources.csv", index=False, encoding="utf-8-sig")
    plot_over_ratio_bar(summary, out_dir / f"{split}_over_ratio_bar.png")
    plot_time_box(summary, out_dir / f"{split}_over_ratio_by_time.png")
    plot_mixed_score(summary, out_dir / f"{split}_mixed_score_by_time.png")
    save_json(out_dir / f"{split}_summary.json", {
        "source_count": int(len(summary)),
        "mean_mixed_score": float(summary["mixed_score"].mean()),
        "median_mixed_score": float(summary["mixed_score"].median()),
        "majority_accuracy": float(summary["majority_correct"].mean()),
        "highly_mixed_count_ge_0_4": int((summary["mixed_score"] >= 0.4).sum()),
    })
    print(f"patch consistency saved: {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate 30 patch predictions per source image.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    diagnose(Path(args.run_dir), Path(args.predictions_csv) if args.predictions_csv else None, split=args.split)
