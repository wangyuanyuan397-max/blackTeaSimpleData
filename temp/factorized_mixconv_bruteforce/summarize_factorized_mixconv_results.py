"""Summarize Factorized MixConv brute-force runs by F000-F111 config."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

import yaml


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = THIS_DIR / "runs_BaSic"
RUN_NAME_RE = re.compile(r"^(F[01]{3})_seed(\d+)")
KERNELS_BY_CONFIG = {
    "F000": [],
    "F001": [9],
    "F010": [7],
    "F011": [7, 9],
    "F100": [5],
    "F101": [5, 9],
    "F110": [5, 7],
    "F111": [5, 7, 9],
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def parse_run_identity(run_dir: Path, config_data: dict[str, Any]) -> tuple[str, int]:
    run_name = str(config_data.get("run_name") or run_dir.name)
    match = RUN_NAME_RE.match(run_name) or RUN_NAME_RE.match(run_dir.name)
    if not match:
        raise ValueError(f"Cannot parse config/seed from run directory: {run_dir}")
    return match.group(1), int(match.group(2))


def best_validation_value(metrics: dict[str, Any], key: str) -> Any:
    values = metrics.get("best_validation_metrics")
    if isinstance(values, dict):
        return values.get(key)
    return None


def collect_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("*/test_metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            continue
        metrics = read_json(metrics_path)
        config_data = read_yaml(config_path)
        config_name, seed = parse_run_identity(run_dir, config_data)
        rows.append(
            {
                "run_id": f"{config_name}_seed{seed}",
                "config_name": config_name,
                "seed": seed,
                "factorized_kernels": KERNELS_BY_CONFIG.get(config_name, []),
                "run_directory": str(run_dir),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "plus_minus_one_accuracy": metrics.get("plus_minus_one_accuracy"),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_acc": best_validation_value(metrics, "val_acc"),
                "best_val_loss": best_validation_value(metrics, "val_loss"),
                "parameters_total": metrics.get("parameters_total"),
                "parameters_trainable": metrics.get("parameters_trainable"),
                "flops_g": metrics.get("flops_g"),
                "training_time_seconds": metrics.get("training_time_seconds"),
                "inference_ms_per_sample": metrics.get("inference_ms_per_sample"),
            }
        )
    return rows


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else 0.0 if len(values) == 1 else None


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for config_name in sorted({str(row["config_name"]) for row in rows}):
        group = [row for row in rows if row["config_name"] == config_name]

        def numeric(key: str) -> list[float]:
            return [value for value in (to_float(row.get(key)) for row in group) if value is not None]

        summary_rows.append(
            {
                "config_name": config_name,
                "factorized_kernels": KERNELS_BY_CONFIG.get(config_name, []),
                "seed_count": len(group),
                "accuracy_mean": mean(numeric("accuracy")),
                "accuracy_std": stdev(numeric("accuracy")),
                "macro_f1_mean": mean(numeric("macro_f1")),
                "macro_f1_std": stdev(numeric("macro_f1")),
                "plus_minus_one_accuracy_mean": mean(numeric("plus_minus_one_accuracy")),
                "best_val_acc_mean": mean(numeric("best_val_acc")),
                "best_epoch_mean": mean(numeric("best_epoch")),
                "parameters_total": group[0].get("parameters_total"),
                "parameters_trainable": group[0].get("parameters_trainable"),
                "flops_g": group[0].get("flops_g"),
                "training_time_seconds_mean": mean(numeric("training_time_seconds")),
                "inference_ms_per_sample_mean": mean(numeric("inference_ms_per_sample")),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            to_float(row.get("accuracy_mean")) is None,
            -(to_float(row.get("accuracy_mean")) or 0.0),
        )
    )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_results(runs_root: Path = DEFAULT_RUNS_ROOT) -> tuple[Path, Path] | None:
    runs_root = Path(runs_root)
    rows = collect_rows(runs_root)
    if not rows:
        print(f"No completed test_metrics.json files found under {runs_root}")
        return None
    all_results_path = runs_root / "factorized_mixconv_all_results.csv"
    summary_path = runs_root / "factorized_mixconv_summary_by_config.csv"
    write_csv(all_results_path, rows)
    write_csv(summary_path, summarize_rows(rows))
    print(f"Wrote all-run results: {all_results_path}")
    print(f"Wrote config summary: {summary_path}")
    return all_results_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Factorized MixConv brute-force runs.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Run output root containing per-run test_metrics.json files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_results(args.runs_root)


if __name__ == "__main__":
    main()
