"""Summarize MixNet-S deformable-attention brute-force runs."""

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
DEFAULT_RUNS_ROOT = THIS_DIR / "runs_grid30_408"
RUN_NAME_RE = re.compile(r"^(D[01]{5})_seed(\d+)")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def bits_to_stage_ids(config_code: str) -> list[int]:
    bits = config_code[1:]
    return [index for index, char in enumerate(bits) if char == "1"]


def stage_key(config_code: str) -> str:
    stages = bits_to_stage_ids(config_code)
    if not stages:
        return "baseline"
    return "+".join(f"S{stage_id}" for stage_id in stages)


def parse_run_identity(run_dir: Path, config_data: dict[str, Any]) -> tuple[str, int]:
    run_name = str(config_data.get("run_name") or run_dir.name)
    match = RUN_NAME_RE.match(run_name) or RUN_NAME_RE.match(run_dir.name)
    if not match:
        raise ValueError(f"Cannot parse D-code/seed from run directory: {run_dir}")
    return match.group(1), int(match.group(2))


def nested_value(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


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
    if len(values) >= 2:
        return statistics.stdev(values)
    if len(values) == 1:
        return 0.0
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
        config_code, seed = parse_run_identity(run_dir, config_data)
        stages = bits_to_stage_ids(config_code)
        best_validation = metrics.get("best_validation_metrics") or {}
        detailed_validation = metrics.get("best_validation_detailed_metrics") or {}
        deform_metadata = metrics.get("deformable_attention") or {}

        rows.append(
            {
                "run_id": f"{config_code}_seed{seed}",
                "config_code": config_code,
                "stage_bits": config_code[1:],
                "stage_key": stage_key(config_code),
                "deform_stage_ids": stages,
                "num_deform_blocks": len(stages),
                "seed": seed,
                "run_directory": str(run_dir),
                "status": "completed" if metrics else "unknown",
                "best_epoch": metrics.get("best_epoch"),
                "best_val_acc": best_validation.get("val_acc"),
                "best_val_loss": best_validation.get("val_loss"),
                "best_val_mae": best_validation.get("val_mae"),
                "best_val_qwk": best_validation.get("val_qwk"),
                "best_val_macro_f1": metrics.get("best_val_macro_f1")
                or detailed_validation.get("macro_f1"),
                "best_val_accuracy_detailed": metrics.get("best_val_accuracy_detailed")
                or detailed_validation.get("accuracy"),
                "test_accuracy": metrics.get("accuracy"),
                "test_macro_f1": metrics.get("macro_f1"),
                "test_mae": metrics.get("mae"),
                "test_qwk": metrics.get("qwk"),
                "plus_minus_one_accuracy": metrics.get("plus_minus_one_accuracy"),
                "parameters_total": metrics.get("parameters_total"),
                "parameters_trainable": metrics.get("parameters_trainable"),
                "flops_g": metrics.get("flops_g"),
                "training_time_seconds": metrics.get("training_time_seconds"),
                "inference_ms_per_sample": metrics.get("inference_ms_per_sample"),
                "deform_stage_infos": json.dumps(
                    nested_value(deform_metadata, ("stage_infos",)) or [],
                    ensure_ascii=False,
                ),
                "deform_diagnostics": json.dumps(
                    nested_value(deform_metadata, ("diagnostics",)) or {},
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for config_code in sorted({str(row["config_code"]) for row in rows}):
        group = [row for row in rows if row["config_code"] == config_code]
        stages = bits_to_stage_ids(config_code)

        def numeric(key: str) -> list[float]:
            return [value for value in (to_float(row.get(key)) for row in group) if value is not None]

        val_f1_mean = mean(numeric("best_val_macro_f1"))
        val_f1_std = stdev(numeric("best_val_macro_f1"))
        score = None
        if val_f1_mean is not None and val_f1_std is not None:
            score = val_f1_mean - 0.2 * val_f1_std - 0.002 * len(stages)

        summary_rows.append(
            {
                "config_code": config_code,
                "stage_bits": config_code[1:],
                "stage_key": stage_key(config_code),
                "deform_stage_ids": stages,
                "num_deform_blocks": len(stages),
                "seed_count": len(group),
                "best_val_macro_f1_mean": val_f1_mean,
                "best_val_macro_f1_std": val_f1_std,
                "best_val_acc_mean": mean(numeric("best_val_acc")),
                "best_val_acc_std": stdev(numeric("best_val_acc")),
                "best_val_qwk_mean": mean(numeric("best_val_qwk")),
                "test_accuracy_mean": mean(numeric("test_accuracy")),
                "test_accuracy_std": stdev(numeric("test_accuracy")),
                "test_macro_f1_mean": mean(numeric("test_macro_f1")),
                "test_macro_f1_std": stdev(numeric("test_macro_f1")),
                "parameters_total": group[0].get("parameters_total"),
                "parameters_trainable": group[0].get("parameters_trainable"),
                "flops_g": group[0].get("flops_g"),
                "training_time_seconds_mean": mean(numeric("training_time_seconds")),
                "inference_ms_per_sample_mean": mean(numeric("inference_ms_per_sample")),
                "selection_score": score,
            }
        )

    summary_rows.sort(
        key=lambda row: (
            to_float(row.get("selection_score")) is None,
            -(to_float(row.get("selection_score")) or 0.0),
            -(to_float(row.get("best_val_macro_f1_mean")) or 0.0),
            -(to_float(row.get("best_val_acc_mean")) or 0.0),
            to_float(row.get("best_val_macro_f1_std")) or 0.0,
            int(row.get("num_deform_blocks") or 0),
        )
    )
    for rank, row in enumerate(summary_rows, start=1):
        row["rank_by_validation"] = rank
    return summary_rows


def single_stage_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in summary_rows
        if row.get("config_code") == "D00000" or int(row.get("num_deform_blocks") or 0) == 1
    ]
    return sorted(
        selected,
        key=lambda row: (
            int(row.get("num_deform_blocks") or 0),
            str(row.get("config_code")),
        ),
    )


def stage_frequency_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((row for row in summary_rows if row.get("config_code") == "D00000"), None)
    baseline_f1 = to_float(baseline.get("best_val_macro_f1_mean")) if baseline else None
    if baseline_f1 is None:
        return []

    improved = [
        row
        for row in summary_rows
        if row.get("config_code") != "D00000"
        and (to_float(row.get("best_val_macro_f1_mean")) or -1.0) > baseline_f1
    ]
    total = len(improved)
    rows: list[dict[str, Any]] = []
    for stage_id in range(5):
        count = sum(1 for row in improved if stage_id in (row.get("deform_stage_ids") or []))
        rows.append(
            {
                "stage": f"S{stage_id}",
                "count_above_baseline": count,
                "total_configs_above_baseline": total,
                "frequency": count / total if total else 0.0,
                "baseline_best_val_macro_f1_mean": baseline_f1,
            }
        )
    return rows


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

    summary = summarize_rows(rows)
    all_results_path = runs_root / "deform_attention_all_results.csv"
    summary_path = runs_root / "deform_attention_summary_by_config.csv"
    single_stage_path = runs_root / "deform_attention_single_stage_summary.csv"
    frequency_path = runs_root / "deform_attention_stage_frequency_above_baseline.csv"

    write_csv(all_results_path, rows)
    write_csv(summary_path, summary)
    write_csv(single_stage_path, single_stage_rows(summary))
    write_csv(frequency_path, stage_frequency_rows(summary))

    print(f"Wrote all-run results: {all_results_path}")
    print(f"Wrote config summary: {summary_path}")
    print(f"Wrote single-stage summary: {single_stage_path}")
    print(f"Wrote stage-frequency summary: {frequency_path}")
    print("\nTop validation-ranked configs:")
    for row in summary[:10]:
        print(
            f"  {row['rank_by_validation']:>2}. {row['stage_key']:<17} "
            f"val_macro_f1={row.get('best_val_macro_f1_mean')} "
            f"score={row.get('selection_score')}"
        )
    return all_results_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize deformable attention sweep runs.")
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
