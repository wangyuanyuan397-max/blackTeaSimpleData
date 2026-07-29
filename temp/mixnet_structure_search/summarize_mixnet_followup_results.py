"""Summarize follow-up MixNet-S structure-search runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = THIS_DIR / "runs_BaSic"

SEED_REPRO_RE = re.compile(r"^repro_(.+)_seed(\d+)(?:_|$)")
GATE_RE = re.compile(
    r"^gate_(only_s0_k357|s235_k357)_(g[0-3])_(none|static|sigmoid|softmax)(?:_|$)"
)
KERNEL_RE = r"(k3579|k357|k35|k3)"
GRID_RE = re.compile(
    rf"^s235grid_s2{KERNEL_RE}_s3{KERNEL_RE}_s5{KERNEL_RE}(?:_|$)"
)

GATE_ORDER = {"g0": 0, "g1": 1, "g2": 2, "g3": 3}
STRUCTURE_ORDER = {
    "original_mixnet_s": 0,
    "only_s0_k357": 1,
    "s235_k357": 2,
    "stride2_k357_g3_softmax": 3,
}

LEGACY_IDENTITIES: dict[str, list[dict[str, Any]]] = {
    "00_p00_original": [
        {"family": "seed_repro", "structure": "original_mixnet_s"},
    ],
    "10_p10_only_s0_k357": [
        {"family": "seed_repro", "structure": "only_s0_k357"},
        {
            "family": "champion_gates",
            "structure": "only_s0_k357",
            "gate_id": "g0",
            "gate_type": "none",
        },
    ],
    "stagemask_100000_k357": [
        {"family": "seed_repro", "structure": "only_s0_k357"},
        {
            "family": "champion_gates",
            "structure": "only_s0_k357",
            "gate_id": "g0",
            "gate_type": "none",
        },
    ],
    "stagemask_001101_k357": [
        {"family": "seed_repro", "structure": "s235_k357"},
        {
            "family": "champion_gates",
            "structure": "s235_k357",
            "gate_id": "g0",
            "gate_type": "none",
        },
    ],
    "p03_stride2_k357_g3_softmax": [
        {"family": "seed_repro", "structure": "stride2_k357_g3_softmax"},
    ],
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        return text


def read_run_config(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line[0].isspace() or ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        if key in {"run_name", "random_seed"}:
            config[key] = parse_scalar(value)
    return config


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


def mean_std_text(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None:
        return ""
    if std_value is None:
        return f"{mean_value:.6f}"
    return f"{mean_value:.6f} +/- {std_value:.6f}"


def best_validation_value(metrics: dict[str, Any], key: str) -> Any:
    values = metrics.get("best_validation_metrics")
    return values.get(key) if isinstance(values, dict) else None


def run_name_without_timestamp(run_name: str) -> str:
    match = re.match(r"^(.+)_20\d{6}_\d{6}(?:_\d{2})?$", run_name)
    return match.group(1) if match else run_name


def identify_run(run_name: str, seed: int) -> list[dict[str, Any]]:
    name = run_name_without_timestamp(run_name)
    identities: list[dict[str, Any]] = []

    seed_match = SEED_REPRO_RE.match(name)
    if seed_match:
        identities.append(
            {
                "family": "seed_repro",
                "structure": seed_match.group(1),
                "seed": int(seed_match.group(2)),
            }
        )

    gate_match = GATE_RE.match(name)
    if gate_match:
        identities.append(
            {
                "family": "champion_gates",
                "structure": gate_match.group(1),
                "gate_id": gate_match.group(2),
                "gate_type": gate_match.group(3),
                "seed": seed,
            }
        )

    grid_match = GRID_RE.match(name)
    if grid_match:
        identities.append(
            {
                "family": "s235_kernel_grid",
                "structure": "s235_kernel_grid",
                "s2_kernel": grid_match.group(1),
                "s3_kernel": grid_match.group(2),
                "s5_kernel": grid_match.group(3),
                "seed": seed,
            }
        )

    for identity in LEGACY_IDENTITIES.get(name, []):
        copied = dict(identity)
        copied.setdefault("seed", seed)
        copied["legacy_source_name"] = name
        identities.append(copied)

    return identities


def logical_key(row: dict[str, Any]) -> tuple[Any, ...]:
    family = row.get("family")
    if family == "seed_repro":
        return (family, row.get("structure"), row.get("seed"))
    if family == "champion_gates":
        return (
            family,
            row.get("structure"),
            row.get("gate_id"),
            row.get("gate_type"),
            row.get("seed"),
        )
    if family == "s235_kernel_grid":
        return (
            family,
            row.get("s2_kernel"),
            row.get("s3_kernel"),
            row.get("s5_kernel"),
            row.get("seed"),
        )
    return (family, row.get("run_name"))


def collect_rows(runs_root: Path, include_duplicates: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("*/test_metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            continue

        metrics = read_json(metrics_path)
        config_data = read_run_config(config_path)
        run_name = str(config_data.get("run_name") or run_dir.name)
        seed = int(config_data.get("random_seed") or 2026)
        identities = identify_run(run_name, seed)
        if not identities:
            continue

        base_row = {
            "run_name": run_name,
            "seed": seed,
            "run_directory": str(run_dir),
            "run_mtime": run_dir.stat().st_mtime,
            "best_epoch": metrics.get("best_epoch"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "mae": metrics.get("mae"),
            "qwk": metrics.get("qwk"),
            "plus_minus_one_accuracy": metrics.get("plus_minus_one_accuracy"),
            "best_val_acc": best_validation_value(metrics, "val_acc"),
            "best_val_loss": best_validation_value(metrics, "val_loss"),
            "best_val_mae": best_validation_value(metrics, "val_mae"),
            "best_val_qwk": best_validation_value(metrics, "val_qwk"),
            "parameters_total": metrics.get("parameters_total"),
            "parameters_trainable": metrics.get("parameters_trainable"),
            "flops_g": metrics.get("flops_g"),
            "training_time_seconds": metrics.get("training_time_seconds"),
            "inference_ms_per_sample": metrics.get("inference_ms_per_sample"),
        }
        for identity in identities:
            row = dict(base_row)
            row.update(identity)
            rows.append(row)

    if include_duplicates:
        return rows

    latest_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = logical_key(row)
        current = latest_by_key.get(key)
        if current is None or float(row["run_mtime"]) > float(current["run_mtime"]):
            latest_by_key[key] = row
    return list(latest_by_key.values())


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        value
        for value in (to_float(row.get(key)) for row in rows)
        if value is not None
    ]


def summarize_seed_repro(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    seed_rows = [row for row in rows if row.get("family") == "seed_repro"]
    for structure in sorted({str(row["structure"]) for row in seed_rows}):
        group = [row for row in seed_rows if row.get("structure") == structure]
        output: dict[str, Any] = {
            "structure": structure,
            "seed_count": len(group),
            "seeds": " ".join(str(row.get("seed")) for row in sorted(group, key=lambda item: int(item.get("seed") or 0))),
            "parameters_total": group[0].get("parameters_total"),
            "parameters_trainable": group[0].get("parameters_trainable"),
            "flops_g": group[0].get("flops_g"),
        }
        for metric in (
            "accuracy",
            "macro_f1",
            "qwk",
            "mae",
            "plus_minus_one_accuracy",
            "best_val_acc",
            "best_val_qwk",
            "training_time_seconds",
            "inference_ms_per_sample",
        ):
            metric_mean = mean(numeric_values(group, metric))
            metric_std = stdev(numeric_values(group, metric))
            output[f"{metric}_mean"] = metric_mean
            output[f"{metric}_std"] = metric_std
            output[f"{metric}_mean_std"] = mean_std_text(metric_mean, metric_std)
        summary_rows.append(output)

    summary_rows.sort(
        key=lambda row: (
            STRUCTURE_ORDER.get(str(row.get("structure")), 99),
            -(to_float(row.get("macro_f1_mean")) or -1.0),
            -(to_float(row.get("accuracy_mean")) or -1.0),
        )
    )
    return summary_rows


def summarize_champion_gates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gate_rows = [row for row in rows if row.get("family") == "champion_gates"]
    baselines: dict[str, dict[str, Any]] = {}
    for row in gate_rows:
        if row.get("gate_id") == "g0":
            baselines[str(row["structure"])] = row

    summary_rows: list[dict[str, Any]] = []
    for row in sorted(
        gate_rows,
        key=lambda item: (
            STRUCTURE_ORDER.get(str(item.get("structure")), 99),
            GATE_ORDER.get(str(item.get("gate_id")), 99),
        ),
    ):
        baseline = baselines.get(str(row["structure"]), {})
        output = dict(row)
        output.pop("run_mtime", None)
        for metric in ("accuracy", "macro_f1", "qwk", "mae", "best_val_acc", "best_val_qwk"):
            current = to_float(row.get(metric))
            base_value = to_float(baseline.get(metric))
            output[f"delta_{metric}_vs_g0"] = (
                current - base_value
                if current is not None and base_value is not None
                else None
            )
        summary_rows.append(output)
    return summary_rows


def summarize_s235_kernel_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grid_rows = [row for row in rows if row.get("family") == "s235_kernel_grid"]
    summary_rows: list[dict[str, Any]] = []
    for row in grid_rows:
        output = dict(row)
        output.pop("run_mtime", None)
        kernels = [str(row.get("s2_kernel")), str(row.get("s3_kernel")), str(row.get("s5_kernel"))]
        output["enabled_stage_count"] = sum(kernel != "k3" for kernel in kernels)
        output["total_kernel_branch_count_s235"] = sum(
            {"k3": 1, "k35": 2, "k357": 3, "k3579": 4}.get(kernel, 0)
            for kernel in kernels
        )
        summary_rows.append(output)

    summary_rows.sort(
        key=lambda row: (
            to_float(row.get("best_val_acc")) is None,
            -(to_float(row.get("best_val_acc")) or -1.0),
            -(to_float(row.get("best_val_qwk")) or -1.0),
            -(to_float(row.get("accuracy")) or -1.0),
            int(row.get("total_kernel_branch_count_s235") or 0),
        )
    )
    for rank, row in enumerate(summary_rows, start=1):
        row["rank_by_best_val_acc"] = rank
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def summarize_results(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    include_duplicates: bool = False,
) -> list[Path]:
    runs_root = Path(runs_root)
    rows = collect_rows(runs_root, include_duplicates=include_duplicates)
    if not rows:
        print(f"No follow-up test_metrics.json files found under {runs_root}")
        return []

    paths = [
        write_csv(runs_root / "mixnet_followup_all_results.csv", [
            {key: value for key, value in row.items() if key != "run_mtime"}
            for row in sorted(rows, key=lambda item: str(item.get("run_name")))
        ]),
        write_csv(
            runs_root / "mixnet_seed_repro_summary.csv",
            summarize_seed_repro(rows),
        ),
        write_csv(
            runs_root / "mixnet_champion_gate_summary.csv",
            summarize_champion_gates(rows),
        ),
        write_csv(
            runs_root / "mixnet_s235_kernel_grid_summary.csv",
            summarize_s235_kernel_grid(rows),
        ),
    ]
    written = [path for path in paths if path is not None]
    for path in written:
        print(f"Wrote {path}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MixNet-S follow-up runs.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Run output root containing per-run test_metrics.json files.",
    )
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="Keep duplicate logical runs instead of using the latest run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_results(args.runs_root, include_duplicates=args.include_duplicates)


if __name__ == "__main__":
    main()
