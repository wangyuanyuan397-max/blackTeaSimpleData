"""Paired statistical analysis for MixNet-S SAA three-seed predictions."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs_01234_BaSic_grid30_408"
RUN_PATTERN = re.compile(
    r"^(saa_rep_(baseline|down_e1_g4|down_e2_g1|mid_e1_g1)_seed(\d+))_"
    r"(\d{8}_\d{6})(?:_\d+)?$"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pair SAA candidates with the same-seed baseline by image_path and "
            "compute McNemar plus paired bootstrap statistics."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Batch run directory (default: {DEFAULT_RUNS_ROOT}).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of paired bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2026,
        help="Bootstrap random seed (default: 2026).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    return parser.parse_args()


def find_latest_prediction_runs(runs_root: Path) -> dict[tuple[str, int], Path]:
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")
    latest: dict[tuple[str, int], tuple[str, Path]] = {}
    for directory in runs_root.iterdir():
        if not directory.is_dir():
            continue
        match = RUN_PATTERN.fullmatch(directory.name)
        if match is None:
            continue
        prediction_path = directory / "test_predictions.csv"
        if not prediction_path.is_file():
            continue
        family = match.group(2)
        seed = int(match.group(3))
        timestamp = match.group(4)
        key = (family, seed)
        if key not in latest or timestamp > latest[key][0]:
            latest[key] = (timestamp, prediction_path)
    return {key: value[1] for key, value in latest.items()}


def load_predictions(path: Path) -> dict[str, tuple[int, int]]:
    predictions: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            image_path = str(row["image_path"])
            if image_path in predictions:
                raise ValueError(f"Duplicate image_path in {path}: {image_path}")
            predictions[image_path] = (
                int(row["true_label"]),
                int(row["pred_label"]),
            )
    if not predictions:
        raise ValueError(f"Prediction CSV is empty: {path}")
    return predictions


def align_predictions(
    baseline_path: Path,
    candidate_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = load_predictions(baseline_path)
    candidate = load_predictions(candidate_path)
    if set(baseline) != set(candidate):
        baseline_only = sorted(set(baseline) - set(candidate))[:3]
        candidate_only = sorted(set(candidate) - set(baseline))[:3]
        raise ValueError(
            "Prediction path sets differ: "
            f"baseline_only={baseline_only}, candidate_only={candidate_only}"
        )
    ordered_paths = sorted(baseline)
    true_labels = np.asarray([baseline[path][0] for path in ordered_paths], dtype=np.int64)
    baseline_predictions = np.asarray(
        [baseline[path][1] for path in ordered_paths], dtype=np.int64
    )
    candidate_true_labels = np.asarray(
        [candidate[path][0] for path in ordered_paths], dtype=np.int64
    )
    if not np.array_equal(true_labels, candidate_true_labels):
        raise ValueError(
            f"True labels differ between {baseline_path} and {candidate_path}."
        )
    candidate_predictions = np.asarray(
        [candidate[path][1] for path in ordered_paths], dtype=np.int64
    )
    return true_labels, baseline_predictions, candidate_predictions


def exact_mcnemar_p_value(baseline_only: int, candidate_only: int) -> float:
    discordant = int(baseline_only) + int(candidate_only)
    if discordant == 0:
        return 1.0
    lower = min(int(baseline_only), int(candidate_only))
    term = math.pow(0.5, discordant)
    tail = term
    for successes in range(1, lower + 1):
        term *= (discordant - successes + 1) / successes
        tail += term
    return min(1.0, 2.0 * tail)


def paired_mean_bootstrap_ci(
    paired_deltas: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    sample_count = int(paired_deltas.shape[0])
    means = np.empty(samples, dtype=np.float64)
    chunk_size = 1000
    for start in range(0, samples, chunk_size):
        end = min(samples, start + chunk_size)
        indices = rng.integers(
            0,
            sample_count,
            size=(end - start, sample_count),
        )
        means[start:end] = paired_deltas[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def compare_predictions(
    family: str,
    seed: int,
    baseline_path: Path,
    candidate_path: Path,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    true_labels, baseline_predictions, candidate_predictions = align_predictions(
        baseline_path,
        candidate_path,
    )
    labels = np.unique(true_labels)
    baseline_correct = baseline_predictions == true_labels
    candidate_correct = candidate_predictions == true_labels
    baseline_only = int(np.logical_and(baseline_correct, ~candidate_correct).sum())
    candidate_only = int(np.logical_and(~baseline_correct, candidate_correct).sum())
    accuracy_deltas = candidate_correct.astype(np.float64) - baseline_correct.astype(
        np.float64
    )
    baseline_abs_errors = np.abs(baseline_predictions - true_labels).astype(np.float64)
    candidate_abs_errors = np.abs(candidate_predictions - true_labels).astype(np.float64)
    mae_deltas = candidate_abs_errors - baseline_abs_errors
    accuracy_ci = paired_mean_bootstrap_ci(accuracy_deltas, bootstrap_samples, rng)
    mae_ci = paired_mean_bootstrap_ci(mae_deltas, bootstrap_samples, rng)

    baseline_accuracy = float(baseline_correct.mean())
    candidate_accuracy = float(candidate_correct.mean())
    baseline_mae = float(baseline_abs_errors.mean())
    candidate_mae = float(candidate_abs_errors.mean())
    baseline_f1 = float(
        f1_score(true_labels, baseline_predictions, labels=labels, average="macro")
    )
    candidate_f1 = float(
        f1_score(true_labels, candidate_predictions, labels=labels, average="macro")
    )
    baseline_qwk = float(
        cohen_kappa_score(
            true_labels,
            baseline_predictions,
            labels=labels,
            weights="quadratic",
        )
    )
    candidate_qwk = float(
        cohen_kappa_score(
            true_labels,
            candidate_predictions,
            labels=labels,
            weights="quadratic",
        )
    )
    return {
        "candidate_family": family,
        "seed": seed,
        "sample_count": int(true_labels.shape[0]),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": candidate_accuracy - baseline_accuracy,
        "accuracy_delta_ci95_low": accuracy_ci[0],
        "accuracy_delta_ci95_high": accuracy_ci[1],
        "baseline_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "mae_delta": candidate_mae - baseline_mae,
        "mae_delta_ci95_low": mae_ci[0],
        "mae_delta_ci95_high": mae_ci[1],
        "baseline_macro_f1": baseline_f1,
        "candidate_macro_f1": candidate_f1,
        "macro_f1_delta": candidate_f1 - baseline_f1,
        "baseline_qwk": baseline_qwk,
        "candidate_qwk": candidate_qwk,
        "qwk_delta": candidate_qwk - baseline_qwk,
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "mcnemar_discordant": baseline_only + candidate_only,
        "mcnemar_exact_p": exact_mcnemar_p_value(
            baseline_only,
            candidate_only,
        ),
        "baseline_predictions_csv": str(baseline_path),
        "candidate_predictions_csv": str(candidate_path),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_arguments()
    runs_root = args.runs_root.resolve()
    prediction_runs = find_latest_prediction_runs(runs_root)
    seeds = sorted(seed for family, seed in prediction_runs if family == "baseline")
    if not seeds:
        raise FileNotFoundError(
            f"No saa_rep_baseline_seed* test_predictions.csv found in {runs_root}"
        )
    candidate_families = ("down_e1_g4", "down_e2_g1", "mid_e1_g1")
    rng = np.random.default_rng(int(args.random_seed))
    rows = []
    missing = []
    for seed in seeds:
        baseline_path = prediction_runs[("baseline", seed)]
        for family in candidate_families:
            candidate_path = prediction_runs.get((family, seed))
            if candidate_path is None:
                missing.append(f"{family}/seed{seed}")
                continue
            rows.append(
                compare_predictions(
                    family=family,
                    seed=seed,
                    baseline_path=baseline_path,
                    candidate_path=candidate_path,
                    bootstrap_samples=int(args.bootstrap_samples),
                    rng=rng,
                )
            )
    if missing:
        raise FileNotFoundError(f"Missing paired prediction runs: {missing}")
    if not rows:
        raise RuntimeError("No complete baseline/candidate prediction pairs found.")
    output_path = (
        args.output.resolve()
        if args.output is not None
        else runs_root / "mixnet_saa_paired_statistics.csv"
    )
    write_rows(output_path, rows)
    print(f"Paired SAA statistics: {output_path}")
    print(f"Comparisons: {len(rows)}; bootstrap samples per comparison: {args.bootstrap_samples}")


if __name__ == "__main__":
    main()
