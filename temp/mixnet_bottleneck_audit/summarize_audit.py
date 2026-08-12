"""Aggregate audit runs and turn effect patterns into a conservative evidence table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from audit_common import write_csv, write_json


TEMP_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MixNet bottleneck audit results.")
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*/seed_*/metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("dry_run"):
            continue
        job = payload["job"]
        test = payload["split_metrics"]["test"]
        train = payload["split_metrics"]["train"]
        row = {
            "job": job["name"],
            "hypothesis": job["hypothesis"],
            "seed": int(payload["seed"]),
            "train_policy": job["train_policy"],
            "loss_type": job["loss_type"],
            "time_codes": "-".join(job["time_codes"]),
            "parameters_trainable": payload["parameters_trainable"],
            "train_patch_accuracy": train["patch"]["accuracy"],
            "test_patch_accuracy": test["patch"]["accuracy"],
            "test_patch_f1": test["patch"]["macro_f1"],
            "test_patch_qwk": test["patch"]["qwk"],
            "test_patch_nll": test["patch"]["nll"],
            "test_patch_ece": test["patch"]["ece_15bin"],
            "test_max_abs_logit": test["patch"]["max_abs_logit"],
            "test_mean_top1_top2_logit_margin": test["patch"]["mean_top1_top2_logit_margin"],
            "test_parent_accuracy": test["parent"]["accuracy"],
            "test_parent_accuracy_wilson95_low": test["parent"]["accuracy_wilson95_low"],
            "test_parent_accuracy_wilson95_high": test["parent"]["accuracy_wilson95_high"],
            "test_parent_majority_accuracy": test["parent"]["majority_vote_accuracy"],
            "test_parent_count": test["parent"]["sample_count"],
            "test_parent_consistency": test["parent"]["mean_patch_consistency"],
            "adjacent_error_fraction": test["patch"].get("adjacent_error_fraction"),
            "far_error_count": test["patch"].get("far_error_count"),
        }
        rows.append(row)
    return rows


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["job"]].append(row)
    metrics = (
        "train_patch_accuracy",
        "test_patch_accuracy",
        "test_patch_f1",
        "test_patch_qwk",
        "test_patch_nll",
        "test_patch_ece",
        "test_max_abs_logit",
        "test_mean_top1_top2_logit_margin",
        "test_parent_accuracy",
        "test_parent_accuracy_wilson95_low",
        "test_parent_accuracy_wilson95_high",
        "test_parent_majority_accuracy",
        "test_parent_consistency",
        "adjacent_error_fraction",
        "far_error_count",
    )
    output: list[dict[str, Any]] = []
    for job, job_rows in grouped.items():
        first = job_rows[0]
        aggregate_row: dict[str, Any] = {
            "job": job,
            "hypothesis": first["hypothesis"],
            "time_codes": first["time_codes"],
            "train_policy": first["train_policy"],
            "loss_type": first["loss_type"],
            "seed_count": len(job_rows),
            "seeds": ",".join(str(row["seed"]) for row in job_rows),
            "test_parent_count_per_seed": first["test_parent_count"],
        }
        for metric in metrics:
            values = [float(row[metric]) for row in job_rows if row.get(metric) is not None]
            aggregate_row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            aggregate_row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        output.append(aggregate_row)
    return sorted(output, key=lambda row: row["job"])


def job_map(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["job"]: row for row in rows}


def evidence_rows(aggregated: Sequence[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    jobs = job_map(aggregated)
    evidence: list[dict[str, Any]] = []

    binary_names = [f"B0{index}_" for index in range(1, 7)]
    adjacent = [jobs[name] for name in jobs for prefix in binary_names[:4] if name.startswith(prefix)]
    far = [jobs[name] for name in jobs for prefix in binary_names[4:] if name.startswith(prefix)]
    if adjacent and far:
        adjacent_mean = float(np.mean([row["test_patch_accuracy_mean"] for row in adjacent]))
        far_mean = float(np.mean([row["test_patch_accuracy_mean"] for row in far]))
        gap = far_mean - adjacent_mean
        evidence.append(
            {
                "hypothesis": "H1/H2 local ambiguity or adjacent visual overlap",
                "status": "supported" if gap >= 0.10 and adjacent_mean < 0.90 else "not_decisive",
                "primary_effect": gap,
                "evidence": f"far-pair acc mean={far_mean:.4f}, adjacent-pair mean={adjacent_mean:.4f}, gap={gap:.4f}",
                "caveat": "Binary separability alone cannot distinguish within-parent label mismatch from population-level adjacent overlap.",
            }
        )

    baseline = jobs.get("M01_five_class_ce")
    head = jobs.get("F01_head_only")
    last = jobs.get("F02_last_stage")
    if baseline and head and last:
        full_acc = baseline["test_patch_accuracy_mean"]
        head_gap = full_acc - head["test_patch_accuracy_mean"]
        last_gap = full_acc - last["test_patch_accuracy_mean"]
        memorization_gap = baseline["train_patch_accuracy_mean"] - full_acc
        status = "supported" if head_gap <= 0.03 and memorization_gap >= 0.15 else "not_decisive"
        evidence.append(
            {
                "hypothesis": "H3 small-independent-sample memorization",
                "status": status,
                "primary_effect": head_gap,
                "evidence": f"full-head_only test gap={head_gap:.4f}, full-last_stage gap={last_gap:.4f}, full train-test gap={memorization_gap:.4f}",
                "caveat": "A weak frozen classifier may also indicate domain adaptation is necessary; inspect all three policies and multiple seeds.",
            }
        )

    loss_names = ("L02_label_smoothing_0p1", "L03_adjacent_soft_0p2", "L04_ce_ordinal_aux_0p5")
    if baseline and any(name in jobs for name in loss_names):
        candidates = [jobs[name] for name in loss_names if name in jobs]
        best = max(candidates, key=lambda row: row["test_patch_accuracy_mean"])
        acc_gain = best["test_patch_accuracy_mean"] - baseline["test_patch_accuracy_mean"]
        nll_gain = baseline["test_patch_nll_mean"] - best["test_patch_nll_mean"]
        evidence.append(
            {
                "hypothesis": "H4 hard CE supervision mismatch",
                "status": "supported" if acc_gain >= 0.02 and nll_gain > 0 else "not_decisive",
                "primary_effect": acc_gain,
                "evidence": f"best diagnostic={best['job']}, accuracy gain={acc_gain:.4f}, NLL reduction={nll_gain:.4f}",
                "caveat": "These are diagnostic relaxations, not a claim of methodological novelty.",
            }
        )

    feature_files = list(run_dir.glob("**/within_vs_between.csv"))
    if feature_files:
        ratios: list[float] = []
        for feature_file in feature_files:
            with feature_file.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if int(row["is_adjacent"]):
                        ratios.append(float(row["within_to_between_ratio"]))
        if ratios:
            mean_ratio = float(np.mean(ratios))
            evidence.append(
                {
                    "hypothesis": "H1 parent-internal heterogeneity vs adjacent-class distance",
                    "status": "supported" if mean_ratio >= 0.50 else "not_decisive",
                    "primary_effect": mean_ratio,
                    "evidence": f"mean adjacent within/between feature ratio={mean_ratio:.4f}",
                    "caveat": "Geometry is checkpoint-dependent and test has few independent parents; repeat per seed.",
                }
            )

    color_files = list(run_dir.glob("M01_five_class_ce/seed_*/color_robustness.json"))
    if color_files:
        variant_accs: dict[str, list[float]] = defaultdict(list)
        for path in color_files:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for variant, metrics in payload.items():
                variant_accs[variant].append(float(metrics["patch"]["accuracy"]))
        means = {name: float(np.mean(values)) for name, values in variant_accs.items()}
        rgb = means.get("rgb")
        if rgb is not None:
            stress = means.get("color_jitter_stress", rgb)
            grayscale = means.get("grayscale", rgb)
            evidence.append(
                {
                    "hypothesis": "H5 color/style sensitivity",
                    "status": "supported" if rgb - stress >= 0.10 else "not_decisive",
                    "primary_effect": rgb - stress,
                    "evidence": f"RGB acc={rgb:.4f}, deterministic color-stress acc={stress:.4f}, grayscale acc={grayscale:.4f}",
                    "caveat": "Grayscale loss proves color dependence, which can be legitimate. Only controlled nuisance perturbations diagnose brittleness; neither alone proves a shortcut.",
                }
            )
    return evidence


def main() -> None:
    args = parse_args()
    rows = read_metrics(args.run_dir)
    if not rows:
        raise ValueError(f"No non-dry-run metrics found under {args.run_dir}")
    aggregated = aggregate(rows)
    evidence = evidence_rows(aggregated, args.run_dir)
    write_csv(args.run_dir / "aggregate_summary.csv", aggregated)
    write_csv(args.run_dir / "evidence_matrix.csv", evidence)
    write_json(args.run_dir / "evidence_matrix.json", evidence)
    print(f"Wrote {args.run_dir / 'aggregate_summary.csv'}")
    print(f"Wrote {args.run_dir / 'evidence_matrix.csv'}")


if __name__ == "__main__":
    main()
