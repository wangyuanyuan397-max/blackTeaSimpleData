"""Run a small finetuning hyperparameter sweep from one SSL checkpoint.

Use this on the server after SimSiam pretraining has produced
mixnet_s_simsiam_backbone.pth. The script runs variants sequentially and writes
a compact CSV summary for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


FINETUNE_SCRIPT = Path("temp") / "mixnet_simsiam_ssl" / "finetune_mixnet_classifier.py"


VARIANTS = [
    {
        "name": "resize_lr3e4_wd1e4_acc",
        "description": "Weak resize+flip augmentation, higher AdamW LR, lower WD, select by val_acc.",
        "args": [
            "--train-transform-mode", "resize_flip",
            "--train-jitter-strength", "0.02",
            "--lr", "0.0003",
            "--weight-decay", "0.0001",
            "--epochs", "100",
            "--patience", "25",
            "--selection-metric", "val_acc",
        ],
    },
    {
        "name": "resize_lr1e4_wd1e5_acc",
        "description": "Weak augmentation, baseline LR, much lower WD, select by val_acc.",
        "args": [
            "--train-transform-mode", "resize_flip",
            "--train-jitter-strength", "0.02",
            "--lr", "0.0001",
            "--weight-decay", "0.00001",
            "--epochs", "100",
            "--patience", "25",
            "--selection-metric", "val_acc",
        ],
    },
    {
        "name": "crop095_lr3e4_wd1e4_acc",
        "description": "Milder crop than default, higher AdamW LR, lower WD, select by val_acc.",
        "args": [
            "--train-transform-mode", "crop",
            "--train-crop-min", "0.95",
            "--train-jitter-strength", "0.04",
            "--lr", "0.0003",
            "--weight-decay", "0.0001",
            "--epochs", "100",
            "--patience", "25",
            "--selection-metric", "val_acc",
        ],
    },
    {
        "name": "headfast_resize_lr1e4_wd1e4_acc",
        "description": "Weak augmentation with a 5x faster classifier head.",
        "args": [
            "--train-transform-mode", "resize_flip",
            "--train-jitter-strength", "0.02",
            "--lr", "0.0001",
            "--backbone-lr-mult", "1.0",
            "--head-lr-mult", "5.0",
            "--weight-decay", "0.0001",
            "--epochs", "100",
            "--patience", "25",
            "--selection-metric", "val_acc",
        ],
    },
    {
        "name": "resize_lr3e4_wd1e4_qwk",
        "description": "Same as first variant, but save best checkpoint by val_qwk.",
        "args": [
            "--train-transform-mode", "resize_flip",
            "--train-jitter-strength", "0.02",
            "--lr", "0.0003",
            "--weight-decay", "0.0001",
            "--epochs", "100",
            "--patience", "25",
            "--selection-metric", "val_qwk",
        ],
    },
    {
        "name": "sgd_resize_lr5e3_wd1e4_acc",
        "description": "Weak augmentation with SGD; slower but useful if AdamW plateaus.",
        "args": [
            "--train-transform-mode", "resize_flip",
            "--train-jitter-strength", "0.02",
            "--optimizer", "sgd",
            "--lr", "0.005",
            "--weight-decay", "0.0001",
            "--warmup-epochs", "5",
            "--epochs", "150",
            "--patience", "30",
            "--selection-metric", "val_acc",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recommended MixNet-S SSL finetune sweeps.")
    parser.add_argument("--dataset-root", default="datasets_01234_BaSic")
    parser.add_argument(
        "--ssl-checkpoint",
        default=(
            "temp/mixnet_simsiam_ssl/runs/simsiam_mixnet_s_basic408/"
            "mixnet_s_simsiam_backbone.pth"
        ),
    )
    parser.add_argument("--runs-root", default="temp/mixnet_simsiam_ssl/runs/tuning")
    parser.add_argument("--python", default="python")
    parser.add_argument("--image-size", type=int, default=408)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--variant", action="append", default=[], help="Variant name to run. Repeatable.")
    parser.add_argument("--exclude-sgd", action="store_true", help="Skip the slower SGD variant.")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Rerun variants even if summary.json exists.")
    parser.add_argument("--list", action="store_true", help="List variants and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--smoke", action="store_true", help="Run each selected variant for a tiny smoke check.")
    return parser.parse_args()


def selected_variants(args: argparse.Namespace) -> list[dict]:
    selected = VARIANTS
    if args.exclude_sgd:
        selected = [variant for variant in selected if not variant["name"].startswith("sgd_")]
    if args.variant:
        wanted = set(args.variant)
        names = {variant["name"] for variant in selected}
        missing = sorted(wanted - names)
        if missing:
            raise ValueError(f"Unknown variant(s): {missing}. Use --list to see valid names.")
        selected = [variant for variant in selected if variant["name"] in wanted]
    if args.max_runs > 0:
        selected = selected[: args.max_runs]
    return selected


def command_for_variant(args: argparse.Namespace, variant: dict) -> tuple[list[str], Path]:
    output_dir = Path(args.runs_root) / variant["name"]
    cmd = [
        args.python,
        str(FINETUNE_SCRIPT),
        "--dataset-root", args.dataset_root,
        "--ssl-checkpoint", args.ssl_checkpoint,
        "--output-dir", output_dir.as_posix(),
        "--image-size", str(args.image_size),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
        *variant["args"],
    ]
    if args.smoke:
        cmd.extend(["--dry-run", "--max-samples", "40", "--epochs", "1", "--num-workers", "0"])
    return cmd, output_dir


def read_summary(output_dir: Path, variant: dict) -> dict | None:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    best_val = summary.get("best_val_metrics", {})
    test = summary.get("test_metrics", {})
    return {
        "variant": variant["name"],
        "description": variant["description"],
        "best_epoch": summary.get("best_epoch"),
        "selection_metric": summary.get("selection_metric"),
        "best_selection_value": summary.get("best_selection_value"),
        "best_val_acc": best_val.get("accuracy"),
        "best_val_qwk": best_val.get("qwk"),
        "best_val_macro_f1": best_val.get("macro_f1"),
        "best_val_loss": best_val.get("loss"),
        "test_acc": test.get("accuracy"),
        "test_qwk": test.get("qwk"),
        "test_macro_f1": test.get("macro_f1"),
        "test_mae": test.get("mae"),
        "output_dir": str(output_dir),
    }


def write_results_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "best_epoch",
        "selection_metric",
        "best_selection_value",
        "best_val_acc",
        "best_val_qwk",
        "best_val_macro_f1",
        "best_val_loss",
        "test_acc",
        "test_qwk",
        "test_macro_f1",
        "test_mae",
        "description",
        "output_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_results(rows: list[dict]) -> None:
    if not rows:
        print("No completed summaries found yet.")
        return

    def fmt(value) -> str:
        return "NA" if value is None else f"{float(value):.4f}"

    sorted_rows = sorted(rows, key=lambda row: row.get("test_acc") or 0.0, reverse=True)
    print("\nCompleted variants sorted by test_acc:")
    for row in sorted_rows:
        print(
            f"{row['variant']}: "
            f"val_acc={fmt(row.get('best_val_acc'))} "
            f"val_qwk={fmt(row.get('best_val_qwk'))} "
            f"test_acc={fmt(row.get('test_acc'))} "
            f"test_qwk={fmt(row.get('test_qwk'))} "
            f"best_epoch={row.get('best_epoch')}"
        )


def main() -> None:
    args = parse_args()
    variants = selected_variants(args)

    if args.list:
        for variant in variants:
            print(f"{variant['name']}: {variant['description']}")
        return

    completed: list[dict] = []
    for variant in variants:
        cmd, output_dir = command_for_variant(args, variant)
        summary = read_summary(output_dir, variant)
        if summary is not None and not args.force:
            print(f"Skipping completed variant: {variant['name']}")
            completed.append(summary)
            continue

        printable_cmd = subprocess.list2cmdline(cmd)
        print(f"\n=== Running {variant['name']} ===")
        print(printable_cmd)
        if args.dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(f"Variant failed: {variant['name']} (exit code {result.returncode})")
        summary = read_summary(output_dir, variant)
        if summary is not None:
            completed.append(summary)

    if not args.dry_run and completed:
        results_path = Path(args.runs_root) / "sweep_results.csv"
        write_results_csv(completed, results_path)
        print_results(completed)
        print(f"\nSaved sweep summary: {results_path}")


if __name__ == "__main__":
    main()
