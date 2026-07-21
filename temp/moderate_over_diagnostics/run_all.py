"""Run the Moderate-vs-Over diagnostics in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import create_run_dir


THIS_DIR = Path(__file__).resolve().parent


def run_step(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=THIS_DIR, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Moderate/Over diagnostic experiments.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = create_run_dir(args.run_dir)
    python = sys.executable

    run_step([python, "build_metadata.py", "--run-dir", str(run_dir)])
    run_step([python, "split_sources.py", "--run-dir", str(run_dir), "--seed", str(args.seed)])

    base_train = [
        python, "train_binary_cnn.py", "--run-dir", str(run_dir), "--model", args.model,
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch-size", str(args.batch_size), "--eval-batch-size", str(args.eval_batch_size),
        "--num-workers", str(args.num_workers), "--device", args.device, "--seed", str(args.seed),
    ]
    if not args.skip_cnn:
        for dataset, variant in (("original", "rgb"), ("patch", "rgb"), ("patch", "gray"), ("patch", "blur")):
            run_step(base_train + ["--dataset", dataset, "--variant", variant])
        run_step([python, "diagnose_patch_consistency.py", "--run-dir", str(run_dir)])
    if not args.skip_features:
        run_step([python, "diagnose_color_texture.py", "--run-dir", str(run_dir)])

    print(f"\nAll diagnostics finished: {run_dir}")


if __name__ == "__main__":
    main()
