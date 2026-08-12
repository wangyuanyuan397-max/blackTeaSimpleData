"""Run a controlled, parent-safe audit of the MixNet-S 76% bottleneck."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from audit_common import (
    JobSpec,
    PatchDataset,
    aggregate_parent_predictions,
    classification_metrics,
    compute_loss,
    configure_train_policy,
    create_mixnet,
    job_to_dict,
    load_and_audit_manifest,
    make_loader,
    prediction_rows,
    resolve_device,
    safe_name,
    seed_everything,
    set_train_mode,
    write_csv,
    write_json,
)


TEMP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEMP_ROOT.parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets_01234_BaSic"
DEFAULT_MANIFEST = DEFAULT_DATASET_ROOT / "grid30_crop_manifest.csv"
DEFAULT_RESULTS_ROOT = TEMP_ROOT / "results"


JOBS: tuple[JobSpec, ...] = (
    JobSpec("M01_five_class_ce", "reference", "Five-class MixNet-S CE baseline.", ("00", "10", "20", "30", "40")),
    JobSpec("B01_00_vs_10", "H1_H2", "Adjacent pair 00 vs 10.", ("00", "10")),
    JobSpec("B02_10_vs_20", "H1_H2", "Adjacent pair 10 vs 20.", ("10", "20")),
    JobSpec("B03_20_vs_30", "H1_H2", "Adjacent pair 20 vs 30.", ("20", "30")),
    JobSpec("B04_30_vs_40", "H1_H2", "Adjacent pair 30 vs 40.", ("30", "40")),
    JobSpec("B05_00_vs_40", "H1_H2", "Far-pair control 00 vs 40.", ("00", "40")),
    JobSpec("B06_10_vs_40", "H1_H2", "Far-pair control 10 vs 40.", ("10", "40")),
    JobSpec("F01_head_only", "H3", "Freeze the entire pretrained backbone.", ("00", "10", "20", "30", "40"), train_policy="head_only"),
    JobSpec("F02_last_stage", "H3", "Train final MixNet stage, head convolution, and classifier.", ("00", "10", "20", "30", "40"), train_policy="last_stage"),
    JobSpec("L02_label_smoothing_0p1", "H4", "Uniform label smoothing diagnostic.", ("00", "10", "20", "30", "40"), label_smoothing=0.10),
    JobSpec("L03_adjacent_soft_0p2", "H4", "Put 0.20 target mass on adjacent stages only.", ("00", "10", "20", "30", "40"), loss_type="adjacent_soft", adjacent_alpha=0.20),
    JobSpec("L04_ce_ordinal_aux_0p5", "H4", "CE plus expected-class SmoothL1 ordinal auxiliary.", ("00", "10", "20", "30", "40"), loss_type="ce_ordinal_aux", ordinal_weight=0.50),
)


COLOR_VARIANTS = ("rgb", "grayscale", "color_jitter_stress", "clahe_luminance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled MixNet-S bottleneck audit; all outputs stay under this temp directory."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--jobs", nargs="*", default=None, help="Job names. Omit for core jobs.")
    parser.add_argument("--groups", nargs="*", choices=("core", "binary", "freeze", "loss"), default=("core",))
    parser.add_argument("--seeds", nargs="+", type=int, default=(2026,))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=408)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--selection-metric", choices=("patch_accuracy", "patch_qwk"), default="patch_accuracy")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--keep-pth", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--evaluate-colors", action="store_true", help="Evaluate baseline checkpoint on four color variants.")
    return parser.parse_args()


def select_jobs(args: argparse.Namespace) -> list[JobSpec]:
    by_name = {job.name: job for job in JOBS}
    if args.jobs:
        unknown = sorted(set(args.jobs) - set(by_name))
        if unknown:
            raise ValueError(f"Unknown jobs: {unknown}")
        return [by_name[name] for name in args.jobs]
    groups = set(args.groups)
    names: list[str] = []
    if "core" in groups:
        names.extend(["M01_five_class_ce", "B01_00_vs_10", "B02_10_vs_20", "B03_20_vs_30", "B04_30_vs_40", "B05_00_vs_40", "B06_10_vs_40", "F01_head_only", "F02_last_stage", "L02_label_smoothing_0p1", "L03_adjacent_soft_0p2", "L04_ce_ordinal_aux_0p5"])
    if "binary" in groups:
        names.extend([job.name for job in JOBS if job.name.startswith("B")])
    if "freeze" in groups:
        names.extend(["M01_five_class_ce", "F01_head_only", "F02_last_stage"])
    if "loss" in groups:
        names.extend(["M01_five_class_ce", "L02_label_smoothing_0p1", "L03_adjacent_soft_0p2", "L04_ce_ordinal_aux_0p5"])
    unique_names = list(dict.fromkeys(names))
    return [by_name[name] for name in unique_names]


def evaluate_model(
    model: torch.nn.Module,
    dataset: PatchDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=seed,
    )
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for images, targets, _ in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            logits_all.append(logits.cpu().numpy())
            labels.append(targets.numpy())
    label_array = np.concatenate(labels)
    probability_array = np.concatenate(probabilities)
    logit_array = np.concatenate(logits_all)
    patch_metrics = classification_metrics(label_array, probability_array, dataset.classes)
    sorted_logits = np.sort(logit_array, axis=1)
    patch_metrics.update(
        {
            "max_abs_logit": float(np.abs(logit_array).max()),
            "mean_abs_logit": float(np.abs(logit_array).mean()),
            "mean_top1_top2_logit_margin": float((sorted_logits[:, -1] - sorted_logits[:, -2]).mean()),
        }
    )
    parent_metrics, parent_rows = aggregate_parent_predictions(dataset, label_array, probability_array)
    rows = prediction_rows(dataset, label_array, probability_array)
    return patch_metrics, parent_metrics, rows, parent_rows


def train_one(
    spec: JobSpec,
    *,
    samples,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(seed)
    datasets = {
        split: PatchDataset(
            samples,
            split=split,
            time_codes=spec.time_codes,
            image_size=args.image_size,
            training=split == "train",
            max_parents_per_class=1 if args.dry_run else None,
        )
        for split in ("train", "val", "test")
    }
    train_eval_dataset = PatchDataset(
        samples,
        split="train",
        time_codes=spec.time_codes,
        image_size=args.image_size,
        training=False,
        max_parents_per_class=1 if args.dry_run else None,
    )
    train_loader = make_loader(
        datasets["train"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=seed,
    )
    model = create_mixnet(len(spec.time_codes), pretrained=not args.dry_run).to(device)
    parameter_counts = configure_train_policy(model, spec.train_policy)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.min_learning_rate)
    if warmup_epochs:
        warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
        scheduler = SequentialLR(optimizer, (warmup, cosine), milestones=(warmup_epochs,))
    else:
        scheduler = cosine

    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    epochs_to_run = 1 if args.dry_run else args.epochs
    for epoch in range(epochs_to_run):
        set_train_mode(model, spec.train_policy)
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = compute_loss(logits, labels, spec)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            total += len(labels)

        val_patch, val_parent, _, _ = evaluate_model(
            model,
            datasets["val"],
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            seed=seed,
        )
        train_loss = total_loss / max(total, 1)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": correct / max(total, 1),
            "val_patch_accuracy": val_patch["accuracy"],
            "val_patch_qwk": val_patch["qwk"],
            "val_patch_nll": val_patch["nll"],
            "val_patch_ece": val_patch["ece_15bin"],
            "val_max_abs_logit": val_patch["max_abs_logit"],
            "val_mean_top1_top2_logit_margin": val_patch["mean_top1_top2_logit_margin"],
            "val_adjacent_error_fraction": val_patch.get("adjacent_error_fraction"),
            "val_far_error_count": val_patch.get("far_error_count"),
            "val_parent_accuracy": val_parent["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        score_key = "accuracy" if args.selection_metric == "patch_accuracy" else "qwk"
        score = float(val_patch[score_key])
        if not np.isfinite(score):
            score = -1.0
        if best_state is None or score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        print(
            f"[{spec.name} seed={seed}] epoch {epoch + 1:03d}/{epochs_to_run:03d} "
            f"train={row['train_accuracy']:.4f} val={val_patch['accuracy']:.4f} "
            f"parent={val_parent['accuracy']:.4f} nll={val_patch['nll']:.4f}"
        )
        if not args.dry_run and epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state.")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "history.csv", history)
    split_metrics: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        evaluation_dataset = train_eval_dataset if split == "train" else datasets[split]
        patch_metrics, parent_metrics, rows, parent_rows = evaluate_model(
            model,
            evaluation_dataset,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            seed=seed,
        )
        split_metrics[split] = {"patch": patch_metrics, "parent": parent_metrics}
        write_csv(output_dir / f"{split}_patch_predictions.csv", rows)
        write_csv(output_dir / f"{split}_parent_predictions.csv", parent_rows)

    color_metrics: dict[str, Any] = {}
    if args.evaluate_colors and spec.name == "M01_five_class_ce":
        for variant in COLOR_VARIANTS:
            variant_dataset = PatchDataset(
                samples,
                split="test",
                time_codes=spec.time_codes,
                image_size=args.image_size,
                training=False,
                color_variant=variant,
            )
            patch_metrics, parent_metrics, _, _ = evaluate_model(
                model,
                variant_dataset,
                device=device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                seed=seed,
            )
            color_metrics[variant] = {"patch": patch_metrics, "parent": parent_metrics}
        write_json(output_dir / "color_robustness.json", color_metrics)

    result = {
        "job": job_to_dict(spec),
        "seed": seed,
        "dry_run": bool(args.dry_run),
        "device": str(device),
        "image_size": args.image_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "selection_metric": args.selection_metric,
        "elapsed_seconds": time.time() - started,
        **parameter_counts,
        "split_metrics": split_metrics,
        "color_robustness": color_metrics,
    }
    write_json(output_dir / "metrics.json", result)
    write_json(output_dir / "resolved_job.json", {**job_to_dict(spec), "runtime": vars(args)})
    if args.keep_pth:
        torch.save(
            {
                "model_state_dict": best_state,
                "job": job_to_dict(spec),
                "seed": seed,
                "best_epoch": best_epoch,
                "class_names": list(spec.time_codes),
                "image_size": args.image_size,
            },
            output_dir / "best_model.pth",
        )
    del model, optimizer, scheduler, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    test = result["split_metrics"]["test"]
    train = result["split_metrics"]["train"]
    val = result["split_metrics"]["val"]
    return {
        "job": result["job"]["name"],
        "hypothesis": result["job"]["hypothesis"],
        "seed": result["seed"],
        "train_policy": result["job"]["train_policy"],
        "loss_type": result["job"]["loss_type"],
        "time_codes": "-".join(result["job"]["time_codes"]),
        "parameters_trainable": result["parameters_trainable"],
        "best_epoch": result["best_epoch"],
        "train_patch_accuracy": train["patch"]["accuracy"],
        "val_patch_accuracy": val["patch"]["accuracy"],
        "test_patch_accuracy": test["patch"]["accuracy"],
        "test_patch_macro_f1": test["patch"]["macro_f1"],
        "test_patch_qwk": test["patch"]["qwk"],
        "test_patch_nll": test["patch"]["nll"],
        "test_patch_ece": test["patch"]["ece_15bin"],
        "test_max_abs_logit": test["patch"]["max_abs_logit"],
        "test_mean_top1_top2_logit_margin": test["patch"]["mean_top1_top2_logit_margin"],
        "test_parent_count": test["parent"]["sample_count"],
        "test_parent_accuracy": test["parent"]["accuracy"],
        "test_parent_accuracy_wilson95_low": test["parent"]["accuracy_wilson95_low"],
        "test_parent_accuracy_wilson95_high": test["parent"]["accuracy_wilson95_high"],
        "test_parent_majority_accuracy": test["parent"]["majority_vote_accuracy"],
        "test_parent_qwk": test["parent"]["qwk"],
        "test_mean_parent_consistency": test["parent"]["mean_patch_consistency"],
        "test_adjacent_error_fraction": test["patch"].get("adjacent_error_fraction"),
        "test_far_error_count": test["patch"].get("far_error_count"),
        "elapsed_seconds": result["elapsed_seconds"],
    }


def main() -> None:
    args = parse_args()
    if args.list_jobs:
        for job in JOBS:
            print(f"{job.name:28s} {job.hypothesis:8s} {job.description}")
        return
    selected = select_jobs(args)
    if not selected:
        raise ValueError("No jobs selected.")
    device = resolve_device(args.device)
    samples, manifest_audit = load_and_audit_manifest(args.manifest, args.dataset_root)
    run_name = args.run_name or datetime.now().strftime("audit_%Y%m%d_%H%M%S")
    run_dir = args.results_root / safe_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "manifest_audit.json", manifest_audit)
    write_json(
        run_dir / "run_plan.json",
        {
            "jobs": [job_to_dict(job) for job in selected],
            "seeds": args.seeds,
            "device": str(device),
            "arguments": vars(args),
        },
    )
    print(json.dumps(manifest_audit, ensure_ascii=False, indent=2))
    print(f"Results: {run_dir}")
    all_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        for spec in selected:
            output_dir = run_dir / safe_name(spec.name) / f"seed_{seed}"
            if (output_dir / "metrics.json").is_file():
                print(f"Skip completed job: {spec.name}, seed={seed}")
                with (output_dir / "metrics.json").open("r", encoding="utf-8") as handle:
                    all_results.append(json.load(handle))
                continue
            all_results.append(
                train_one(
                    spec,
                    samples=samples,
                    seed=seed,
                    args=args,
                    output_dir=output_dir,
                    device=device,
                )
            )
            write_csv(run_dir / "summary.csv", [flatten_result(result) for result in all_results])
    print(f"Audit completed: {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
