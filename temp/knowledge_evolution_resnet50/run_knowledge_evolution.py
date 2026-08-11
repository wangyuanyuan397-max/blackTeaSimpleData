"""Run isolated ResNet50 Knowledge Evolution generations.

This script reuses the project's data/model/training components but keeps all
experiment code, configs, and outputs under this temp directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
TRAIN_BATCH_PATH = PROJECT_ROOT / "tools" / "train_batch.py"
DEFAULT_COMMON_CONFIG = EXPERIMENT_DIR / "common_01234_basic_408_train.yaml"
DEFAULT_MODEL_CONFIG = EXPERIMENT_DIR / "resnet50_pretrained_ke.yaml"

for import_path in (PROJECT_ROOT, EXPERIMENT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from ke_core import (  # noqa: E402
    create_kels_masks,
    reset_reset_hypothesis,
    summarize_kels_masks,
    validate_kels_masks,
)


def _load_train_batch_module():
    spec = importlib.util.spec_from_file_location("ke_train_batch_base", TRAIN_BATCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load project trainer: {TRAIN_BATCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ke_train_batch_base"] = module
    spec.loader.exec_module(module)
    return module


train_batch = _load_train_batch_module()


def _load_yaml_mapping(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must contain a YAML mapping: {path}")
    return value


def _validate_common_config(config: dict[str, Any]) -> None:
    required = ("data", "train", "optimizer", "loss")
    missing = [name for name in required if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"Common config is missing mapping sections: {missing}")
    if "model" in config or "models" in config:
        raise ValueError("The common config must not contain model/models.")


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _split_rate_tag(split_rate: float) -> str:
    return f"{split_rate:g}".replace(".", "p")


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _batch_norm_state(model: nn.Module) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        for value_name, value in module.state_dict().items():
            key = f"{module_name}.{value_name}" if module_name else value_name
            snapshot[key] = value.detach().cpu().clone()
    return snapshot


def _verify_batch_norm_unchanged(
    before: dict[str, torch.Tensor],
    model: nn.Module,
) -> int:
    after = _batch_norm_state(model)
    if set(before) != set(after):
        raise RuntimeError("BatchNorm topology changed during the KE reset.")
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    if changed:
        raise RuntimeError(
            "BatchNorm parameters/statistics changed during the KE reset: "
            + ", ".join(changed[:8])
        )
    return len(before)


def _build_runtime_config(
    common_config: dict[str, Any],
    model_config: dict[str, Any],
    model_config_path: Path,
    dataset_root: Path,
    output_dir: Path,
    device: torch.device,
    *,
    epochs: int,
    run_name: str,
):
    common = copy.deepcopy(common_config)
    common["train"]["epochs"] = int(epochs)
    model = copy.deepcopy(model_config)
    model["name"] = run_name
    return train_batch.build_training_config_from_file(
        common,
        model,
        model_config_path,
        dataset_root,
        output_dir,
        device,
    )


def _summary_row(
    metrics: dict[str, Any],
    history: dict[str, list[Any]],
    *,
    run_type: str,
    generation: int | None,
    epochs_requested: int,
) -> dict[str, Any]:
    best = metrics.get("best_validation_metrics") or {}

    def last(key: str):
        values = history.get(key) or []
        return values[-1] if values else None

    return {
        "run_type": run_type,
        "generation": generation,
        "epochs_requested": int(epochs_requested),
        "epochs_completed": len(history.get("train_loss") or []),
        "best_epoch": metrics.get("best_epoch"),
        "best_train_accuracy": metrics.get("best_train_accuracy"),
        "best_validation_accuracy": metrics.get("best_validation_accuracy"),
        "best_validation_mae": best.get("val_mae"),
        "best_validation_qwk": best.get("val_qwk"),
        "train_val_accuracy_gap": metrics.get("train_val_accuracy_gap"),
        "test_accuracy": metrics.get("accuracy"),
        "test_macro_f1": metrics.get("macro_f1"),
        "test_mae": metrics.get("mae"),
        "test_qwk": metrics.get("qwk"),
        "final_train_accuracy": last("train_acc"),
        "final_validation_accuracy": last("val_acc"),
        "training_time_seconds": metrics.get("training_time_seconds"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown_summary(
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    split_rate: float,
    generations: int,
    epochs_per_generation: int,
) -> None:
    columns = (
        "run_type",
        "generation",
        "best_epoch",
        "best_train_accuracy",
        "best_validation_accuracy",
        "test_accuracy",
        "test_macro_f1",
        "test_mae",
        "test_qwk",
        "train_val_accuracy_gap",
    )
    lines = [
        "# Knowledge Evolution run summary",
        "",
        f"- KELS split rate: `{split_rate}`",
        f"- generations: `{generations}`",
        f"- epochs per generation: `{epochs_per_generation}`",
        "- generation transition source: the previous generation's final epoch",
        "- per-generation evaluation source: that generation's best validation checkpoint",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_markdown_value(row.get(column)) for column in columns)
            + " |"
        )
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_generation_trends(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    generation_rows = [row for row in rows if row["run_type"] == "knowledge_evolution"]
    if not generation_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        generations = [int(row["generation"]) for row in generation_rows]
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for key, label in (
            ("best_validation_accuracy", "Best val accuracy"),
            ("test_accuracy", "Test accuracy"),
            ("test_macro_f1", "Test macro-F1"),
        ):
            values = [row.get(key) for row in generation_rows]
            if all(value is not None for value in values):
                axes[0].plot(generations, values, marker="o", label=label)
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Score")
        axes[0].set_xticks(generations)
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        gaps = [row.get("train_val_accuracy_gap") for row in generation_rows]
        if all(value is not None for value in gaps):
            axes[1].plot(generations, gaps, marker="o", color="#c23b22")
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Train - validation accuracy")
        axes[1].set_xticks(generations)
        axes[1].grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(run_dir / "generation_trends.png", dpi=180)
        plt.close(figure)
    except Exception as error:  # Plotting must not invalidate a completed run.
        (run_dir / "plot_warning.txt").write_text(str(error), encoding="utf-8")


def _evaluate_and_archive(
    trainer,
    output_dir: Path,
    training_time_seconds: float,
    *,
    keep_pth: bool,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    history = train_batch.to_builtin(trainer.history)
    metrics = train_batch.evaluate_best_checkpoint(trainer, training_time_seconds)
    train_batch.save_json(output_dir / "history.json", history)
    train_batch.save_json(output_dir / "test_metrics.json", metrics)
    train_batch.save_confusion_matrix_csvs(output_dir, metrics)
    removed = train_batch.cleanup_pth_files(output_dir, keep_pth)
    metrics["keep_pth_files"] = bool(keep_pth)
    metrics["removed_pth_files"] = removed
    train_batch.save_json(output_dir / "test_metrics.json", metrics)
    return metrics, history


def _offline_topology_probe(
    common_config: dict[str, Any],
    model_config: dict[str, Any],
    model_config_path: Path,
    dataset_root: Path,
    split_rate: float,
) -> dict[str, Any]:
    """Build ResNet50 without pretrained weights; never create output files."""
    probe_model_config = copy.deepcopy(model_config)
    probe_model_config["model"]["backbone"]["pretrained"] = False
    preview_dir = EXPERIMENT_DIR / "__dry_run__"
    preview_config = _build_runtime_config(
        common_config,
        probe_model_config,
        model_config_path,
        dataset_root,
        preview_dir,
        torch.device("cpu"),
        epochs=1,
        run_name="ke_offline_topology_probe",
    )
    builder = train_batch.ComponentBuilder(preview_config, torch.device("cpu"), logger=None)
    builder.build_dataloaders()
    model, _ = builder.build_model()
    masks = create_kels_masks(model, split_rate=split_rate)
    return summarize_kels_masks(model, masks, split_rate)


def _run_continuous_control(
    common_config: dict[str, Any],
    model_config: dict[str, Any],
    model_config_path: Path,
    dataset_root: Path,
    run_dir: Path,
    device: torch.device,
    *,
    epochs: int,
    keep_pth: bool,
) -> tuple[dict[str, Any], dict[str, list[Any]], dict[str, Any]]:
    output_dir = run_dir / f"continuous_control_{epochs:03d}_epochs"
    output_dir.mkdir(parents=True, exist_ok=False)
    config = _build_runtime_config(
        common_config,
        model_config,
        model_config_path,
        dataset_root,
        output_dir,
        device,
        epochs=epochs,
        run_name=f"resnet50_continuous_{epochs}_epochs",
    )
    trainer = train_batch.Trainer(config=config, device=device)
    started = time.perf_counter()
    trainer.train()
    training_time = time.perf_counter() - started
    metrics, history = _evaluate_and_archive(
        trainer,
        output_dir,
        training_time,
        keep_pth=keep_pth,
    )
    row = _summary_row(
        metrics,
        history,
        run_type="continuous_control",
        generation=None,
        epochs_requested=epochs,
    )
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, history, row


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ResNet50 + fixed KELS masks + generation-boundary reset, "
            "isolated under temp/."
        )
    )
    parser.add_argument("--common-config", type=Path, default=DEFAULT_COMMON_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--split-rate", type=float, default=0.8)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument(
        "--epochs-per-generation",
        type=int,
        default=None,
        help="Override train.epochs from the common config.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--run-continuous-control",
        action="store_true",
        help="Also train a normal ResNet50 for generations*epochs without resets.",
    )
    parser.add_argument("--dry-run", action="store_true")
    pth_group = parser.add_mutually_exclusive_group()
    pth_group.add_argument("--keep-pth", dest="keep_pth", action="store_true")
    pth_group.add_argument("--discard-pth", dest="keep_pth", action="store_false")
    parser.set_defaults(keep_pth=None)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.generations < 1:
        raise ValueError("--generations must be at least 1.")

    common_config_path = args.common_config.resolve()
    model_config_path = args.model_config.resolve()
    common_config = _load_yaml_mapping(common_config_path, "Common config")
    _validate_common_config(common_config)
    model_config = train_batch.load_model_config(model_config_path)

    epochs_per_generation = int(
        args.epochs_per_generation
        if args.epochs_per_generation is not None
        else common_config["train"]["epochs"]
    )
    if epochs_per_generation < 1:
        raise ValueError("epochs_per_generation must be at least 1.")
    keep_pth = (
        bool(common_config["train"].get("keep_pth_files", False))
        if args.keep_pth is None
        else bool(args.keep_pth)
    )
    device = train_batch.resolve_device(args.device)
    dataset_root = _resolve_project_path(common_config["dataset_root"])
    runs_root = _resolve_project_path(common_config["runs_root"])
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    preview_config = _build_runtime_config(
        common_config,
        model_config,
        model_config_path,
        dataset_root,
        EXPERIMENT_DIR / "__dry_run__",
        device,
        epochs=epochs_per_generation,
        run_name="ke_preview",
    )
    dataset_summary = train_batch.validate_fixed_dataset(preview_config, device)
    mask_summary = _offline_topology_probe(
        common_config,
        model_config,
        model_config_path,
        dataset_root,
        args.split_rate,
    )

    train_batch.print_dataset_summary(dataset_summary)
    print(f"Common config: {common_config_path}")
    print(f"Model config: {model_config_path}")
    print(f"Device: {device}")
    print(
        "KE: "
        f"KELS sr={args.split_rate:g}, generations={args.generations}, "
        f"epochs/generation={epochs_per_generation}"
    )
    print(
        "KELS topology: "
        f"layers={mask_summary['target_layer_count']}, "
        f"fit={mask_summary['fit_fraction']:.4%}, "
        f"reset={mask_summary['reset_fraction']:.4%}"
    )
    print(f"Continuous {args.generations * epochs_per_generation}-epoch control: {args.run_continuous_control}")
    print(f"Keep PTH: {keep_pth}")
    if args.dry_run:
        print("\ndry-run passed; no run directory or pretrained download was created.")
        return

    runs_root.mkdir(parents=True, exist_ok=True)
    run_label = (
        f"ke_resnet50_sr{_split_rate_tag(args.split_rate)}_"
        f"g{args.generations}_e{epochs_per_generation}"
    )
    run_dir = train_batch.create_unique_run_directory(runs_root, run_label)
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "Knowledge Evolution / KELS",
        "split_rate": float(args.split_rate),
        "generations": int(args.generations),
        "epochs_per_generation": epochs_per_generation,
        "continuous_control": bool(args.run_continuous_control),
        "generation_transition_source": "previous_generation_final_epoch",
        "generation_evaluation_source": "generation_best_validation_checkpoint",
        "reset_seed_formula": "random_seed + 10000 + generation_number",
        "common_config": str(common_config_path),
        "model_config": str(model_config_path),
        "dataset_root": str(dataset_root),
        "device": str(device),
        "keep_pth_files": keep_pth,
    }
    train_batch.save_json(run_dir / "run_manifest.json", run_manifest)
    train_batch.save_json(run_dir / "dataset_summary.json", dataset_summary)

    rows: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    previous_final_state: dict[str, torch.Tensor] | None = None
    fixed_masks: dict[str, torch.Tensor] | None = None
    base_seed = int(common_config.get("random_seed", 2026))

    for generation in range(1, args.generations + 1):
        output_dir = run_dir / f"generation_{generation:02d}"
        output_dir.mkdir(parents=True, exist_ok=False)
        run_name = (
            f"ke_resnet50_sr{_split_rate_tag(args.split_rate)}_"
            f"generation_{generation:02d}"
        )
        config = _build_runtime_config(
            common_config,
            model_config,
            model_config_path,
            dataset_root,
            output_dir,
            device,
            epochs=epochs_per_generation,
            run_name=run_name,
        )
        print("\n" + "=" * 80)
        print(f"Starting Knowledge Evolution generation {generation}/{args.generations}")
        print(f"Output: {output_dir}")
        print("=" * 80)
        trainer = train_batch.Trainer(config=config, device=device)

        if fixed_masks is None:
            fixed_masks = create_kels_masks(trainer.model, args.split_rate)
            mask_summary = summarize_kels_masks(
                trainer.model,
                fixed_masks,
                args.split_rate,
            )
            train_batch.save_json(run_dir / "fixed_kels_mask_summary.json", mask_summary)
        else:
            validate_kels_masks(trainer.model, fixed_masks)

        reset_report = None
        if generation > 1:
            if previous_final_state is None:
                raise RuntimeError("The previous generation did not provide a final state.")
            trainer.model.load_state_dict(previous_final_state, strict=True)
            batch_norm_before = _batch_norm_state(trainer.model)
            reset_report = reset_reset_hypothesis(
                trainer.model,
                fixed_masks,
                seed=base_seed + 10000 + generation,
            )
            reset_report["batch_norm_tensors_verified"] = _verify_batch_norm_unchanged(
                batch_norm_before,
                trainer.model,
            )
            train_batch.save_json(output_dir / "generation_reset_report.json", reset_report)
            trainer.logger.info(
                "knowledge_evolution_reset_applied",
                generation=generation,
                split_rate=float(args.split_rate),
                **reset_report,
            )

        started = time.perf_counter()
        trainer.train()
        training_time = time.perf_counter() - started
        if generation < args.generations:
            # Capture BEFORE evaluate_best_checkpoint replaces the model by the
            # best-validation state. KE evolves the final trained generation.
            previous_final_state = _clone_state_dict(trainer.model)
        else:
            previous_final_state = None

        metrics, history = _evaluate_and_archive(
            trainer,
            output_dir,
            training_time,
            keep_pth=keep_pth,
        )
        metrics.update(
            {
                "generation": generation,
                "split_rate": float(args.split_rate),
                "epochs_per_generation": epochs_per_generation,
                "kels_fit_fraction": mask_summary["fit_fraction"],
                "kels_reset_fraction": mask_summary["reset_fraction"],
                "generation_reset": reset_report,
            }
        )
        train_batch.save_json(output_dir / "test_metrics.json", metrics)
        row = _summary_row(
            metrics,
            history,
            run_type="knowledge_evolution",
            generation=generation,
            epochs_requested=epochs_per_generation,
        )
        rows.append(row)
        generation_records.append(
            {
                "generation": generation,
                "output_dir": str(output_dir),
                "summary": row,
                "reset": reset_report,
            }
        )
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    control_record = None
    if args.run_continuous_control:
        control_epochs = args.generations * epochs_per_generation
        control_metrics, control_history, control_row = _run_continuous_control(
            common_config,
            model_config,
            model_config_path,
            dataset_root,
            run_dir,
            device,
            epochs=control_epochs,
            keep_pth=keep_pth,
        )
        rows.append(control_row)
        control_record = {
            "epochs": control_epochs,
            "summary": control_row,
            "metrics": control_metrics,
            "epochs_completed": len(control_history.get("train_loss") or []),
        }

    _write_csv(run_dir / "generation_summary.csv", rows)
    _write_markdown_summary(
        run_dir,
        rows,
        split_rate=args.split_rate,
        generations=args.generations,
        epochs_per_generation=epochs_per_generation,
    )
    _plot_generation_trends(run_dir, rows)
    train_batch.save_json(
        run_dir / "ke_summary.json",
        {
            "manifest": run_manifest,
            "fixed_mask": mask_summary,
            "generations": generation_records,
            "continuous_control": control_record,
        },
    )
    print(f"\nKnowledge Evolution run complete: {run_dir}")
    print(f"Summary: {run_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()

