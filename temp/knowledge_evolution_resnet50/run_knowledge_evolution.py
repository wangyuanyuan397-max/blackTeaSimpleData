"""Run the conclusive five-run ResNet50 Knowledge Evolution V2 matrix.

Everything remains isolated under ``temp/knowledge_evolution_resnet50`` while
the project's data, model, trainer and evaluation components are reused.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
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
DEFAULT_EXPERIMENT_CONFIGS = (
    EXPERIMENT_DIR / "experiments" / "ctrl_10.yaml",
    EXPERIMENT_DIR / "experiments" / "ke_v2_01_sr08_e10.yaml",
    EXPERIMENT_DIR / "experiments" / "ke_v2_02_sr09_e10.yaml",
    EXPERIMENT_DIR / "experiments" / "ctrl_15.yaml",
    EXPERIMENT_DIR / "experiments" / "ke_v2_03_sr09_e15.yaml",
)

for import_path in (PROJECT_ROOT, EXPERIMENT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from ke_core import (  # noqa: E402
    create_kels_masks,
    is_better_validation_checkpoint,
    reset_reset_hypothesis_from_fresh_model,
    summarize_kels_masks,
    validate_kels_masks,
)
from src.engine.hooks import HookManager, LRSchedulerHook  # noqa: E402


def _load_train_batch_module():
    spec = importlib.util.spec_from_file_location("ke_v2_train_batch_base", TRAIN_BATCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load project trainer: {TRAIN_BATCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ke_v2_train_batch_base"] = module
    spec.loader.exec_module(module)
    return module


train_batch = _load_train_batch_module()


@dataclass(frozen=True)
class ExperimentSpec:
    config_path: Path
    experiment_id: str
    experiment_name: str
    generations: int
    epochs_per_generation: int
    kels_reset_enabled: bool
    split_rate: float | None
    transition_source: str
    bn_recalibration_enabled: bool
    bn_recalibration_max_batches: int | None
    keep_final_checkpoint: bool
    raw_config: dict[str, Any]


def _load_yaml_mapping(path: Path, kind: str) -> dict[str, Any]:
    path = path.resolve()
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


def _load_experiment_spec(path: Path) -> ExperimentSpec:
    raw = _load_yaml_mapping(path, "KE-V2 experiment config")
    experiment_id = str(raw.get("id") or "").strip()
    experiment_name = str(raw.get("experiment_name") or "").strip()
    ke = raw.get("knowledge_evolution")
    if not experiment_id or not experiment_name or not isinstance(ke, dict):
        raise ValueError(f"Invalid KE-V2 config identity/section: {path}")
    if not bool(ke.get("enabled", False)):
        raise ValueError(f"knowledge_evolution.enabled must be true: {path}")

    generations = int(ke.get("generations", 0))
    epochs = int(ke.get("epochs_per_generation", 0))
    if generations < 1 or epochs < 1:
        raise ValueError(f"generations and epochs_per_generation must be positive: {path}")
    transition_source = str(ke.get("transition_source") or "")
    if transition_source != "previous_generation_best_validation":
        raise ValueError(
            "KE-V2 requires transition_source=previous_generation_best_validation: "
            f"{path}"
        )
    if bool((ke.get("early_stopping") or {}).get("enabled", True)):
        raise ValueError(f"KE-V2 requires early stopping disabled: {path}")
    if not bool((ke.get("optimizer") or {}).get("restart_each_generation", False)):
        raise ValueError(f"Optimizer restart must be enabled: {path}")
    if not bool((ke.get("scheduler") or {}).get("restart_each_generation", False)):
        raise ValueError(f"Scheduler restart must be enabled: {path}")
    if str((ke.get("batch_norm") or {}).get("policy")) != "fully_inherited":
        raise ValueError(f"BatchNorm policy must be fully_inherited: {path}")
    if str((ke.get("bias") or {}).get("policy")) != "fully_inherited":
        raise ValueError(f"Bias policy must be fully_inherited: {path}")

    kels_reset_enabled = bool((ke.get("kels_reset") or {}).get("enabled", False))
    split_rate_raw = ke.get("split_rate")
    split_rate = float(split_rate_raw) if split_rate_raw is not None else None
    if kels_reset_enabled:
        if split_rate is None or not 0.0 < split_rate <= 1.0:
            raise ValueError(f"Enabled KELS reset requires split_rate in (0, 1]: {path}")
        if str((ke.get("kels_reset") or {}).get("source")) != "fresh_random_model":
            raise ValueError(f"KE-V2 reset source must be fresh_random_model: {path}")
        if not bool(ke.get("fixed_mask", False)):
            raise ValueError(f"Enabled KELS reset requires fixed_mask=true: {path}")

    recalibration = ke.get("bn_recalibration") or {}
    max_batches_raw = recalibration.get("max_batches")
    max_batches = int(max_batches_raw) if max_batches_raw is not None else None
    if max_batches is not None and max_batches < 1:
        raise ValueError(f"bn_recalibration.max_batches must be positive or null: {path}")

    checkpoints = ke.get("checkpoints") or {}
    if not bool(checkpoints.get("keep_best", False)):
        raise ValueError(f"KE-V2 requires checkpoints.keep_best=true: {path}")

    return ExperimentSpec(
        config_path=path.resolve(),
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        generations=generations,
        epochs_per_generation=epochs,
        kels_reset_enabled=kels_reset_enabled,
        split_rate=split_rate,
        transition_source=transition_source,
        bn_recalibration_enabled=bool(recalibration.get("enabled", False)),
        bn_recalibration_max_batches=max_batches,
        keep_final_checkpoint=bool(checkpoints.get("keep_final", True)),
        raw_config=raw,
    )


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _safe_id(value: str) -> str:
    return train_batch.safe_run_name(value.lower().replace("-", "_"))


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
    common["train"]["patience"] = 0
    common["train"]["keep_pth_files"] = True
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


class LexicographicBestCheckpointHook:
    """Save best_val.pth using acc > QWK > lower loss."""

    def __init__(
        self,
        output_dir: Path,
        *,
        generation: int,
        split_rate: float | None,
        transition_source: str,
    ) -> None:
        self.path = Path(output_dir) / "best_val.pth"
        self.generation = int(generation)
        self.split_rate = split_rate
        self.transition_source = transition_source
        self.best_metrics: dict[str, Any] | None = None

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        if not bool(metrics.get("checkpoint_eligible", True)):
            return
        candidate = dict(metrics)
        full_validation = getattr(trainer, "last_full_validation_metrics", {}) or {}
        candidate["val_macro_f1"] = full_validation.get("macro_f1")
        if not is_better_validation_checkpoint(candidate, self.best_metrics):
            return

        self.best_metrics = candidate
        state_dict = trainer.model.state_dict()
        torch.save(
            {
                "generation": self.generation,
                "epoch": int(epoch),
                "epoch_number": int(epoch) + 1,
                "model": state_dict,
                "model_state_dict": state_dict,
                "metrics": candidate,
                "val_accuracy": candidate.get("val_acc"),
                "val_macro_f1": candidate.get("val_macro_f1"),
                "val_mae": candidate.get("val_mae"),
                "val_qwk": candidate.get("val_qwk"),
                "val_loss": candidate.get("val_loss"),
                "train_accuracy": candidate.get("train_acc"),
                "split_rate": self.split_rate,
                "transition_source": self.transition_source,
                "checkpoint_tie_break": [
                    "val_accuracy:max",
                    "val_qwk:max",
                    "val_loss:min",
                ],
            },
            self.path,
        )


class KEV2Trainer(train_batch.Trainer):
    """Project Trainer with V2 checkpoint selection and no early stopping."""

    def __init__(
        self,
        *args,
        generation: int,
        split_rate: float | None,
        transition_source: str,
        **kwargs,
    ) -> None:
        self.ke_generation = int(generation)
        self.ke_split_rate = split_rate
        self.ke_transition_source = transition_source
        self.last_full_validation_metrics: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def _setup_hooks(self) -> HookManager:
        manager = HookManager(self.logger)
        manager.register(
            LexicographicBestCheckpointHook(
                self.output_dir,
                generation=self.ke_generation,
                split_rate=self.ke_split_rate,
                transition_source=self.ke_transition_source,
            )
        )
        if self.scheduler:
            manager.register(LRSchedulerHook(self.scheduler))
        return manager

    def _validate_epoch(self, epoch: int) -> dict[str, float]:
        metrics = super()._validate_epoch(epoch)
        self.last_full_validation_metrics = dict(metrics)
        return metrics

    def restart_optimizer_and_scheduler(self, total_epochs: int) -> None:
        """Discard every optimizer/scheduler state before a generation starts."""
        self.optimizer = self.builder.build_optimizer(self.model)
        self.scheduler = self.builder.build_scheduler(
            self.optimizer,
            total_epochs=int(total_epochs),
        )
        for hook in self.hook_manager.hooks:
            if isinstance(hook, LRSchedulerHook):
                hook.scheduler = self.scheduler
                hook.current_epoch = 0
        self.logger.info(
            "ke_v2_optimizer_scheduler_restarted",
            generation=self.ke_generation,
            total_epochs=int(total_epochs),
        )


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


def _load_best_checkpoint_into_model(
    trainer: KEV2Trainer,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Previous best checkpoint is missing: {checkpoint_path}")
    checkpoint = train_batch.load_checkpoint(checkpoint_path, trainer.device)
    state_dict = checkpoint.get("model") or checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"No model state was found in {checkpoint_path}")
    trainer.model.load_state_dict(state_dict, strict=True)
    return checkpoint


def _build_fresh_random_model(trainer: KEV2Trainer, reset_seed: int) -> nn.Module:
    fresh_config = (
        trainer.config.model_copy(deep=True)
        if hasattr(trainer.config, "model_copy")
        else copy.deepcopy(trainer.config)
    )
    fresh_config.model.backbone.pretrained = False
    cuda_devices = list(range(torch.cuda.device_count())) if trainer.device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(reset_seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(reset_seed))
        builder = train_batch.ComponentBuilder(fresh_config, trainer.device, logger=None)
        builder._num_classes = trainer.builder.num_classes
        fresh_model, _ = builder.build_model()
    return fresh_model


def _apply_bn_recalibration(
    trainer: KEV2Trainer,
    *,
    enabled: bool,
    max_batches: int | None,
) -> dict[str, Any]:
    report = {
        "enabled": bool(enabled),
        "applied": False,
        "max_batches": max_batches,
        "batches_used": 0,
    }
    if not enabled:
        return report

    batch_norm_modules = [
        module
        for module in trainer.model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not batch_norm_modules:
        report["reason"] = "no_batch_norm_modules"
        return report

    was_training = trainer.model.training
    original_momenta = [module.momentum for module in batch_norm_modules]
    trainer.model.eval()
    for module in batch_norm_modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(trainer.train_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = train_batch.move_to_device(batch[0], trainer.device)
                trainer.model(images)
                report["batches_used"] += 1
    finally:
        for module, momentum in zip(batch_norm_modules, original_momenta):
            module.momentum = momentum
        trainer.model.train(was_training)

    report["applied"] = True
    report["batch_norm_layer_count"] = len(batch_norm_modules)
    return report


def _start_validation(trainer: KEV2Trainer, generation: int) -> dict[str, Any]:
    metrics = trainer.evaluator.evaluate(
        trainer.val_loader,
        trainer.loss_fn,
        desc=f"Generation {generation} Start Validation",
    )
    return train_batch.to_builtin(metrics)


def _save_final_checkpoint(
    trainer: KEV2Trainer,
    output_dir: Path,
    generation: int,
) -> Path:
    path = output_dir / "final.pth"
    epochs_completed = len(trainer.history.get("train_loss") or [])
    state_dict = trainer.model.state_dict()
    torch.save(
        {
            "generation": int(generation),
            "epoch": max(epochs_completed - 1, 0),
            "epoch_number": epochs_completed,
            "model": state_dict,
            "model_state_dict": state_dict,
            "final_metrics": {
                key: values[-1]
                for key, values in trainer.history.items()
                if values
            },
            "note": "Not used for generation transition; retained only for audit.",
        },
        path,
    )
    return path


@contextmanager
def _temporary_best_model_alias(output_dir: Path):
    """Expose best_val.pth under the legacy evaluator's expected filename."""
    best_path = output_dir / "best_val.pth"
    alias_path = output_dir / "best_model.pth"
    if not best_path.is_file():
        raise FileNotFoundError(f"Best validation checkpoint is missing: {best_path}")
    if alias_path.exists():
        raise FileExistsError(f"Unexpected legacy checkpoint already exists: {alias_path}")
    try:
        try:
            os.link(best_path, alias_path)
        except OSError:
            shutil.copy2(best_path, alias_path)
        yield
    finally:
        if alias_path.exists():
            alias_path.unlink()


def _evaluate_and_archive(
    trainer: KEV2Trainer,
    output_dir: Path,
    training_time_seconds: float,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    history = train_batch.to_builtin(trainer.history)
    with _temporary_best_model_alias(output_dir):
        metrics = train_batch.evaluate_best_checkpoint(trainer, training_time_seconds)
    train_batch.save_json(output_dir / "history.json", history)
    train_batch.save_json(output_dir / "test_metrics.json", metrics)
    train_batch.save_confusion_matrix_csvs(output_dir, metrics)
    return metrics, history


def _recovery_epoch(
    source_best_val: float | None,
    start_val: float | None,
    history: dict[str, list[Any]],
) -> int | None:
    if source_best_val is None:
        return None
    if start_val is not None and float(start_val) > float(source_best_val):
        return 0
    for epoch, value in enumerate(history.get("val_acc") or [], start=1):
        if float(value) > float(source_best_val):
            return epoch
    return None


def _last(history: dict[str, list[Any]], key: str) -> Any:
    values = history.get(key) or []
    return values[-1] if values else None


def _generation_summary_row(
    spec: ExperimentSpec,
    generation: int,
    source: dict[str, Any] | None,
    start_metrics: dict[str, Any],
    metrics: dict[str, Any],
    history: dict[str, list[Any]],
    recovery_epoch: int | None,
    reset_fraction: float,
) -> dict[str, Any]:
    source_val = source.get("source_best_val_accuracy") if source else None
    start_val = start_metrics.get("accuracy")
    return {
        "experiment_id": spec.experiment_id,
        "run_type": "knowledge_evolution" if spec.kels_reset_enabled else "control",
        "kels_reset_enabled": spec.kels_reset_enabled,
        "split_rate": spec.split_rate,
        "epochs_per_generation": spec.epochs_per_generation,
        "generation": generation,
        "source_generation": source.get("source_generation") if source else None,
        "source_best_epoch": source.get("source_best_epoch") if source else None,
        "source_best_val_accuracy": source_val,
        "source_checkpoint_path": source.get("source_checkpoint_path") if source else None,
        "start_val_accuracy": start_val,
        "start_val_qwk": start_metrics.get("qwk"),
        "start_val_loss": start_metrics.get("loss"),
        "reset_shock": (
            float(source_val) - float(start_val)
            if source_val is not None and start_val is not None
            else None
        ),
        "best_epoch": metrics.get("best_epoch"),
        "best_train_accuracy": metrics.get("best_train_accuracy"),
        "best_validation_accuracy": metrics.get("best_validation_accuracy"),
        "test_accuracy": metrics.get("accuracy"),
        "test_macro_f1": metrics.get("macro_f1"),
        "test_mae": metrics.get("mae"),
        "test_qwk": metrics.get("qwk"),
        "train_val_gap": metrics.get("train_val_accuracy_gap"),
        "final_train_accuracy": _last(history, "train_acc"),
        "final_validation_accuracy": _last(history, "val_acc"),
        "recovery_epoch": recovery_epoch,
        "reset_fraction": float(reset_fraction),
        "epochs_completed": len(history.get("train_loss") or []),
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


def _format_markdown(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_experiment_summary(
    run_dir: Path,
    spec: ExperimentSpec,
    rows: list[dict[str, Any]],
) -> None:
    columns = (
        "generation",
        "source_best_epoch",
        "start_val_accuracy",
        "best_epoch",
        "best_train_accuracy",
        "best_validation_accuracy",
        "test_accuracy",
        "test_macro_f1",
        "test_mae",
        "test_qwk",
        "train_val_gap",
        "final_train_accuracy",
        "final_validation_accuracy",
        "recovery_epoch",
        "reset_fraction",
    )
    lines = [
        f"# {spec.experiment_id} summary",
        "",
        f"- KELS reset: `{spec.kels_reset_enabled}`",
        f"- split rate: `{spec.split_rate}`",
        f"- generations: `{spec.generations}`",
        f"- epochs/generation: `{spec.epochs_per_generation}`",
        "- transition: `previous_generation_best_validation`",
        "- checkpoint tie-break: `val_accuracy > val_qwk > lower val_loss`",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_markdown(row.get(column)) for column in columns)
            + " |"
        )
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_generation_trends(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        generations = [int(row["generation"]) for row in rows]
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for key, label in (
            ("best_validation_accuracy", "Best val accuracy"),
            ("test_accuracy", "Test accuracy"),
            ("test_macro_f1", "Test macro-F1"),
        ):
            values = [row.get(key) for row in rows]
            if all(value is not None for value in values):
                axes[0].plot(generations, values, marker="o", label=label)
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Score")
        axes[0].set_xticks(generations)
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        gaps = [row.get("train_val_gap") for row in rows]
        if all(value is not None for value in gaps):
            axes[1].plot(generations, gaps, marker="o", color="#c23b22")
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Train - validation accuracy")
        axes[1].set_xticks(generations)
        axes[1].grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(run_dir / "generation_trends.png", dpi=180)
        plt.close(figure)
    except Exception as error:
        (run_dir / "plot_warning.txt").write_text(str(error), encoding="utf-8")


def _plot_reset_recovery(run_dir: Path, records: list[dict[str, Any]]) -> None:
    recovery_records = [record for record in records if record["generation"] > 1]
    if not recovery_records:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(
            1,
            len(recovery_records),
            figsize=(5.4 * len(recovery_records), 4.4),
            squeeze=False,
        )
        for axis, record in zip(axes[0], recovery_records):
            history = record["history"]
            values = [record["start_metrics"].get("accuracy")]
            values.extend(history.get("val_acc") or [])
            x_values = list(range(len(values)))
            axis.plot(x_values, values, marker="o", label="Validation accuracy")
            source_value = record["source"]["source_best_val_accuracy"]
            axis.axhline(
                source_value,
                color="#555555",
                linestyle="--",
                label="Source best validation",
            )
            recovery = record["summary"]["recovery_epoch"]
            if recovery is not None:
                axis.scatter(
                    [recovery],
                    [values[int(recovery)]],
                    color="#16823b",
                    s=55,
                    zorder=3,
                    label=f"Recovery epoch {recovery}",
                )
            axis.set_title(f"Generation {record['generation']}")
            axis.set_xlabel("0 = start; then training epoch")
            axis.set_ylabel("Validation accuracy")
            axis.set_xticks(x_values)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(run_dir / "reset_shock_recovery.png", dpi=180)
        plt.close(figure)
    except Exception as error:
        (run_dir / "recovery_plot_warning.txt").write_text(str(error), encoding="utf-8")


def _offline_topology_probe(
    common_config: dict[str, Any],
    model_config: dict[str, Any],
    model_config_path: Path,
    dataset_root: Path,
    split_rate: float,
) -> dict[str, Any]:
    probe_model_config = copy.deepcopy(model_config)
    probe_model_config["model"]["backbone"]["pretrained"] = False
    preview_config = _build_runtime_config(
        common_config,
        probe_model_config,
        model_config_path,
        dataset_root,
        EXPERIMENT_DIR / "__dry_run__",
        torch.device("cpu"),
        epochs=1,
        run_name="ke_v2_offline_topology_probe",
    )
    builder = train_batch.ComponentBuilder(preview_config, torch.device("cpu"), logger=None)
    builder.build_dataloaders()
    model, _ = builder.build_model()
    masks = create_kels_masks(model, split_rate=split_rate)
    return summarize_kels_masks(model, masks, split_rate)


def _run_experiment(
    spec: ExperimentSpec,
    common_config: dict[str, Any],
    model_config: dict[str, Any],
    model_config_path: Path,
    dataset_root: Path,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    train_batch.save_json(run_dir / "experiment_config.json", spec.raw_config)
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": spec.experiment_id,
        "experiment_name": spec.experiment_name,
        "kels_reset_enabled": spec.kels_reset_enabled,
        "split_rate": spec.split_rate,
        "generations": spec.generations,
        "epochs_per_generation": spec.epochs_per_generation,
        "generation_transition_source": spec.transition_source,
        "checkpoint_tie_break": ["val_accuracy:max", "val_qwk:max", "val_loss:min"],
        "reset_source": "fresh_random_model" if spec.kels_reset_enabled else None,
        "reset_seed_formula": "random_seed + 10000 + generation_number",
        "batch_norm_policy": "fully_inherited",
        "bias_policy": "fully_inherited",
        "optimizer_restart_each_generation": True,
        "scheduler_restart_each_generation": True,
        "early_stopping_enabled": False,
        "bn_recalibration_enabled": spec.bn_recalibration_enabled,
        "best_checkpoint_retained": True,
        "final_checkpoint_retained": spec.keep_final_checkpoint,
        "dataset_root": str(dataset_root),
        "device": str(device),
    }
    train_batch.save_json(run_dir / "run_manifest.json", run_manifest)

    fixed_masks: dict[str, torch.Tensor] | None = None
    mask_summary: dict[str, Any] | None = None
    previous_best_path: Path | None = None
    base_seed = int(common_config.get("random_seed", 2026))
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for generation in range(1, spec.generations + 1):
        output_dir = run_dir / f"generation_{generation:02d}"
        output_dir.mkdir(parents=True, exist_ok=False)
        run_name = f"{_safe_id(spec.experiment_id)}_generation_{generation:02d}"
        config = _build_runtime_config(
            common_config,
            model_config,
            model_config_path,
            dataset_root,
            output_dir,
            device,
            epochs=spec.epochs_per_generation,
            run_name=run_name,
        )
        print("\n" + "=" * 80)
        print(
            f"{spec.experiment_id}: generation {generation}/{spec.generations}; "
            f"KELS reset={spec.kels_reset_enabled}; sr={spec.split_rate}"
        )
        print(f"Output: {output_dir}")
        print("=" * 80)
        trainer = KEV2Trainer(
            config=config,
            device=device,
            generation=generation,
            split_rate=spec.split_rate,
            transition_source=spec.transition_source,
        )

        if spec.kels_reset_enabled:
            if fixed_masks is None:
                fixed_masks = create_kels_masks(trainer.model, spec.split_rate)
                mask_summary = summarize_kels_masks(
                    trainer.model,
                    fixed_masks,
                    spec.split_rate,
                )
                train_batch.save_json(
                    run_dir / "fixed_kels_mask_summary.json",
                    mask_summary,
                )
            else:
                validate_kels_masks(trainer.model, fixed_masks)

        source: dict[str, Any] | None = None
        reset_report: dict[str, Any] | None = None
        if generation > 1:
            if previous_best_path is None:
                raise RuntimeError("Previous generation best checkpoint was not recorded.")
            source_checkpoint = _load_best_checkpoint_into_model(
                trainer,
                previous_best_path,
            )
            source_metrics = source_checkpoint.get("metrics") or {}
            source = {
                "source_generation": int(source_checkpoint.get("generation", generation - 1)),
                "source_best_epoch": int(
                    source_checkpoint.get(
                        "epoch_number",
                        int(source_checkpoint.get("epoch", -1)) + 1,
                    )
                ),
                "source_best_val_accuracy": float(
                    source_checkpoint.get(
                        "val_accuracy",
                        source_metrics.get("val_acc"),
                    )
                ),
                "source_checkpoint_path": str(previous_best_path.resolve()),
                "transition_source": "best_validation_checkpoint",
            }
            print(
                "Transition source: "
                f"generation={source['source_generation']}, "
                f"best_epoch={source['source_best_epoch']}, "
                f"best_val={source['source_best_val_accuracy']:.6f}, "
                f"path={source['source_checkpoint_path']}"
            )

            if spec.kels_reset_enabled:
                if fixed_masks is None or mask_summary is None:
                    raise RuntimeError("Fixed KELS masks were not initialized.")
                reset_seed = base_seed + 10000 + generation
                batch_norm_before = _batch_norm_state(trainer.model)
                fresh_model = _build_fresh_random_model(trainer, reset_seed)
                reset_report = reset_reset_hypothesis_from_fresh_model(
                    trainer.model,
                    fresh_model,
                    fixed_masks,
                )
                del fresh_model
                reset_report.update(
                    {
                        "reset_seed": reset_seed,
                        "source_generation": source["source_generation"],
                        "source_best_epoch": source["source_best_epoch"],
                        "source_best_val_accuracy": source[
                            "source_best_val_accuracy"
                        ],
                        "source_checkpoint_path": source["source_checkpoint_path"],
                        "transition_source": "best_validation_checkpoint",
                        "reset_fraction": mask_summary["reset_fraction"],
                        "batch_norm_tensors_verified": _verify_batch_norm_unchanged(
                            batch_norm_before,
                            trainer.model,
                        ),
                    }
                )
            else:
                reset_report = {
                    **source,
                    "applied": False,
                    "reset_parameters_requested": 0,
                    "reset_parameters_changed": 0,
                    "reset_changed_fraction": 0.0,
                    "reset_fraction": 0.0,
                    "batch_norm_policy": "fully_inherited",
                    "classifier_bias_policy": "fully_inherited",
                }
            train_batch.save_json(
                output_dir / "generation_transition_report.json",
                reset_report,
            )

        recalibration_report = _apply_bn_recalibration(
            trainer,
            enabled=spec.bn_recalibration_enabled and generation > 1,
            max_batches=spec.bn_recalibration_max_batches,
        )
        train_batch.save_json(
            output_dir / "bn_recalibration_report.json",
            recalibration_report,
        )
        start_metrics = _start_validation(trainer, generation)
        train_batch.save_json(
            output_dir / "start_validation_metrics.json",
            start_metrics,
        )

        # Explicitly discard the optimizer/scheduler constructed by Trainer.__init__.
        # This occurs after loading/resetting the generation start model.
        trainer.restart_optimizer_and_scheduler(spec.epochs_per_generation)
        started = time.perf_counter()
        trainer.train()
        training_time = time.perf_counter() - started

        final_path = _save_final_checkpoint(trainer, output_dir, generation)
        best_path = output_dir / "best_val.pth"
        if not best_path.is_file():
            raise FileNotFoundError(f"Training did not create {best_path}")
        previous_best_path = best_path

        metrics, history = _evaluate_and_archive(
            trainer,
            output_dir,
            training_time,
        )
        source_val = source.get("source_best_val_accuracy") if source else None
        recovery = _recovery_epoch(
            source_val,
            start_metrics.get("accuracy"),
            history,
        )
        reset_fraction = (
            float(mask_summary["reset_fraction"])
            if spec.kels_reset_enabled and mask_summary is not None and generation > 1
            else 0.0
        )
        row = _generation_summary_row(
            spec,
            generation,
            source,
            start_metrics,
            metrics,
            history,
            recovery,
            reset_fraction,
        )
        rows.append(row)
        record = {
            "generation": generation,
            "source": source,
            "start_metrics": start_metrics,
            "reset": reset_report,
            "bn_recalibration": recalibration_report,
            "summary": row,
            "history": history,
            "best_checkpoint_path": str(best_path.resolve()),
            "final_checkpoint_path": (
                str(final_path.resolve()) if spec.keep_final_checkpoint else None
            ),
        }
        records.append(record)
        train_batch.save_json(output_dir / "generation_summary.json", record)
        if not spec.keep_final_checkpoint:
            final_path.unlink()

        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_csv(run_dir / "generation_summary.csv", rows)
    _write_experiment_summary(run_dir, spec, rows)
    _plot_generation_trends(run_dir, rows)
    _plot_reset_recovery(run_dir, records)
    result = {
        "experiment_id": spec.experiment_id,
        "experiment_name": spec.experiment_name,
        "status": "success",
        "run_dir": str(run_dir),
        "manifest": run_manifest,
        "fixed_mask": mask_summary,
        "generation_rows": rows,
        "generations": records,
    }
    train_batch.save_json(run_dir / "experiment_summary.json", result)
    return result


def _build_comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = {
        result["experiment_id"]: result
        for result in results
        if result.get("status") == "success"
    }
    comparisons = []
    for ke_id, control_id in (
        ("KE-V2-01", "CTRL-10"),
        ("KE-V2-02", "CTRL-10"),
        ("KE-V2-03", "CTRL-15"),
    ):
        if ke_id not in successful or control_id not in successful:
            continue
        ke_row = successful[ke_id]["generation_rows"][-1]
        control_row = successful[control_id]["generation_rows"][-1]
        accuracy_delta = float(ke_row["test_accuracy"]) - float(
            control_row["test_accuracy"]
        )
        f1_delta = float(ke_row["test_macro_f1"]) - float(
            control_row["test_macro_f1"]
        )
        mae_delta = float(ke_row["test_mae"]) - float(control_row["test_mae"])
        qwk_delta = float(ke_row["test_qwk"]) - float(control_row["test_qwk"])
        meets_accuracy = accuracy_delta >= 0.01
        meets_secondary = f1_delta > 0.0 and mae_delta <= 0.0 and qwk_delta >= 0.0
        if meets_accuracy and meets_secondary:
            verdict = "worth_continuing"
        elif accuracy_delta > 0.0:
            verdict = "limited_or_mixed_effect"
        else:
            verdict = "does_not_beat_control"
        comparisons.append(
            {
                "ke_experiment": ke_id,
                "control_experiment": control_id,
                "generation_compared": ke_row["generation"],
                "ke_test_accuracy": ke_row["test_accuracy"],
                "control_test_accuracy": control_row["test_accuracy"],
                "test_accuracy_delta": accuracy_delta,
                "test_accuracy_delta_pp": accuracy_delta * 100.0,
                "test_macro_f1_delta": f1_delta,
                "test_mae_delta": mae_delta,
                "test_qwk_delta": qwk_delta,
                "meets_accuracy_plus_1pp": meets_accuracy,
                "macro_f1_up_mae_qwk_not_worse": meets_secondary,
                "verdict": verdict,
            }
        )
    return comparisons


def _write_matrix_summary(
    matrix_dir: Path,
    results: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    all_rows = [
        row
        for result in results
        if result.get("status") == "success"
        for row in result["generation_rows"]
    ]
    _write_csv(matrix_dir / "all_generation_summary.csv", all_rows)
    _write_csv(matrix_dir / "ke_vs_control_comparison.csv", comparisons)
    lines = [
        "# KE-V2 final validation matrix",
        "",
        "The paired difference isolates the contribution of KELS reset because each",
        "control uses the same Best-Val carry-over and optimizer/scheduler restart.",
        "",
        "| KE | Control | Acc delta (pp) | Macro-F1 delta | MAE delta | QWK delta | Verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in comparisons:
        lines.append(
            "| {ke_experiment} | {control_experiment} | {test_accuracy_delta_pp:.4f} "
            "| {test_macro_f1_delta:.6f} | {test_mae_delta:.6f} "
            "| {test_qwk_delta:.6f} | {verdict} |".format(**row)
        )
    failed = [result for result in results if result.get("status") != "success"]
    if failed:
        lines.extend(["", "## Failed experiments", ""])
        lines.extend(f"- {item['experiment_id']}: {item.get('error')}" for item in failed)
    (matrix_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_matrix_accuracy(matrix_dir: Path, results: list[dict[str, Any]]) -> None:
    successful = [result for result in results if result.get("status") == "success"]
    if not successful:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(9.5, 5.2))
        for result in successful:
            rows = result["generation_rows"]
            axis.plot(
                [row["generation"] for row in rows],
                [row["test_accuracy"] for row in rows],
                marker="o",
                label=result["experiment_id"],
            )
        axis.set_xlabel("Generation")
        axis.set_ylabel("Test accuracy")
        axis.set_xticks([1, 2, 3])
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(matrix_dir / "matrix_test_accuracy.png", dpi=180)
        plt.close(figure)
    except Exception as error:
        (matrix_dir / "matrix_plot_warning.txt").write_text(str(error), encoding="utf-8")


def _select_specs(requested: list[str] | None) -> list[ExperimentSpec]:
    specs = [_load_experiment_spec(path) for path in DEFAULT_EXPERIMENT_CONFIGS]
    if not requested:
        return specs
    by_key: dict[str, ExperimentSpec] = {}
    for spec in specs:
        by_key[spec.experiment_id.lower()] = spec
        by_key[spec.config_path.stem.lower()] = spec
    selected = []
    for value in requested:
        key = value.lower()
        if key not in by_key:
            raise ValueError(
                f"Unknown experiment {value!r}; choose from "
                + ", ".join(spec.experiment_id for spec in specs)
            )
        if by_key[key] in selected:
            raise ValueError(f"Experiment was requested twice: {value}")
        selected.append(by_key[key])
    return selected


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the five-run KE-V2 Best-Val carry-over validation matrix."
    )
    parser.add_argument("--common-config", type=Path, default=DEFAULT_COMMON_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Optional IDs/stems; default runs CTRL-10, KE-V2-01/02, CTRL-15, KE-V2-03.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    specs = _select_specs(args.experiments)
    if args.list_experiments:
        for spec in specs:
            print(
                f"{spec.experiment_id}: reset={spec.kels_reset_enabled}, "
                f"sr={spec.split_rate}, generations={spec.generations}, "
                f"epochs/generation={spec.epochs_per_generation}, "
                f"config={spec.config_path}"
            )
        return

    common_config_path = args.common_config.resolve()
    model_config_path = args.model_config.resolve()
    common_config = _load_yaml_mapping(common_config_path, "Common config")
    _validate_common_config(common_config)
    model_config = train_batch.load_model_config(model_config_path)
    device = train_batch.resolve_device(args.device)
    dataset_root = _resolve_project_path(common_config["dataset_root"])
    runs_root = _resolve_project_path(common_config["runs_root"])
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    preview_spec = specs[0]
    preview_config = _build_runtime_config(
        common_config,
        model_config,
        model_config_path,
        dataset_root,
        EXPERIMENT_DIR / "__dry_run__",
        device,
        epochs=preview_spec.epochs_per_generation,
        run_name="ke_v2_preview",
    )
    dataset_summary = train_batch.validate_fixed_dataset(preview_config, device)
    unique_split_rates = sorted(
        {spec.split_rate for spec in specs if spec.split_rate is not None}
    )
    topology_by_split_rate = {
        str(split_rate): _offline_topology_probe(
            common_config,
            model_config,
            model_config_path,
            dataset_root,
            split_rate,
        )
        for split_rate in unique_split_rates
    }

    train_batch.print_dataset_summary(dataset_summary)
    print(f"Common config: {common_config_path}")
    print(f"Model config: {model_config_path}")
    print(f"Device: {device}")
    print("KE-V2 matrix:")
    for spec in specs:
        print(
            f"  {spec.experiment_id}: reset={spec.kels_reset_enabled}, "
            f"sr={spec.split_rate}, g={spec.generations}, "
            f"epochs/g={spec.epochs_per_generation}, source=Best-Val"
        )
    for split_rate, summary in topology_by_split_rate.items():
        print(
            f"  KELS sr={split_rate}: layers={summary['target_layer_count']}, "
            f"fit={summary['fit_fraction']:.4%}, "
            f"reset={summary['reset_fraction']:.4%}"
        )
    if args.dry_run:
        print("\ndry-run passed; no run directory or pretrained download was created.")
        return

    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    matrix_dir = runs_root / f"ke_v2_matrix_{timestamp}"
    matrix_dir.mkdir(parents=True, exist_ok=False)
    train_batch.save_json(matrix_dir / "dataset_summary.json", dataset_summary)
    train_batch.save_json(
        matrix_dir / "dry_run_topology_summary.json",
        topology_by_split_rate,
    )

    results: list[dict[str, Any]] = []
    for spec in specs:
        experiment_dir = matrix_dir / _safe_id(spec.experiment_id)
        try:
            result = _run_experiment(
                spec,
                common_config,
                model_config,
                model_config_path,
                dataset_root,
                experiment_dir,
                device,
            )
        except Exception:
            error = traceback.format_exc()
            print(error)
            experiment_dir.mkdir(parents=True, exist_ok=True)
            (experiment_dir / "failure.txt").write_text(error, encoding="utf-8")
            result = {
                "experiment_id": spec.experiment_id,
                "experiment_name": spec.experiment_name,
                "status": "failed",
                "run_dir": str(experiment_dir),
                "error": error,
            }
        results.append(result)
        train_batch.save_json(matrix_dir / "matrix_progress.json", results)
        if result["status"] == "failed" and args.fail_fast:
            break

    comparisons = _build_comparisons(results)
    _write_matrix_summary(matrix_dir, results, comparisons)
    _plot_matrix_accuracy(matrix_dir, results)
    train_batch.save_json(
        matrix_dir / "matrix_summary.json",
        {"results": results, "comparisons": comparisons},
    )
    print(f"\nKE-V2 matrix complete: {matrix_dir}")
    print(f"Summary: {matrix_dir / 'SUMMARY.md'}")
    failed = [result["experiment_id"] for result in results if result["status"] == "failed"]
    if failed:
        raise SystemExit(f"Failed experiments: {failed}")


if __name__ == "__main__":
    main()

