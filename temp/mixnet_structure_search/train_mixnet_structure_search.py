"""Run generated MixNet-S structure search configs with fixed BaSiC settings."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_TRAIN_BATCH_PATH = PROJECT_ROOT / "tools" / "train_batch.py"
COMMON_CONFIG = Path("temp/mixnet_structure_search/common_01234_basic_408_train.yaml")
CONFIG_ROOT = Path("temp/mixnet_structure_search/configs")

PHASE_DIRS = {
    "position": CONFIG_ROOT / "position",
    "stage_mask": CONFIG_ROOT / "stage_mask",
    "kernel_continuous": CONFIG_ROOT / "kernel_continuous",
    "gates": CONFIG_ROOT / "gates",
}
KEEP_PTH_FLAGS = {"--keep-pth", "--keep-pth-files"}
DISCARD_PTH_FLAGS = {"--discard-pth", "--discard-pth-files"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_train_batch_module():
    spec = importlib.util.spec_from_file_location("train_batch_mixnet_search_base", BASE_TRAIN_BATCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base batch trainer: {BASE_TRAIN_BATCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_batch_mixnet_search_base"] = module
    spec.loader.exec_module(module)
    return module


def parse_phase_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=("position", "stage_mask", "kernel_continuous", "gates", "smoke", "all"),
        default=("position",),
        help="Generated config phase to run before forwarding remaining args to train_batch.py.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional directory of YAML configs to run instead of --phase.",
    )
    return parser.parse_known_args(argv)


def collect_config_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.config_dir is not None:
        directory = args.config_dir
        paths = sorted(directory.glob("*.yaml"))
        if not paths:
            raise FileNotFoundError(f"No YAML configs found in {directory}")
        return tuple(paths)

    phases = list(args.phase)
    if "all" in phases:
        phases = ["position", "stage_mask", "kernel_continuous", "gates"]
    if "smoke" in phases:
        smoke_dir = PHASE_DIRS["position"]
        paths = [
            smoke_dir / "00_p00_original.yaml",
            smoke_dir / "01_p01_none_k3.yaml",
            smoke_dir / "02_p02_all_k357.yaml",
        ]
    else:
        paths = []
        for phase in phases:
            directory = PHASE_DIRS[phase]
            paths.extend(sorted(directory.glob("*.yaml")))

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing generated configs. Run: python temp\\mixnet_structure_search\\generate_mixnet_search_configs.py"
        )
    return tuple(paths)


def enforce_discard_pth(args: list[str]) -> list[str]:
    if any(arg in KEEP_PTH_FLAGS for arg in args):
        raise ValueError("MixNet structure search does not retain .pth files; remove --keep-pth.")
    if any(arg in DISCARD_PTH_FLAGS for arg in args):
        return args
    return [*args, "--discard-pth"]


def _install_memory_only_checkpoints(train_batch_base: Any) -> None:
    class InMemoryCheckpointHook:
        def __init__(self, metric: str, mode: str, min_delta: float = 0.0) -> None:
            self.metric = metric
            self.mode = mode
            self.min_delta = float(min_delta or 0.0)
            self.best_value = float("-inf") if mode == "max" else float("inf")
            self.best_epoch = 0
            self.has_saved_once = False

        def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
            if self.metric not in metrics:
                return

            value = float(metrics[self.metric])
            if self.mode == "max":
                is_best = value > self.best_value + self.min_delta
            else:
                is_best = value < self.best_value - self.min_delta

            if not (is_best or not self.has_saved_once):
                return

            self.best_value = value
            self.best_epoch = epoch
            self.has_saved_once = True
            trainer._memory_best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in trainer.model.state_dict().items()
                },
                "metrics": dict(metrics),
                self.metric: value,
            }

    class MemoryCheckpointTrainer(train_batch_base.Trainer):
        def _setup_hooks(self):
            hook_manager = super()._setup_hooks()
            for index, hook in enumerate(hook_manager.hooks):
                if hook.__class__.__name__ != "CheckpointHook":
                    continue
                hook_manager.hooks[index] = InMemoryCheckpointHook(
                    metric=hook.metric,
                    mode=hook.mode,
                    min_delta=hook.min_delta,
                )
                if self.logger:
                    self.logger.info("checkpoint_hook_replaced", mode="memory_only")
            return hook_manager

    def evaluate_best_checkpoint_from_memory(
        trainer: Any,
        training_time_seconds: float,
    ) -> dict[str, Any]:
        checkpoint = getattr(trainer, "_memory_best_checkpoint", None)
        if checkpoint is None:
            raise FileNotFoundError("No in-memory best checkpoint was captured during training.")

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        trainer.model.load_state_dict(state_dict)
        trainer.model.eval()

        test_metrics = trainer.evaluator.evaluate(
            trainer.test_loader,
            trainer.loss_fn,
            desc="Testing best in-memory checkpoint",
        )
        class_names = list(trainer.test_loader.dataset.classes)
        confusion_matrix = trainer.evaluator.compute_confusion_matrix(
            trainer.test_loader,
            num_classes=len(class_names),
        )
        per_class_accuracy: dict[str, float] = {}
        for class_index, class_name in enumerate(class_names):
            class_total = int(confusion_matrix[class_index, :].sum())
            class_correct = int(confusion_matrix[class_index, class_index])
            per_class_accuracy[class_name] = class_correct / class_total if class_total else 0.0

        result = train_batch_base.to_builtin(test_metrics)
        result.update(
            {
                "class_names": class_names,
                "per_class_accuracy": per_class_accuracy,
                "confusion_matrix": train_batch_base.to_builtin(confusion_matrix),
                "num_samples": int(confusion_matrix.sum()),
                "best_epoch": int(checkpoint.get("epoch", -1)) + 1,
                "best_validation_metrics": train_batch_base.to_builtin(checkpoint.get("metrics", {})),
            }
        )
        result["training_time_seconds"] = training_time_seconds
        result.update(train_batch_base.compute_classification_details(confusion_matrix, class_names))
        image_size = int(getattr(trainer.config.data, "image_size", 224) or 224)
        result.update(
            train_batch_base.profile_model_complexity(
                trainer.model,
                trainer.device,
                image_size=image_size,
            )
        )
        result.update(
            train_batch_base.measure_inference_time(
                trainer.model,
                trainer.test_loader,
                trainer.device,
            )
        )
        result.update(train_batch_base.save_probabilistic_prediction_csv(trainer))
        result.update(train_batch_base.save_ordinal_representation_artifacts(trainer))
        return result

    train_batch_base.Trainer = MemoryCheckpointTrainer
    train_batch_base.evaluate_best_checkpoint = evaluate_best_checkpoint_from_memory


def main() -> None:
    phase_args, remaining = parse_phase_args(sys.argv[1:])
    config_paths = collect_config_paths(phase_args)
    remaining = enforce_discard_pth(remaining)

    train_batch_base = _load_train_batch_module()
    _install_memory_only_checkpoints(train_batch_base)
    train_batch_base.COMMON_CONFIG = COMMON_CONFIG
    train_batch_base.CONFIG_LIST = config_paths
    train_batch_base.PYCHARM_DEVICE = "auto"
    train_batch_base.PYCHARM_DRY_RUN = False
    train_batch_base.PYCHARM_FAIL_FAST = False
    train_batch_base.PYCHARM_KEEP_PTH_FILES = False

    sys.argv = [sys.argv[0], *remaining]
    train_batch_base.main()


if __name__ == "__main__":
    main()
