"""Run the MixNet-S deformable-attention brute-force configs."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
BASE_TRAIN_BATCH_PATH = PROJECT_ROOT / "tools" / "train_batch.py"
COMMON_CONFIG = Path("temp/mixnet_deformable_attention_bruteforce/common_01234_grid30_408_train.yaml")
CONFIG_ROOT = Path("temp/mixnet_deformable_attention_bruteforce/configs")
RUNS_ROOT = PROJECT_ROOT / "temp/mixnet_deformable_attention_bruteforce/runs_grid30_408"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def _load_train_batch_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "train_batch_deform_attention_base",
        BASE_TRAIN_BATCH_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base batch trainer: {BASE_TRAIN_BATCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_batch_deform_attention_base"] = module
    spec.loader.exec_module(module)
    return module


def parse_local_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional directory of generated YAML configs to run.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Regenerate the 96 YAML configs before running.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip CSV summarization after training.",
    )
    return parser.parse_known_args(argv)


def collect_config_paths(config_dir: Path | None) -> tuple[Path, ...]:
    directory = CONFIG_ROOT if config_dir is None else config_dir
    paths = tuple(sorted(directory.glob("D[01][01][01][01][01]_seed*.yaml")))
    if paths:
        return paths

    from generate_deform_configs import generate_configs

    generate_configs()
    paths = tuple(sorted(directory.glob("D[01][01][01][01][01]_seed*.yaml")))
    if not paths:
        raise FileNotFoundError(f"No generated YAML configs found in {directory}.")
    return paths


def _collect_stage_metadata(trainer: Any) -> dict[str, Any]:
    backbone = getattr(getattr(trainer, "model", None), "backbone", None)
    if backbone is None:
        return {}
    metadata: dict[str, Any] = {}
    describe = getattr(backbone, "describe_deformable_stages", None)
    if callable(describe):
        metadata["stage_infos"] = describe()
    diagnostics = getattr(backbone, "collect_deformable_diagnostics", None)
    if callable(diagnostics):
        metadata["diagnostics"] = diagnostics()
    return metadata


def install_validation_detail_patch(train_batch_base: Any) -> None:
    original_evaluate_best_checkpoint = train_batch_base.evaluate_best_checkpoint

    def evaluate_best_checkpoint_with_validation_details(trainer: Any, training_time_seconds: float):
        result = original_evaluate_best_checkpoint(trainer, training_time_seconds)

        val_metrics = trainer.evaluator.evaluate(
            trainer.val_loader,
            trainer.loss_fn,
            desc="Validating best checkpoint details",
        )
        class_names = list(trainer.val_loader.dataset.classes)
        confusion_matrix = trainer.evaluator.compute_confusion_matrix(
            trainer.val_loader,
            num_classes=len(class_names),
        )
        val_details = train_batch_base.compute_classification_details(
            confusion_matrix,
            class_names,
        )
        detailed = train_batch_base.to_builtin(val_metrics)
        detailed.update(train_batch_base.to_builtin(val_details))
        detailed["confusion_matrix"] = train_batch_base.to_builtin(confusion_matrix)
        detailed["class_names"] = class_names

        result["best_validation_detailed_metrics"] = detailed
        result["best_val_macro_f1"] = detailed.get("macro_f1")
        result["best_val_accuracy_detailed"] = detailed.get("accuracy")
        result["deformable_attention"] = _collect_stage_metadata(trainer)
        return result

    train_batch_base.evaluate_best_checkpoint = evaluate_best_checkpoint_with_validation_details


def should_summarize(forwarded_args: list[str], no_summary: bool) -> bool:
    if no_summary:
        return False
    return "--dry-run" not in forwarded_args and "--list-models" not in forwarded_args


def main() -> None:
    local_args, forwarded_args = parse_local_args(sys.argv[1:])
    if local_args.generate:
        from generate_deform_configs import generate_configs

        generate_configs()

    config_paths = collect_config_paths(local_args.config_dir)
    summarize_after_run = should_summarize(forwarded_args, local_args.no_summary)

    train_batch_base = _load_train_batch_module()
    import mixnet_deformable_backbone  # noqa: F401 - registers mixnet_s_deformable

    install_validation_detail_patch(train_batch_base)

    train_batch_base.COMMON_CONFIG = COMMON_CONFIG
    train_batch_base.CONFIG_LIST = config_paths
    train_batch_base.PYCHARM_DEVICE = "auto"
    train_batch_base.PYCHARM_DRY_RUN = False
    train_batch_base.PYCHARM_FAIL_FAST = False
    train_batch_base.PYCHARM_KEEP_PTH_FILES = False

    exit_error: BaseException | None = None
    try:
        sys.argv = [sys.argv[0], *forwarded_args]
        train_batch_base.main()
    except BaseException as exc:
        exit_error = exc
    finally:
        if summarize_after_run and RUNS_ROOT.exists():
            from summarize_deform_results import summarize_results

            summarize_results(RUNS_ROOT)
    if exit_error is not None:
        raise exit_error


if __name__ == "__main__":
    main()
