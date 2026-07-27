"""Run generated MixNet-S explicit scale-interaction configs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
BASE_TRAIN_BATCH_PATH = PROJECT_ROOT / "tools" / "train_batch.py"
LOCAL_BACKBONE_PATH = THIS_DIR / "mixnet_explicit_scale_interaction_backbone.py"

COMMON_CONFIG = Path("temp/mixnet_explicit_scale_interaction/common_01234_basic_408_train.yaml")
CONFIG_ROOT = Path("temp/mixnet_explicit_scale_interaction/configs")

PHASE_DIRS = {
    "operators": CONFIG_ROOT / "operators",
    "edge_modes": CONFIG_ROOT / "edge_modes",
    "stage_single": CONFIG_ROOT / "stage_single",
    "stage_mask": CONFIG_ROOT / "stage_mask",
    "block_single": CONFIG_ROOT / "block_single",
    "block_subset": CONFIG_ROOT / "block_subset",
}

KEEP_PTH_FLAGS = {"--keep-pth", "--keep-pth-files"}
DISCARD_PTH_FLAGS = {"--discard-pth", "--discard-pth-files"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_local_backbone() -> None:
    _load_module_from_path("mixnet_explicit_scale_interaction_backbone", LOCAL_BACKBONE_PATH)


def _load_train_batch_module():
    return _load_module_from_path("train_batch_explicit_scale_interaction_base", BASE_TRAIN_BATCH_PATH)


def parse_phase_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=(
            "operators",
            "edge_modes",
            "stage_single",
            "stage_mask",
            "block_single",
            "block_subset",
            "smoke",
            "all",
        ),
        default=("operators",),
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
        phases = ["operators", "edge_modes"]

    if "smoke" in phases:
        paths = [
            PHASE_DIRS["operators"] / "00_baseline.yaml",
            PHASE_DIRS["operators"] / "01_scale_attention.yaml",
            PHASE_DIRS["operators"] / "04_weighted_sum.yaml",
            PHASE_DIRS["operators"] / "05_cross_residual_bidir.yaml",
        ]
    else:
        paths = []
        for phase in phases:
            directory = PHASE_DIRS[phase]
            pattern = "edge_*.yaml" if phase == "edge_modes" else "*.yaml"
            paths.extend(sorted(directory.glob(pattern)))

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing generated configs. Run: "
            "python temp\\mixnet_explicit_scale_interaction\\generate_explicit_scale_interaction_configs.py --phase all"
        )
    return tuple(paths)


def default_discard_pth(args: list[str]) -> list[str]:
    if any(arg in KEEP_PTH_FLAGS or arg in DISCARD_PTH_FLAGS for arg in args):
        return args
    return [*args, "--discard-pth"]


def _find_explicit_scale_backbone(model: Any) -> Any | None:
    if hasattr(model, "module"):
        model = model.module
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "interaction_summary"):
        return backbone
    return None


def _install_explicit_scale_metrics(train_batch_base: Any) -> None:
    original_evaluate = train_batch_base.evaluate_best_checkpoint

    def evaluate_best_checkpoint_with_interaction_metadata(
        trainer: Any,
        training_time_seconds: float,
    ) -> dict[str, Any]:
        metrics = original_evaluate(trainer, training_time_seconds)
        backbone = _find_explicit_scale_backbone(trainer.model)
        if backbone is not None:
            metrics["mixnet_explicit_scale_interaction"] = train_batch_base.to_builtin(
                backbone.interaction_summary()
            )
        return metrics

    original_write_csv = train_batch_base.write_ablation_csv_files

    def write_ablation_csv_files_with_interaction_summary(
        runs_root: Path,
        batch_timestamp: str,
        results: list[dict[str, Any]],
    ) -> list[Path]:
        written_paths = original_write_csv(runs_root, batch_timestamp, results)
        rows = []
        for result in results:
            metrics = result.get("metrics") or {}
            search = metrics.get("mixnet_explicit_scale_interaction")
            if not search and "mixnet_explicit_scale_interaction" not in str(result.get("config_path", "")):
                continue
            search = search or {}
            rows.append(
                {
                    "batch_timestamp": batch_timestamp,
                    "model_name": result.get("model_name"),
                    "status": result.get("status"),
                    "config_path": result.get("config_path"),
                    "interaction_type": search.get("interaction_type"),
                    "placement": search.get("placement"),
                    "stage_mask": "".join("1" if value else "0" for value in (search.get("stage_mask") or [])),
                    "block_indices": json.dumps(search.get("block_indices"), ensure_ascii=True),
                    "edge_mode": search.get("edge_mode"),
                    "residual_strength": search.get("residual_strength"),
                    "attention_hidden_dim": search.get("attention_hidden_dim"),
                    "modified_block_count": search.get("modified_block_count"),
                    "skipped_block_count": search.get("skipped_block_count"),
                    "accuracy": result.get("accuracy"),
                    "macro_f1": result.get("macro_f1"),
                    "qwk": result.get("qwk"),
                    "parameters_total": result.get("parameters_total"),
                    "parameters_trainable": result.get("parameters_trainable"),
                    "flops_g": result.get("flops_g"),
                    "run_directory": result.get("run_directory"),
                    "error": result.get("error"),
                }
            )
        if rows:
            path = runs_root / "explicit_scale_interaction_summary.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            written_paths.append(path)
        return written_paths

    train_batch_base.evaluate_best_checkpoint = evaluate_best_checkpoint_with_interaction_metadata
    train_batch_base.write_ablation_csv_files = write_ablation_csv_files_with_interaction_summary


def main() -> None:
    phase_args, remaining = parse_phase_args(sys.argv[1:])
    config_paths = collect_config_paths(phase_args)
    remaining = default_discard_pth(remaining)

    _load_local_backbone()
    train_batch_base = _load_train_batch_module()
    _install_explicit_scale_metrics(train_batch_base)

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
