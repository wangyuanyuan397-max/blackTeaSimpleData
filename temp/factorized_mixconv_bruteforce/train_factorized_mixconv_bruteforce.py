"""Run the Factorized MixConv brute-force configs through tools/train_batch.py."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
BASE_TRAIN_BATCH_PATH = PROJECT_ROOT / "tools" / "train_batch.py"
COMMON_CONFIG = Path("temp/factorized_mixconv_bruteforce/common_01234_basic_408_train.yaml")
CONFIG_ROOT = Path("temp/factorized_mixconv_bruteforce/configs")
RUNS_ROOT = PROJECT_ROOT / "temp/factorized_mixconv_bruteforce/runs_BaSic"
KEEP_PTH_FLAGS = {"--keep-pth", "--keep-pth-files"}
DISCARD_PTH_FLAGS = {"--discard-pth", "--discard-pth-files"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def _load_train_batch_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "train_batch_factorized_mixconv_base",
        BASE_TRAIN_BATCH_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base batch trainer: {BASE_TRAIN_BATCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_batch_factorized_mixconv_base"] = module
    spec.loader.exec_module(module)
    return module


def parse_local_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional directory of YAML configs to run instead of the generated 24-config set.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Regenerate the 24 YAML configs before running.",
    )
    return parser.parse_known_args(argv)


def enforce_discard_pth(args: list[str]) -> list[str]:
    if any(arg in KEEP_PTH_FLAGS for arg in args):
        raise ValueError("This brute-force batch does not retain .pth files; remove --keep-pth.")
    if any(arg in DISCARD_PTH_FLAGS for arg in args):
        return args
    return [*args, "--discard-pth"]


def collect_config_paths(config_dir: Path | None) -> tuple[Path, ...]:
    directory = CONFIG_ROOT if config_dir is None else config_dir
    paths = tuple(sorted(directory.glob("F*_seed*.yaml")))
    if not paths:
        raise FileNotFoundError(
            f"No generated YAML configs found in {directory}. "
            "Run: python temp\\factorized_mixconv_bruteforce\\generate_factorized_mixconv_configs.py"
        )
    return paths


def should_summarize(forwarded_args: list[str]) -> bool:
    return "--dry-run" not in forwarded_args and "--list-models" not in forwarded_args


def main() -> None:
    local_args, forwarded_args = parse_local_args(sys.argv[1:])
    if local_args.generate:
        from generate_factorized_mixconv_configs import generate_configs

        generate_configs()

    config_paths = collect_config_paths(local_args.config_dir)
    forwarded_args = enforce_discard_pth(forwarded_args)
    summarize_after_run = should_summarize(forwarded_args)

    train_batch_base = _load_train_batch_module()
    import factorized_mixconv_backbone  # noqa: F401 - registers mixnet_s_factorized

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
            from summarize_factorized_mixconv_results import summarize_results

            summarize_results(RUNS_ROOT)
    if exit_error is not None:
        raise exit_error


if __name__ == "__main__":
    main()
