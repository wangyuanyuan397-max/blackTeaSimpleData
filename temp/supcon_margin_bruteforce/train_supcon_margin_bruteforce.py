"""Queue SupCon + Margin brute-force configs through the shared batch trainer."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
BASE_TRAIN_BATCH_PATH = PROJECT_ROOT / "tools" / "train_batch.py"
COMMON_CONFIG = Path("temp/supcon_margin_bruteforce/common_01234_basic_408_train.yaml")
CONFIG_ROOT = Path("temp/supcon_margin_bruteforce/configs")
RUNS_ROOT = PROJECT_ROOT / "temp/supcon_margin_bruteforce/runs_BaSic"

PHASE_DIRS = {
    "baseline": CONFIG_ROOT / "baseline",
    "smoke": CONFIG_ROOT / "smoke",
    "core": CONFIG_ROOT / "core",
    "full": CONFIG_ROOT / "full",
}
# 暴力搜索实验统一不保留 .pth 权重，只保留日志、指标和汇总表。
KEEP_PTH_FLAGS = {"--keep-pth", "--keep-pth-files"}
DISCARD_PTH_FLAGS = {"--discard-pth", "--discard-pth-files"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def _load_train_batch_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "train_batch_supcon_margin_base",
        BASE_TRAIN_BATCH_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base batch trainer: {BASE_TRAIN_BATCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_batch_supcon_margin_base"] = module
    spec.loader.exec_module(module)
    return module


def parse_local_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=("baseline", "smoke", "core", "full", "all", "exhaustive"),
        default=("smoke",),
        help="SupCon+Margin config phase to run before forwarding remaining args.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional directory of YAML configs to run instead of --phase.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Regenerate SupCon+Margin YAML configs before collecting them.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip CSV summarization after training.",
    )
    return parser.parse_known_args(argv)


def expand_phases(phases: list[str]) -> list[str]:
    if "all" in phases:
        return ["baseline", "core"]
    if "exhaustive" in phases:
        return ["baseline", "full"]
    return phases


def collect_config_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.config_dir is not None:
        paths = tuple(sorted(args.config_dir.glob("*.yaml")))
        if not paths:
            raise FileNotFoundError(f"No YAML configs found in {args.config_dir}")
        return paths

    paths: list[Path] = []
    for phase in expand_phases(list(args.phase)):
        directory = PHASE_DIRS[phase]
        paths.extend(sorted(directory.glob("*.yaml")))
    missing_dirs = [
        str(PHASE_DIRS[phase])
        for phase in expand_phases(list(args.phase))
        if not PHASE_DIRS[phase].is_dir()
    ]
    if missing_dirs or not paths:
        raise FileNotFoundError(
            "Missing SupCon+Margin configs. Run: "
            "python temp\\supcon_margin_bruteforce\\generate_supcon_margin_configs.py --phase all"
        )
    return tuple(paths)


def enforce_discard_pth(args: list[str]) -> list[str]:
    # 如果误传保留权重的参数，立即停止；否则自动追加 --discard-pth。
    if any(arg in KEEP_PTH_FLAGS for arg in args):
        raise ValueError("SupCon+Margin brute-force runs discard .pth files; remove --keep-pth.")
    if any(arg in DISCARD_PTH_FLAGS for arg in args):
        return args
    return [*args, "--discard-pth"]


def should_summarize(forwarded_args: list[str], no_summary: bool) -> bool:
    return not no_summary and "--dry-run" not in forwarded_args and "--list-models" not in forwarded_args


def main() -> None:
    local_args, forwarded_args = parse_local_args(sys.argv[1:])
    if local_args.generate:
        from generate_supcon_margin_configs import main as generate_main

        old_argv = sys.argv
        try:
            generate_phase = "all" if "all" in local_args.phase else local_args.phase[0]
            if "exhaustive" in local_args.phase:
                generate_phase = "exhaustive"
            sys.argv = ["generate_supcon_margin_configs.py", "--phase", generate_phase]
            generate_main()
        finally:
            sys.argv = old_argv

    config_paths = collect_config_paths(local_args)
    forwarded_args = enforce_discard_pth(forwarded_args)
    summarize_after_run = should_summarize(forwarded_args, local_args.no_summary)

    train_batch_base = _load_train_batch_module()
    train_batch_base.COMMON_CONFIG = COMMON_CONFIG
    train_batch_base.CONFIG_LIST = config_paths
    train_batch_base.PYCHARM_DEVICE = "auto"
    train_batch_base.PYCHARM_DRY_RUN = False
    train_batch_base.PYCHARM_FAIL_FAST = False
    # PyCharm/脚本入口同样强制不保留权重，和命令行 --discard-pth 保持一致。
    train_batch_base.PYCHARM_KEEP_PTH_FILES = False

    exit_error: BaseException | None = None
    try:
        sys.argv = [sys.argv[0], *forwarded_args]
        train_batch_base.main()
    except BaseException as exc:
        exit_error = exc
    finally:
        if summarize_after_run and RUNS_ROOT.exists():
            from summarize_supcon_margin_results import summarize_results

            summarize_results(RUNS_ROOT)
    if exit_error is not None:
        raise exit_error


if __name__ == "__main__":
    main()
