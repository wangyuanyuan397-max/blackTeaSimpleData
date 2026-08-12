"""Batch runner for the 01234 five-class experiments.

This wrapper reuses tools/train_batch.py while pinning the 01234 BaSiC grid30
408 common config. Ortho-Shot remains the default; wrapper flags select the
isolated RePr diagnostic or SAFNet paper/source comparison queues.
"""

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN_BATCH_PATH = PROJECT_ROOT / "tools" / "train_batch.py"

spec = importlib.util.spec_from_file_location("train_batch_base", BASE_TRAIN_BATCH_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base batch trainer: {BASE_TRAIN_BATCH_PATH}")

train_batch_base = importlib.util.module_from_spec(spec)
sys.modules["train_batch_base"] = train_batch_base
spec.loader.exec_module(train_batch_base)

train_batch_base.COMMON_CONFIG = Path(
    "configs/fixed_split_01234_BaSic_grid30_408_train.yaml"
)

ORTHOSHOT_PHASE_COUNTS = {
    "phase1_dbt": 12,
    "phase2_single_augmentation": 16,
    "phase3_combined_augmentation": 7,
    "phase4_maxup": 5,
}


def _consume_orthoshot_phase(argv: list[str]) -> str:
    """Read the wrapper-only phase option before the shared parser runs."""
    phase = "phase1_dbt"
    matches = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--orthoshot-phase":
            if index + 1 >= len(argv):
                raise SystemExit("--orthoshot-phase requires a phase name.")
            matches.append(argv[index + 1])
            del argv[index : index + 2]
            continue
        if argument.startswith("--orthoshot-phase="):
            matches.append(argument.split("=", 1)[1])
            del argv[index]
            continue
        index += 1
    if len(matches) > 1:
        raise SystemExit("--orthoshot-phase may only be specified once.")
    if matches:
        phase = matches[0]
    if phase not in ORTHOSHOT_PHASE_COUNTS:
        choices = ", ".join(ORTHOSHOT_PHASE_COUNTS)
        raise SystemExit(f"Unknown Ortho-Shot phase {phase!r}; choose one of: {choices}.")
    return phase


REPR_DIAGNOSTIC = "--repr-diagnostic" in sys.argv
SAFNET_COMPARISON = "--safnet-comparison" in sys.argv
if REPR_DIAGNOSTIC and SAFNET_COMPARISON:
    raise SystemExit(
        "--repr-diagnostic and --safnet-comparison cannot be used together."
    )

if REPR_DIAGNOSTIC:
    sys.argv.remove("--repr-diagnostic")
    if any(argument.startswith("--orthoshot-phase") for argument in sys.argv):
        raise SystemExit(
            "--repr-diagnostic and --orthoshot-phase cannot be used together."
        )
    ORTHOSHOT_PHASE = None
    CONFIG_DIR = Path("configs/fixed_split_01234_models/mixnet_repr")
    EXPECTED_CONFIG_COUNT = 6
elif SAFNET_COMPARISON:
    sys.argv.remove("--safnet-comparison")
    if any(argument.startswith("--orthoshot-phase") for argument in sys.argv):
        raise SystemExit(
            "--safnet-comparison and --orthoshot-phase cannot be used together."
        )
    ORTHOSHOT_PHASE = None
    CONFIG_DIR = Path("configs/fixed_split_01234_models/safnet_comparison")
    EXPECTED_CONFIG_COUNT = 4
else:
    ORTHOSHOT_PHASE = _consume_orthoshot_phase(sys.argv)
    CONFIG_DIR = Path(
        f"configs/fixed_split_01234_models/mixnet_orthoshot/{ORTHOSHOT_PHASE}"
    )
    EXPECTED_CONFIG_COUNT = ORTHOSHOT_PHASE_COUNTS[ORTHOSHOT_PHASE]
CONFIG_NAMES_FILE = CONFIG_DIR / "CONFIG_NAMES.txt"


def _load_config_names(path: Path) -> tuple[str, ...]:
    names = tuple(
        line.strip()
        for line in (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(names) != EXPECTED_CONFIG_COUNT:
        raise RuntimeError(
            f"{path.as_posix()} should list {EXPECTED_CONFIG_COUNT} configs, "
            f"got {len(names)}."
        )
    return names


CONFIG_NAMES = _load_config_names(CONFIG_NAMES_FILE)
train_batch_base.CONFIG_LIST = tuple(CONFIG_DIR / f"{name}.yaml" for name in CONFIG_NAMES)

train_batch_base.PYCHARM_DEVICE = "auto"
train_batch_base.PYCHARM_DRY_RUN = False
train_batch_base.PYCHARM_FAIL_FAST = False
train_batch_base.PYCHARM_KEEP_PTH_FILES = False


def main() -> None:
    if any(arg in {"--keep-pth", "--keep-pth-files"} for arg in sys.argv[1:]):
        raise SystemExit(
            "Checkpoint retention is controlled by each 01234 YAML; do not pass "
            "--keep-pth to the wrapper."
        )
    train_batch_base.main()


if __name__ == "__main__":
    main()
