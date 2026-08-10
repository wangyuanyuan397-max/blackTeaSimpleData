"""Batch runner for the 01234 five-class experiments.

This wrapper reuses tools/train_batch.py while pinning the 01234 BaSiC grid30
408 common config and the MixNet-S SAA three-seed reproduction queue.
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

CONFIG_DIR = Path(
    "configs/fixed_split_01234_models/mixnet_saa_fusion/seed_reproduction"
)
CONFIG_NAMES_FILE = CONFIG_DIR / "CONFIG_NAMES.txt"
EXPECTED_CONFIG_COUNT = 12


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
            "This 01234 batch discards .pth files; please do not pass --keep-pth."
        )
    train_batch_base.main()


if __name__ == "__main__":
    main()
