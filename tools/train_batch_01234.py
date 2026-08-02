"""Batch runner for the 01234 five-class experiments.

This wrapper reuses tools/train_batch.py while pinning the 01234 grid30 408
common config and the current experiment queue.
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
    "configs/fixed_split_01234_grid30_408_train.yaml"
)

CONFIG_DIR = Path(
    "configs/fixed_split_01234_models/transformer_vit_swin_grid30_408"
)
CONFIG_NAMES = (
    "fixed_timm_vit_small_patch16_224_grid30_408",
    "fixed_timm_swin_tiny_patch4_window7_224_grid30_408",
)
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
