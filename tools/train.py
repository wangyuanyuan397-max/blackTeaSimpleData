"""Train one YAML configuration without LOGOCV-specific orchestration."""

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models  # noqa: E402,F401 - populate registries
from src.engine import ComponentBuilder, Trainer  # noqa: E402
from src.schemas import load_config  # noqa: E402
from src.utils import configure_logging, get_logger  # noqa: E402


def main() -> None:
    # 未提供单配置参数时，直接进入固定 train/val/test 的多模型批量训练入口。
    if '--config' not in sys.argv[1:] and '-c' not in sys.argv[1:]:
        from train_batch import main as run_batch

        run_batch()
        return
    parser = argparse.ArgumentParser(description="Train one experiment configuration.")
    parser.add_argument("-c", "--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()

    config = load_config(args.config, validate=False)
    configure_logging(
        log_level=getattr(config, "log_level", "INFO"),
        log_format=getattr(config, "log_format", "console"),
    )
    logger = get_logger("train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = ComponentBuilder(config, device, logger)
    trainer = Trainer(config=config, builder=builder, logger=logger, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
