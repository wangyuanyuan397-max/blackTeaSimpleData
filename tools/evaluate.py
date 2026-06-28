"""Evaluate one checkpoint without LOGOCV-specific aggregation."""

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models  # noqa: E402,F401 - populate registries
from src.engine import ComponentBuilder, Evaluator  # noqa: E402
from src.schemas import load_config  # noqa: E402
from src.utils import configure_logging, get_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one model checkpoint.")
    parser.add_argument("-c", "--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()

    config = load_config(args.config, validate=False)
    configure_logging(
        log_level=getattr(config, "log_level", "INFO"),
        log_format=getattr(config, "log_format", "console"),
    )
    logger = get_logger("evaluate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = ComponentBuilder(config, device, logger)
    _, val_loader, test_loader = builder.build_dataloaders()
    model, strategy = builder.build_model()
    loss_fn = builder.build_loss().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)

    evaluator = Evaluator(model, strategy, device, logger)
    loader = val_loader if args.split == "val" else test_loader
    metrics = evaluator.evaluate(loader, loss_fn)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
