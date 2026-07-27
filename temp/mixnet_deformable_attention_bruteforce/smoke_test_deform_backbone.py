"""Smoke-test the local MixNet-S deformable-attention backbone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from mixnet_deformable_backbone import MixNetSDeformableBackbone  # noqa: E402


def assert_deform_grads(model: MixNetSDeformableBackbone) -> None:
    missing = [
        name
        for name, parameter in model.named_parameters()
        if "deform_blocks" in name and parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise RuntimeError(f"Missing gradients for deformable parameters: {missing[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test MixNet-S deformable backbone.")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    baseline = MixNetSDeformableBackbone(
        pretrained=False,
        input_size=args.input_size,
        deform_stage_ids=[],
    ).to(device)
    if baseline.deform_blocks:
        raise AssertionError("Baseline branch should not contain deformable blocks.")
    if len(baseline.stage_infos) != 5:
        raise AssertionError(f"Expected 5 discovered stages, got {len(baseline.stage_infos)}.")

    model = MixNetSDeformableBackbone(
        pretrained=False,
        input_size=args.input_size,
        deform_stage_ids=[0, 1, 2, 3, 4],
    ).to(device)
    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)
    y = model(x)
    if y.ndim != 2 or y.shape[0] != args.batch_size:
        raise AssertionError(f"Expected [B, D] features, got {tuple(y.shape)}.")
    if not torch.isfinite(y).all():
        raise AssertionError("Backbone output contains non-finite values.")

    loss = y.mean()
    loss.backward()
    assert_deform_grads(model)

    print("Smoke test passed.")
    print("Discovered stages:")
    for info in model.describe_deformable_stages():
        print(f"  S{info['stage_id']}: {info['channels']} x {info['height']} x {info['width']}")


if __name__ == "__main__":
    main()
