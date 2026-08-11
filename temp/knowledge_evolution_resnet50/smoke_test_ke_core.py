"""Fast, data-free checks for the isolated KELS implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ke_core import (
    create_kels_masks,
    is_better_validation_checkpoint,
    reset_reset_hypothesis,
    reset_reset_hypothesis_from_fresh_model,
    summarize_kels_masks,
)


class TinyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.conv2(self.bn1(self.conv1(inputs)))
        return self.fc(self.pool(features).flatten(1))


def main() -> None:
    torch.manual_seed(7)
    model = TinyNetwork()
    masks = create_kels_masks(model, split_rate=0.5)
    repeated_masks = create_kels_masks(model, split_rate=0.5)

    assert all(torch.equal(masks[name], repeated_masks[name]) for name in masks)
    assert int(masks["conv1"].sum()) == 4 * 3 * 3 * 3
    assert int(masks["conv2"].sum()) == 4 * 4 * 3 * 3
    assert int(masks["fc"].sum()) == 5 * 4

    old_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    report = reset_reset_hypothesis(model, masks, seed=2027)

    target_modules = dict(model.named_modules())
    for name, mask in masks.items():
        old_weight = old_state[f"{name}.weight"]
        new_weight = target_modules[name].weight.detach()
        assert torch.equal(new_weight[mask], old_weight[mask])
        assert torch.count_nonzero(new_weight[~mask] != old_weight[~mask]) > 0

    # BN parameters/statistics and classifier bias are fully inherited.
    for state_name in ("bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var"):
        assert torch.equal(model.state_dict()[state_name], old_state[state_name])
    assert torch.equal(model.fc.bias.detach(), old_state["fc.bias"])
    assert report["max_fit_abs_difference"] == 0.0
    assert report["reset_changed_fraction"] > 0.99
    assert model(torch.randn(2, 3, 16, 16)).shape == (2, 5)

    # V2 copies RESET weights from a complete fresh model while preserving BN/bias.
    torch.manual_seed(8)
    v2_model = TinyNetwork()
    v2_masks = create_kels_masks(v2_model, split_rate=0.5)
    v2_old_state = {
        name: value.detach().clone() for name, value in v2_model.state_dict().items()
    }
    torch.manual_seed(2028)
    fresh_model = TinyNetwork()
    v2_report = reset_reset_hypothesis_from_fresh_model(
        v2_model,
        fresh_model,
        v2_masks,
    )
    v2_modules = dict(v2_model.named_modules())
    fresh_modules = dict(fresh_model.named_modules())
    for name, mask in v2_masks.items():
        assert torch.equal(
            v2_modules[name].weight.detach()[mask],
            v2_old_state[f"{name}.weight"][mask],
        )
        assert torch.equal(
            v2_modules[name].weight.detach()[~mask],
            fresh_modules[name].weight.detach()[~mask],
        )
    for state_name in ("bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var"):
        assert torch.equal(v2_model.state_dict()[state_name], v2_old_state[state_name])
    assert torch.equal(v2_model.fc.bias.detach(), v2_old_state["fc.bias"])
    assert v2_report["reset_changed_fraction"] > 0.99

    # Fixed tie-break: accuracy > QWK > lower validation loss.
    best = {"val_acc": 0.75, "val_qwk": 0.92, "val_loss": 0.8}
    assert is_better_validation_checkpoint(
        {"val_acc": 0.76, "val_qwk": 0.1, "val_loss": 9.0}, best
    )
    assert is_better_validation_checkpoint(
        {"val_acc": 0.75, "val_qwk": 0.93, "val_loss": 9.0}, best
    )
    assert is_better_validation_checkpoint(
        {"val_acc": 0.75, "val_qwk": 0.92, "val_loss": 0.7}, best
    )
    assert not is_better_validation_checkpoint(
        {"val_acc": 0.75, "val_qwk": 0.92, "val_loss": 0.9}, best
    )

    summary = summarize_kels_masks(model, masks, split_rate=0.5)
    print(
        "KE core smoke test passed: "
        f"layers={summary['target_layer_count']}, "
        f"fit={summary['fit_fraction']:.4f}, "
        f"reset={summary['reset_fraction']:.4f}"
    )


if __name__ == "__main__":
    main()
