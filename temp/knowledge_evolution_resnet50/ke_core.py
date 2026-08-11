"""KELS masks and generation-boundary reset logic for Knowledge Evolution.

The masks are deliberately absent from ``forward``.  They are used only at a
generation boundary to keep the fit hypothesis and reinitialize the reset
hypothesis.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


TargetModule = nn.Conv2d | nn.Linear


def _validate_split_rate(split_rate: float) -> float:
    split_rate = float(split_rate)
    if not 0.0 < split_rate <= 1.0:
        raise ValueError(f"split_rate must be in (0, 1], got {split_rate}.")
    return split_rate


def _fit_width(width: int, split_rate: float, layer_name: str) -> int:
    fit_width = int(math.floor(int(width) * split_rate))
    if fit_width < 1:
        raise ValueError(
            f"KELS gives an empty fit hypothesis for {layer_name!r}: "
            f"floor({width} * {split_rate}) == 0."
        )
    return fit_width


def iter_kels_targets(model: nn.Module) -> list[tuple[str, TargetModule]]:
    """Return Conv2d and Linear layers in stable ``named_modules`` order."""
    targets: list[tuple[str, TargetModule]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise ValueError(
                    f"KELS in this isolated ResNet50 experiment only supports "
                    f"groups=1, but {name!r} has groups={module.groups}."
                )
            targets.append((name, module))
        elif isinstance(module, nn.Linear):
            targets.append((name, module))
    if not targets:
        raise ValueError("No Conv2d or Linear layers were found for KELS.")
    return targets


@torch.no_grad()
def create_kels_masks(
    model: nn.Module,
    split_rate: float = 0.8,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create one fixed, boolean KELS mask per Conv2d/Linear weight tensor.

    Conv2d fit region:
      * first convolution: all RGB inputs x first ``sr * Cout`` outputs;
      * later convolutions: first ``sr * Cin`` inputs x first ``sr * Cout`` outputs.

    Linear fit region:
      * all output classes x first ``sr * Cin`` input features.
    """
    split_rate = _validate_split_rate(split_rate)
    targets = iter_kels_targets(model)
    first_conv_name = next(
        (name for name, module in targets if isinstance(module, nn.Conv2d)),
        None,
    )
    masks: dict[str, torch.Tensor] = {}

    for name, module in targets:
        mask = torch.zeros(module.weight.shape, dtype=torch.bool, device=device)
        if isinstance(module, nn.Conv2d):
            fit_out = _fit_width(module.out_channels, split_rate, name)
            if name == first_conv_name:
                # KELS keeps all RGB inputs in the first layer.
                mask[:fit_out, :, :, :] = True
            else:
                fit_in = _fit_width(module.in_channels, split_rate, name)
                mask[:fit_out, :fit_in, :, :] = True
        else:
            fit_in = _fit_width(module.in_features, split_rate, name)
            # Never split output classes.
            mask[:, :fit_in] = True
        masks[name] = mask

    return masks


def validate_kels_masks(
    model: nn.Module,
    masks: Mapping[str, torch.Tensor],
) -> None:
    """Fail early if fixed masks no longer match the model topology."""
    targets = dict(iter_kels_targets(model))
    if set(targets) != set(masks):
        missing = sorted(set(targets) - set(masks))
        unexpected = sorted(set(masks) - set(targets))
        raise ValueError(
            f"KELS mask/model layer mismatch; missing={missing}, "
            f"unexpected={unexpected}."
        )
    for name, module in targets.items():
        if tuple(masks[name].shape) != tuple(module.weight.shape):
            raise ValueError(
                f"KELS mask shape mismatch for {name!r}: "
                f"mask={tuple(masks[name].shape)}, "
                f"weight={tuple(module.weight.shape)}."
            )
        if masks[name].dtype != torch.bool:
            raise TypeError(f"KELS mask for {name!r} must be bool.")


def summarize_kels_masks(
    model: nn.Module,
    masks: Mapping[str, torch.Tensor],
    split_rate: float,
) -> dict[str, Any]:
    """Return JSON-ready layer and parameter counts for an audit trail."""
    validate_kels_masks(model, masks)
    targets = dict(iter_kels_targets(model))
    layers: list[dict[str, Any]] = []
    total_parameters = 0
    fit_parameters = 0

    for name, mask in masks.items():
        module = targets[name]
        total = int(mask.numel())
        fit = int(mask.sum().item())
        reset = total - fit
        total_parameters += total
        fit_parameters += fit
        layers.append(
            {
                "name": name,
                "type": type(module).__name__,
                "weight_shape": list(mask.shape),
                "fit_parameters": fit,
                "reset_parameters": reset,
                "fit_fraction": fit / total,
                "reset_fraction": reset / total,
            }
        )

    reset_parameters = total_parameters - fit_parameters
    return {
        "method": "KELS",
        "split_rate": float(split_rate),
        "target_layer_count": len(layers),
        "target_weight_parameters": total_parameters,
        "fit_parameters": fit_parameters,
        "reset_parameters": reset_parameters,
        "fit_fraction": fit_parameters / total_parameters,
        "reset_fraction": reset_parameters / total_parameters,
        "layers": layers,
    }


def validation_checkpoint_key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    """Rank validation checkpoints by acc, then QWK, then lower loss."""

    def metric(*names: str, default: float) -> float:
        for name in names:
            value = metrics.get(name)
            if value is not None:
                return float(value)
        return float(default)

    accuracy = metric("val_acc", "val_accuracy", "accuracy", default=float("-inf"))
    qwk = metric("val_qwk", "qwk", default=float("-inf"))
    loss = metric("val_loss", "loss", default=float("inf"))
    return accuracy, qwk, -loss


def is_better_validation_checkpoint(
    candidate: Mapping[str, Any],
    current_best: Mapping[str, Any] | None,
) -> bool:
    """Apply the fixed KE-V2 checkpoint tie-break deterministically."""
    if current_best is None:
        return True
    return validation_checkpoint_key(candidate) > validation_checkpoint_key(current_best)


@contextmanager
def _isolated_torch_seed(seed: int, uses_cuda: bool):
    """Use a deterministic reset seed without consuming the caller's RNG."""
    cuda_devices = list(range(torch.cuda.device_count())) if uses_cuda else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        if uses_cuda:
            torch.cuda.manual_seed_all(int(seed))
        yield


@torch.no_grad()
def reset_reset_hypothesis(
    model: nn.Module,
    masks: Mapping[str, torch.Tensor],
    *,
    seed: int,
) -> dict[str, Any]:
    """Keep mask=1 weights, randomly reinitialize mask=0 weights.

    BatchNorm is not targeted.  Every Conv2d/Linear bias is restored after
    ``reset_parameters``; in particular, every classifier bias remains in the
    fit hypothesis.
    """
    validate_kels_masks(model, masks)
    targets = dict(iter_kels_targets(model))
    uses_cuda = any(module.weight.is_cuda for module in targets.values())

    fit_parameters = 0
    reset_parameters = 0
    reset_changed_parameters = 0
    preserved_bias_tensors = 0
    max_fit_abs_difference = 0.0

    with _isolated_torch_seed(seed, uses_cuda):
        for name, module in targets.items():
            mask = masks[name].to(device=module.weight.device, non_blocking=True)
            old_weight = module.weight.detach().clone()
            old_bias = (
                module.bias.detach().clone()
                if module.bias is not None
                else None
            )

            module.reset_parameters()
            random_weight = module.weight.detach().clone()
            module.weight.copy_(torch.where(mask, old_weight, random_weight))
            if old_bias is not None:
                module.bias.copy_(old_bias)
                preserved_bias_tensors += 1

            fit_count = int(mask.sum().item())
            reset_count = int(mask.numel() - fit_count)
            fit_parameters += fit_count
            reset_parameters += reset_count
            if fit_count:
                fit_difference = (
                    module.weight[mask] - old_weight[mask]
                ).abs().max().item()
                max_fit_abs_difference = max(
                    max_fit_abs_difference,
                    float(fit_difference),
                )
            if reset_count:
                reset_changed_parameters += int(
                    torch.count_nonzero(
                        module.weight[~mask] != old_weight[~mask]
                    ).item()
                )

    if max_fit_abs_difference != 0.0:
        raise RuntimeError(
            "FIT weights changed during a generation reset; "
            f"max_abs_difference={max_fit_abs_difference}."
        )

    return {
        "reset_seed": int(seed),
        "target_layer_count": len(targets),
        "fit_parameters_preserved": fit_parameters,
        "reset_parameters_requested": reset_parameters,
        "reset_parameters_changed": reset_changed_parameters,
        "reset_changed_fraction": (
            reset_changed_parameters / reset_parameters
            if reset_parameters
            else 0.0
        ),
        "preserved_bias_tensors": preserved_bias_tensors,
        "max_fit_abs_difference": max_fit_abs_difference,
        "batch_norm_policy": "fully_inherited",
        "classifier_bias_policy": "fully_inherited",
    }


@torch.no_grad()
def reset_reset_hypothesis_from_fresh_model(
    model: nn.Module,
    fresh_model: nn.Module,
    masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Copy RESET weights from a normally initialized, non-pretrained model.

    Only Conv2d/Linear weights are mixed. Biases and every BatchNorm tensor in
    ``model`` remain untouched. The fresh model must have exactly the same KELS
    target names and weight shapes.
    """
    validate_kels_masks(model, masks)
    validate_kels_masks(fresh_model, masks)
    targets = dict(iter_kels_targets(model))
    fresh_targets = dict(iter_kels_targets(fresh_model))

    fit_parameters = 0
    reset_parameters = 0
    reset_changed_parameters = 0
    preserved_bias_tensors = 0
    max_fit_abs_difference = 0.0

    for name, module in targets.items():
        fresh_module = fresh_targets[name]
        mask = masks[name].to(device=module.weight.device, non_blocking=True)
        old_weight = module.weight.detach().clone()
        old_bias = module.bias.detach().clone() if module.bias is not None else None
        fresh_weight = fresh_module.weight.detach().to(
            device=module.weight.device,
            dtype=module.weight.dtype,
            non_blocking=True,
        )

        module.weight.copy_(torch.where(mask, old_weight, fresh_weight))

        fit_count = int(mask.sum().item())
        reset_count = int(mask.numel() - fit_count)
        fit_parameters += fit_count
        reset_parameters += reset_count
        if fit_count:
            fit_difference = (module.weight[mask] - old_weight[mask]).abs().max().item()
            max_fit_abs_difference = max(max_fit_abs_difference, float(fit_difference))
        if reset_count:
            reset_changed_parameters += int(
                torch.count_nonzero(module.weight[~mask] != old_weight[~mask]).item()
            )
        if old_bias is not None:
            if not torch.equal(module.bias.detach(), old_bias):
                raise RuntimeError(f"Bias changed during fresh-model reset for {name!r}.")
            preserved_bias_tensors += 1

    if max_fit_abs_difference != 0.0:
        raise RuntimeError(
            "FIT weights changed during a fresh-model generation reset; "
            f"max_abs_difference={max_fit_abs_difference}."
        )

    return {
        "reset_source": "fresh_random_model_pretrained_false",
        "target_layer_count": len(targets),
        "fit_parameters_preserved": fit_parameters,
        "reset_parameters_requested": reset_parameters,
        "reset_parameters_changed": reset_changed_parameters,
        "reset_changed_fraction": (
            reset_changed_parameters / reset_parameters if reset_parameters else 0.0
        ),
        "preserved_bias_tensors": preserved_bias_tensors,
        "max_fit_abs_difference": max_fit_abs_difference,
        "batch_norm_policy": "fully_inherited",
        "classifier_bias_policy": "fully_inherited",
    }
