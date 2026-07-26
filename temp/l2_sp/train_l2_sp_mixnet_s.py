"""Run the fixed MixNet-S baseline with L2-SP regularization.

This file intentionally lives under temp/ and monkey-patches only the batch
entrypoint's Trainer symbol. The project src/ code and baseline YAML files are
left unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_TRAIN_BATCH_PATH = PROJECT_ROOT / "tools" / "train_batch.py"
L2SP_MODEL_CONFIG = Path("temp/l2_sp/fixed_timm_mixnet_s_l2sp_alpha001.yaml")
COMMON_CONFIG = Path("configs/fixed_split_01234_grid30_408_train.yaml")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_train_batch_module():
    spec = importlib.util.spec_from_file_location("train_batch_l2sp_base", BASE_TRAIN_BATCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base batch trainer: {BASE_TRAIN_BATCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_batch_l2sp_base"] = module
    spec.loader.exec_module(module)
    return module


train_batch_base = _load_train_batch_module()


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return vars(value) if hasattr(value, "__dict__") else {}


def _matches_any_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


class L2SPTrainer(train_batch_base.Trainer):
    """Trainer subclass that adds alpha * ||theta - theta_0||^2 to train loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_sp_cfg = self._load_l2_sp_config()
        self.l2_sp_enabled = bool(self.l2_sp_cfg.get("enabled", False))
        self.l2_sp_alpha = float(self.l2_sp_cfg.get("alpha", 0.0) or 0.0)
        self.l2_sp_reference: dict[str, torch.Tensor] = {}
        self.l2_sp_parameter_count = 0
        self.last_classification_loss = 0.0
        self.last_l2_sp_loss = 0.0

        if self.l2_sp_enabled and self.l2_sp_alpha > 0:
            self.refresh_l2_sp_reference()
        elif self.l2_sp_enabled and self.logger:
            self.logger.warning("l2_sp_disabled", reason="alpha_is_not_positive")

    def _load_l2_sp_config(self) -> dict[str, Any]:
        train_cfg = getattr(self.config, "train", None)
        cfg = None
        if isinstance(train_cfg, dict):
            cfg = train_cfg.get("l2_sp")
        elif train_cfg is not None and hasattr(train_cfg, "l2_sp"):
            cfg = getattr(train_cfg, "l2_sp")
        return _as_plain_dict(cfg)

    def _should_constrain_parameter(self, name: str, param: torch.nn.Parameter) -> bool:
        include_prefixes = list(self.l2_sp_cfg.get("include_prefixes") or ["backbone."])
        exclude_prefixes = list(
            self.l2_sp_cfg.get("exclude_prefixes")
            or ["head.", "aux_head.", "classifier."]
        )
        trainable_only = bool(self.l2_sp_cfg.get("trainable_only", False))

        if trainable_only and not param.requires_grad:
            return False
        if include_prefixes and not _matches_any_prefix(name, include_prefixes):
            return False
        if exclude_prefixes and _matches_any_prefix(name, exclude_prefixes):
            return False
        return True

    def refresh_l2_sp_reference(self) -> None:
        """Snapshot the starting point after pretrained weights are loaded."""
        reference = {}
        parameter_count = 0
        for name, param in self.model.named_parameters():
            if not self._should_constrain_parameter(name, param):
                continue
            reference[name] = param.detach().clone()
            parameter_count += param.numel()

        if not reference:
            raise ValueError(
                "L2-SP is enabled, but no parameters matched include/exclude prefixes. "
                "Check train.l2_sp.include_prefixes in the temp YAML."
            )

        self.l2_sp_reference = reference
        self.l2_sp_parameter_count = parameter_count
        if self.logger:
            self.logger.info(
                "l2_sp_reference_created",
                alpha=self.l2_sp_alpha,
                normalize=str(self.l2_sp_cfg.get("normalize", "mean")),
                tensors=len(reference),
                parameters=parameter_count,
                include_prefixes=list(self.l2_sp_cfg.get("include_prefixes") or ["backbone."]),
                exclude_prefixes=list(
                    self.l2_sp_cfg.get("exclude_prefixes")
                    or ["head.", "aux_head.", "classifier."]
                ),
            )

    def _l2_sp_penalty(self) -> torch.Tensor:
        penalty = next(self.model.parameters()).new_zeros(())
        parameter_count = 0
        for name, param in self.model.named_parameters():
            reference_param = self.l2_sp_reference.get(name)
            if reference_param is None:
                continue
            if reference_param.device != param.device or reference_param.dtype != param.dtype:
                reference_param = reference_param.to(device=param.device, dtype=param.dtype)
                self.l2_sp_reference[name] = reference_param
            penalty = penalty + (param - reference_param).pow(2).sum()
            parameter_count += param.numel()

        normalize = str(self.l2_sp_cfg.get("normalize", "mean")).lower()
        if normalize == "sum":
            return penalty
        if normalize != "mean":
            raise ValueError("train.l2_sp.normalize must be either 'mean' or 'sum'.")
        return penalty / max(1, parameter_count)

    def _compute_loss(self, outputs, labels, extra_targets=None):
        classification_loss = super()._compute_loss(outputs, labels, extra_targets=extra_targets)
        self.last_classification_loss = float(classification_loss.detach().item())

        if not self.l2_sp_enabled or self.l2_sp_alpha <= 0:
            self.last_l2_sp_loss = 0.0
            return classification_loss

        sp_loss = self._l2_sp_penalty()
        self.last_l2_sp_loss = float(sp_loss.detach().item())
        return classification_loss + self.l2_sp_alpha * sp_loss


def main() -> None:
    train_batch_base.Trainer = L2SPTrainer
    train_batch_base.COMMON_CONFIG = COMMON_CONFIG
    train_batch_base.CONFIG_LIST = (L2SP_MODEL_CONFIG,)
    train_batch_base.PYCHARM_DEVICE = "auto"
    train_batch_base.PYCHARM_DRY_RUN = False
    train_batch_base.PYCHARM_FAIL_FAST = False
    train_batch_base.PYCHARM_KEEP_PTH_FILES = False
    train_batch_base.main()


if __name__ == "__main__":
    main()
