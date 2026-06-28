"""MambaOut backbones built from timm."""

from __future__ import annotations

import types
from typing import Iterable, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import BACKBONES


_DEFAULT_MAMBAOUT_MODEL = "mambaout_tiny.in1k"
_FALLBACK_SIZE_PRIORITY = ("tiny", "femto", "kobe")
_VALID_ABLATION_MODES = {
    None,
    "gate_fixed",
    "dwconv_center3",
    "dwconv_identity",
    "gate_fixed_dwconv_identity",
}


def _normalize_ablation_mode(mode: Optional[str]) -> Optional[str]:
    if mode in (None, "", "none", "None"):
        return None
    if mode not in _VALID_ABLATION_MODES:
        raise ValueError(
            f"Unsupported MambaOut ablation_mode: {mode!r}. "
            f"Available modes: {sorted(m for m in _VALID_ABLATION_MODES if m is not None)}"
        )
    return str(mode)


def _is_missing_model_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    missing_markers = (
        "unknown model",
        "is not a registered model",
        "not a valid model",
        "model not found",
        "no model named",
    )
    return any(marker in msg for marker in missing_markers)


def _candidate_names_for_size(size: str, models: Iterable[str]) -> list[str]:
    plain = f"mambaout_{size}"
    candidates = [name for name in models if name == plain or name.startswith(f"{plain}.")]
    return sorted(candidates, key=lambda name: (not name.endswith(".in1k"), len(name), name))


def _resolve_fallback_model_name() -> tuple[str, list[str]]:
    available = sorted(timm.list_models("*mambaout*", pretrained=True))
    if not available:
        available = sorted(timm.list_models("*mambaout*"))

    for size in _FALLBACK_SIZE_PRIORITY:
        candidates = _candidate_names_for_size(size, available)
        if candidates:
            return candidates[0], available

    raise ValueError(
        "No tiny/femto/kobe MambaOut model is available in timm. "
        f"Available MambaOut models: {available}"
    )


def _is_mambaout_gated_block(module: nn.Module) -> bool:
    required_attrs = ("norm", "fc1", "fc2", "conv", "split_indices", "act", "ls", "drop_path")
    if not all(hasattr(module, attr) for attr in required_attrs):
        return False
    if not isinstance(getattr(module, "conv"), nn.Conv2d):
        return False
    try:
        split_indices = tuple(getattr(module, "split_indices"))
    except TypeError:
        return False
    return len(split_indices) == 3


def _make_ablation_forward(ablation_mode: str):
    skip_gate = ablation_mode in {"gate_fixed", "gate_fixed_dwconv_identity"}
    skip_dwconv = ablation_mode in {"dwconv_identity", "gate_fixed_dwconv_identity"}
    center3_dwconv = ablation_mode == "dwconv_center3"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)
        g, i, c = torch.split(x, self.split_indices, dim=-1)

        if not skip_dwconv:
            c = c.permute(0, 3, 1, 2)
            if center3_dwconv:
                weight = self.conv.weight
                kernel_h, kernel_w = weight.shape[-2:]
                if kernel_h < 3 or kernel_w < 3:
                    raise RuntimeError(
                        "dwconv_center3 requires a depthwise kernel of at least 3x3, "
                        f"got {kernel_h}x{kernel_w}."
                    )
                mask = torch.zeros_like(weight)
                start_h = (kernel_h - 3) // 2
                start_w = (kernel_w - 3) // 2
                mask[:, :, start_h : start_h + 3, start_w : start_w + 3] = 1.0
                c = F.conv2d(
                    c,
                    weight * mask,
                    bias=self.conv.bias,
                    stride=self.conv.stride,
                    padding=self.conv.padding,
                    dilation=self.conv.dilation,
                    groups=self.conv.groups,
                )
            else:
                c = self.conv(c)
            c = c.permute(0, 2, 3, 1)

        feat = torch.cat((i, c), dim=-1)
        if skip_gate:
            x = self.fc2(feat)
        else:
            x = self.fc2(self.act(g) * feat)
        x = self.ls(x)
        x = self.drop_path(x)
        return x + shortcut

    return forward


def _apply_mambaout_ablation(model: nn.Module, ablation_mode: Optional[str]) -> int:
    mode = _normalize_ablation_mode(ablation_mode)
    if mode is None:
        return 0

    patched_blocks = 0
    ablation_forward = _make_ablation_forward(mode)
    for _, module in model.named_modules():
        if _is_mambaout_gated_block(module):
            module._original_forward = module.forward
            module.forward = types.MethodType(ablation_forward, module)
            patched_blocks += 1

    if patched_blocks == 0:
        raise RuntimeError(
            f"MambaOut ablation_mode={mode!r} was requested, "
            "but no compatible GatedConvBlock was found."
        )
    return patched_blocks


@BACKBONES.register("mambaout_tiny_ce")
class MambaOutTinyCEBackbone(nn.Module):
    """MambaOut-Tiny CE baseline wrapper.

    This wrapper intentionally keeps timm's classifier head inside the
    backbone so it can call timm.create_model(..., num_classes=4) directly.
    Pair it with an identity head in the experiment config.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MAMBAOUT_MODEL,
        pretrained: bool = True,
        num_classes: int = 4,
        ablation_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.requested_model_name = str(model_name)
        self.actual_model_name = self.requested_model_name
        self.available_mambaout_models: list[str] = []
        self.num_classes = int(num_classes)
        self.ablation_mode = _normalize_ablation_mode(ablation_mode)

        try:
            self.model = timm.create_model(
                self.requested_model_name,
                pretrained=pretrained,
                num_classes=self.num_classes,
                **kwargs,
            )
        except Exception as exc:
            if not _is_missing_model_error(exc):
                raise
            fallback_name, available = _resolve_fallback_model_name()
            self.actual_model_name = fallback_name
            self.available_mambaout_models = available
            self.model = timm.create_model(
                fallback_name,
                pretrained=pretrained,
                num_classes=self.num_classes,
                **kwargs,
            )

        self.patched_ablation_blocks = _apply_mambaout_ablation(self.model, self.ablation_mode)
        self.out_features = self.num_classes
        print(
            "[MambaOut] "
            f"requested_model_name={self.requested_model_name} "
            f"actual_model_name={self.actual_model_name} "
            f"ablation_mode={self.ablation_mode} "
            f"patched_blocks={self.patched_ablation_blocks}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
