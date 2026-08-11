"""Helpers for tensor batches that may contain nested multi-view inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def move_to_device(value: Any, device: torch.device | str) -> Any:
    """Recursively move tensors in a nested batch to ``device``."""

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return type(value)((key, move_to_device(item, device)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def batch_size_of(value: Any) -> int:
    """Infer the leading batch dimension from a tensor or nested input."""

    if torch.is_tensor(value):
        if value.ndim == 0:
            raise ValueError("Cannot infer batch size from a scalar tensor.")
        return int(value.size(0))
    if isinstance(value, Mapping):
        preferred = value.get("global")
        if preferred is not None:
            return batch_size_of(preferred)
        for item in value.values():
            try:
                return batch_size_of(item)
            except (TypeError, ValueError):
                continue
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return batch_size_of(item)
            except (TypeError, ValueError):
                continue
    raise TypeError(f"Cannot infer batch size from {type(value).__name__}.")
