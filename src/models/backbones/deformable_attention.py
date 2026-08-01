"""Single-scale 2D deformable self-attention for CNN feature maps."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def choose_num_heads(channels: int, requested_heads: int) -> int:
    """Return the largest head count not exceeding requested_heads that divides channels."""
    for num_heads in range(min(channels, requested_heads), 0, -1):
        if channels % num_heads == 0:
            return num_heads
    return 1


class DeformableAttention2d(nn.Module):
    """Single-scale deformable self-attention with [B, C, H, W] input/output."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_points: int = 4,
        max_offset: float = 2.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        layer_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if num_points <= 0:
            raise ValueError("num_points must be positive.")
        if max_offset <= 0:
            raise ValueError("max_offset must be positive.")

        self.channels = int(channels)
        self.num_heads = choose_num_heads(int(channels), int(num_heads))
        self.num_points = int(num_points)
        self.head_dim = self.channels // self.num_heads
        self.max_offset = float(max_offset)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.channels)
        self.value_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.offset_proj = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                groups=self.channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.channels,
                self.num_heads * self.num_points * 2,
                kernel_size=1,
            ),
        )
        self.weight_proj = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                groups=self.channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.channels,
                self.num_heads * self.num_points,
                kernel_size=1,
            ),
        )
        self.output_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.projection_dropout = nn.Dropout2d(float(projection_dropout))
        self.layer_scale = nn.Parameter(
            torch.full((1, self.channels, 1, 1), float(layer_scale_init))
        )

        self.last_offset_mean: Tensor | None = None
        self.last_offset_max: Tensor | None = None
        self.last_attention_entropy: Tensor | None = None
        self.last_layer_scale_mean: Tensor | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        offset_output = self.offset_proj[-1]
        weight_output = self.weight_proj[-1]
        nn.init.zeros_(offset_output.weight)
        nn.init.zeros_(weight_output.weight)
        nn.init.zeros_(weight_output.bias)

        angles = (
            torch.arange(self.num_heads, dtype=torch.float32)
            * 2.0
            * math.pi
            / float(self.num_heads)
        )
        directions = torch.stack([angles.cos(), angles.sin()], dim=-1)
        directions = directions / directions.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        directions = directions[:, None, :].repeat(1, self.num_points, 1)
        radius = torch.linspace(0.5, 1.0, self.num_points, dtype=torch.float32).view(
            1,
            self.num_points,
            1,
        )
        initial_offsets = directions * radius * self.max_offset
        with torch.no_grad():
            offset_output.bias.copy_(initial_offsets.reshape(-1))

    @staticmethod
    def make_base_grid(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        x = 2.0 * (x + 0.5) / float(width) - 1.0
        y = 2.0 * (y + 0.5) / float(height) - 1.0
        return torch.stack([x, y], dim=-1).view(1, 1, 1, height, width, 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}.")

        batch_size, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")

        residual = x
        normalized = self.norm(x)
        value = self.value_proj(normalized)
        value = value.view(batch_size, self.num_heads, self.head_dim, height, width)
        value = value.reshape(batch_size * self.num_heads, self.head_dim, height, width)

        offsets = self.offset_proj(normalized)
        offsets = offsets.view(
            batch_size,
            self.num_heads,
            self.num_points,
            2,
            height,
            width,
        )
        offsets = offsets.permute(0, 1, 2, 4, 5, 3).contiguous()
        offsets = torch.tanh(offsets / self.max_offset) * self.max_offset

        offsets_x = offsets[..., 0] * 2.0 / float(width)
        offsets_y = offsets[..., 1] * 2.0 / float(height)
        normalized_offsets = torch.stack([offsets_x, offsets_y], dim=-1)
        base_grid = self.make_base_grid(
            height=height,
            width=width,
            device=x.device,
            dtype=x.dtype,
        )
        sampling_grid = base_grid + normalized_offsets
        sampling_grid = sampling_grid.permute(0, 1, 3, 4, 2, 5).contiguous()
        sampling_grid = sampling_grid.view(
            batch_size * self.num_heads,
            height,
            width * self.num_points,
            2,
        )

        sampled = F.grid_sample(
            input=value,
            grid=sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.view(
            batch_size,
            self.num_heads,
            self.head_dim,
            height,
            width,
            self.num_points,
        )

        weights = self.weight_proj(normalized)
        weights = weights.view(batch_size, self.num_heads, self.num_points, height, width)
        weights = F.softmax(weights, dim=2)

        self.last_offset_mean = offsets.detach().square().sum(dim=-1).sqrt().mean()
        self.last_offset_max = offsets.detach().abs().max()
        self.last_attention_entropy = (
            -weights.detach().clamp_min(1e-8).log().mul(weights.detach()).sum(dim=2).mean()
        )
        self.last_layer_scale_mean = self.layer_scale.detach().mean()

        weights = self.attention_dropout(weights)
        weights = weights.permute(0, 1, 3, 4, 2).unsqueeze(2)
        attended = (sampled * weights).sum(dim=-1)
        attended = attended.reshape(batch_size, channels, height, width)
        attended = self.output_proj(attended)
        attended = self.projection_dropout(attended)
        return residual + self.layer_scale * attended
