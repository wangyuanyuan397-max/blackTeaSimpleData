"""EfficientNetV2-S 的轻量 Global-Local spatial attention pooling 实验。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from ...utils.registry import BACKBONES


class FermentationSensitiveRegionPooling(nn.Module):
    """在最后一层 feature map 上生成 spatial-softmax 注意力，并做加权池化。"""

    def __init__(self, in_channels: int, reduction: int = 8, temperature: float = 1.0) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels 必须大于 0。")
        if reduction <= 0:
            raise ValueError("reduction 必须大于 0。")
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0。")

        hidden_channels = max(int(in_channels) // int(reduction), 32)
        self.temperature = float(temperature)
        self.attention_generator = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回局部加权特征 [B, C] 和注意力图 [B, 1, H, W]。"""
        if feature_map.ndim != 4:
            raise ValueError(f"期望输入为 [B, C, H, W]，实际为 {tuple(feature_map.shape)}。")

        batch_size, channels, height, width = feature_map.shape
        attention_logits = self.attention_generator(feature_map)
        attention = attention_logits.flatten(2)
        attention = F.softmax(attention / self.temperature, dim=-1)
        attention = attention.view(batch_size, 1, height, width)
        local_feature = (feature_map * attention).sum(dim=(2, 3))
        return local_feature, attention


@BACKBONES.register("efficientnet_v2_s_region_attention")
class EfficientNetV2SRegionAttentionBackbone(nn.Module):
    """EfficientNetV2-S + Global/Local attention pooling 的可控消融 backbone。

    pooling_mode:
    - global：只用 GAP，全局 baseline；
    - local：只用 spatial attention pooling 得到的局部特征；
    - global_local：使用 g + gamma * l，gamma 初始化为 0。
    """

    def __init__(
        self,
        pretrained: bool = True,
        pooling_mode: str = "global_local",
        reduction: int = 8,
        temperature: float = 1.0,
        local_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        pooling_mode = str(pooling_mode).lower()
        if pooling_mode not in {"global", "local", "global_local"}:
            raise ValueError("pooling_mode 只能是 global、local 或 global_local。")

        weights = EfficientNet_V2_S_Weights.DEFAULT if bool(pretrained) else None
        backbone = efficientnet_v2_s(weights=weights)
        self.features = backbone.features
        self.out_features = int(backbone.classifier[1].in_features)
        self.pooling_mode = pooling_mode
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = FermentationSensitiveRegionPooling(
            in_channels=self.out_features,
            reduction=int(reduction),
            temperature=float(temperature),
        )
        self.local_scale = nn.Parameter(torch.tensor(float(local_scale_init), dtype=torch.float32))
        self.last_attention_map: torch.Tensor | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """输出给现有 linear head 使用的 [B, out_features] 特征向量。"""
        feature_map = self.features(images)
        global_feature = self.global_pool(feature_map).flatten(1)

        if self.pooling_mode == "global":
            self.last_attention_map = None
            return global_feature

        local_feature, attention = self.local_pool(feature_map)
        self.last_attention_map = attention.detach()

        if self.pooling_mode == "local":
            return local_feature

        return global_feature + self.local_scale * local_feature
