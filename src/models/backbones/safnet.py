"""SAFNet backbone adapted from the authors' reference implementation.

Architecture:
    SCAM-ResNet50 -> class projection -> AMSAFF -> class projection

AMSAFF intentionally operates on a ``[B, num_classes, 1, 1]`` tensor. This
matches both the paper diagram and the supplied reference source, even though
the spatial multi-scale branches consequently operate on a 1x1 feature map.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torchvision import models

from ...utils.registry import BACKBONES


class ConvolutionalChannelAttention(nn.Module):
    """Channel attention used after the first two SCAM convolutions."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = channels // reduction
        if hidden_channels < 1:
            raise ValueError(
                f"CCA requires channels >= reduction, got channels={channels}, "
                f"reduction={reduction}."
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(x)
        weights = self.fc1(weights)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = self.sigmoid(weights)
        return x * weights


class SqueezeExcitation(nn.Module):
    """SE attention used after the third SCAM convolution."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = channels // reduction
        if hidden_channels < 1:
            raise ValueError(
                f"SE requires channels >= reduction, got channels={channels}, "
                f"reduction={reduction}."
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        weights = self.avg_pool(x).reshape(batch_size, channels)
        weights = self.fc(weights).reshape(batch_size, channels, 1, 1)
        return x * weights


class SCAMBottleneck(nn.Module):
    """ResNet bottleneck with CCA after conv1/conv2 and SE after conv3."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        width_per_group: int = 64,
        reduction: int = 16,
    ):
        super().__init__()
        width = int(channels * (width_per_group / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.cca1 = ConvolutionalChannelAttention(width, reduction=reduction)

        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.cca2 = ConvolutionalChannelAttention(width, reduction=reduction)

        out_channels = channels * self.expansion
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels, reduction=reduction)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.cca1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.cca2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out = out + identity
        return self.relu(out)


class SCAMResNet50(nn.Module):
    """ResNet50 topology whose bottlenecks are replaced by SCAM blocks."""

    def __init__(
        self,
        num_classes: int,
        groups: int = 1,
        width_per_group: int = 64,
        reduction: int = 16,
    ):
        super().__init__()
        self.in_channels = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.reduction = reduction

        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=3)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * SCAMBottleneck.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        out_channels = channels * SCAMBottleneck.expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers: List[nn.Module] = [
            SCAMBottleneck(
                self.in_channels,
                channels,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                width_per_group=self.width_per_group,
                reduction=self.reduction,
            )
        ]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                SCAMBottleneck(
                    self.in_channels,
                    channels,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                    reduction=self.reduction,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class AdaptiveScaleWeight(nn.Module):
    """Generate two sample-wise weights for the AMSAFF branches."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.softmax(self.fc2(x))


class AMSAFFAttention(nn.Module):
    """Attention block from the supplied AMSAFF implementation."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_context = self.avg_pool(x)
        local_weights = self.sigmoid(self.conv(x))
        return local_weights * global_context


class AMSAFF(nn.Module):
    """Adaptive Multi-Scale Attention Feature Fusion module."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.scale_weight = AdaptiveScaleWeight(channels)
        self.attention = AMSAFFAttention(channels)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.conv1x1(x)
        branch2 = self.conv3x3(x)

        weights = self.scale_weight(x)
        weight1 = weights[:, 0].reshape(-1, 1, 1, 1)
        weight2 = weights[:, 1].reshape(-1, 1, 1, 1)
        fused = weight1 * branch1 + weight2 * branch2

        fused = self.attention(fused)
        fused = self.fuse(fused)
        return x + fused


@BACKBONES.register("safnet")
class SAFNetBackbone(nn.Module):
    """Paper-faithful SAFNet classifier exposed as a project backbone.

    The classifier is contained in this module, so experiment configs must use
    the project's identity head. ``pretrained=False`` reproduces the supplied
    source initialization. When enabled, compatible torchvision ResNet50
    parameters are loaded and the added CCA/SE/AMSAFF layers remain randomly
    initialized.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = False,
        reduction: int = 16,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)
        self.scam_resnet = SCAMResNet50(
            num_classes=self.num_classes,
            reduction=int(reduction),
        )
        self.amsaff = AMSAFF(channels=self.num_classes)
        self.classifier = nn.Linear(self.num_classes, self.num_classes)
        self.out_features = self.num_classes

        self.pretrained_missing_keys: list[str] = []
        self.pretrained_unexpected_keys: list[str] = []
        if self.pretrained:
            self._load_imagenet_resnet50_weights()

        init_mode = "imagenet_partial" if self.pretrained else "scratch"
        print(f"[SAFNet] num_classes={self.num_classes} initialization={init_mode}")

    def _load_imagenet_resnet50_weights(self) -> None:
        source = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        state_dict = source.state_dict()
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
        incompatible = self.scam_resnet.load_state_dict(state_dict, strict=False)
        self.pretrained_missing_keys = list(incompatible.missing_keys)
        self.pretrained_unexpected_keys = list(incompatible.unexpected_keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scam_resnet(x)
        x = x.reshape(x.shape[0], self.num_classes, 1, 1)
        x = self.amsaff(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
