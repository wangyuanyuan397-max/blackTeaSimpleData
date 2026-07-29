"""AD-GBC feature-cleaner backbone wrappers."""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import BACKBONES


def _normalize_positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return parsed


class ADGBCModule(nn.Module):
    """Anisotropic dynamic granular-ball clustering as a feature cleaner."""

    def __init__(
        self,
        in_channels: int,
        num_balls: int = 16,
        projection_dim: int | None = None,
        tau: float = 1.0,
        eps: float = 1e-6,
        normalize_region_desc: bool = True,
        use_refine: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = _normalize_positive_int(in_channels, "in_channels")
        self.num_balls = _normalize_positive_int(num_balls, "num_balls")
        self.projection_dim = (
            _normalize_positive_int(projection_dim, "projection_dim")
            if projection_dim is not None
            else self.in_channels
        )
        self.tau = float(tau)
        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {tau!r}.")
        self.eps = float(eps)
        self.normalize_region_desc = bool(normalize_region_desc)

        # 先把 backbone 输出的特征投影到 AD-GBC 使用的维度，降低聚类计算开销。
        self.proj = (
            nn.Conv2d(self.in_channels, self.projection_dim, kernel_size=1)
            if self.in_channels != self.projection_dim
            else nn.Identity()
        )
        self.proj_back = (
            nn.Conv2d(self.projection_dim, self.in_channels, kernel_size=1)
            if self.in_channels != self.projection_dim
            else nn.Identity()
        )
        # centers/log_scales 分别表示 K 个弹性区域的中心和各向异性尺度。
        self.centers = nn.Parameter(torch.randn(self.num_balls, self.projection_dim) * 0.1)
        self.log_scales = nn.Parameter(torch.zeros(self.num_balls, self.projection_dim))
        self.refine = (
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True),
            )
            if use_refine
            else nn.Identity()
        )

        self._last_alpha: torch.Tensor | None = None
        self._last_z_flat: torch.Tensor | None = None

    def get_scales(self) -> torch.Tensor:
        return F.softplus(self.log_scales) + self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ADGBCModule expects [B,C,H,W], got {tuple(x.shape)}.")
        batch_size, _, height, width = x.shape
        z = self.proj(x)
        z_flat = z.flatten(2).transpose(1, 2)

        # 将每个空间位置软分配到 K 个弹性区域；tau 越小，分配越尖锐。
        centers = self.centers
        scales = self.get_scales()
        diff = z_flat.unsqueeze(2) - centers.view(1, 1, self.num_balls, self.projection_dim)
        normalized = diff / scales.view(1, 1, self.num_balls, self.projection_dim)
        dist = normalized.square().sum(dim=-1)
        alpha = F.softmax(-dist / self.tau, dim=-1)

        # 先聚合每个区域的描述子，再广播回每个空间位置形成净化后的特征。
        region_desc = torch.einsum("bnk,bnd->bkd", alpha, z_flat)
        if self.normalize_region_desc:
            region_desc = region_desc / alpha.sum(dim=1).clamp_min(self.eps).unsqueeze(-1)
        refined_z = torch.einsum("bnk,bkd->bnd", alpha, region_desc)

        refined = refined_z.transpose(1, 2).reshape(batch_size, self.projection_dim, height, width)
        refined = self.proj_back(refined)
        self._last_alpha = alpha
        self._last_z_flat = z_flat
        return self.refine(x + refined)

    def wasserstein_diversity_loss(self) -> torch.Tensor:
        # 多样性损失约束弹性区域不要塌缩到同一处。
        centers = self.centers
        centered = centers - centers.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1).matmul(centered) / float(self.num_balls)
        covariance = covariance + torch.eye(
            covariance.size(0),
            device=covariance.device,
            dtype=covariance.dtype,
        ) * self.eps
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(self.eps)
        trace_sqrt = eigenvalues.sqrt().sum()
        return (
            centers.mean(dim=0).square().sum()
            + torch.trace(covariance)
            - (2.0 / (self.projection_dim ** 0.5)) * trace_sqrt
        )

    def scale_consistency_loss(self) -> torch.Tensor:
        if self._last_alpha is None or self._last_z_flat is None:
            return self.centers.new_zeros(())

        # 尺度一致性让可学习尺度贴近当前 batch 中实际聚到该区域的特征方差。
        alpha = self._last_alpha
        z_flat = self._last_z_flat
        alpha_sum = alpha.sum(dim=1).clamp_min(self.eps)
        observed_mean = torch.einsum("bnk,bnd->bkd", alpha, z_flat) / alpha_sum.unsqueeze(-1)
        centered = z_flat.unsqueeze(2) - observed_mean.unsqueeze(1)
        observed_var = torch.einsum("bnk,bnkd->bkd", alpha, centered.square()) / alpha_sum.unsqueeze(-1)
        observed_scale = observed_var.clamp_min(self.eps).sqrt().mean(dim=0)
        predicted_scale = self.get_scales()
        return F.smooth_l1_loss(
            torch.log(predicted_scale),
            torch.log(observed_scale.detach()),
        )

    def auxiliary_outputs(self) -> dict[str, torch.Tensor]:
        loss_w = self.wasserstein_diversity_loss()
        loss_scale = self.scale_consistency_loss()
        with torch.no_grad():
            if self._last_alpha is None:
                entropy = loss_w.new_zeros(())
            else:
                entropy = -(
                    self._last_alpha.clamp_min(self.eps).log() * self._last_alpha
                ).sum(dim=-1).mean()
            scales = self.get_scales()
            center_norm = self.centers.norm(dim=1).mean()
            scale_mean = scales.mean()
            scale_std = scales.std(unbiased=False)
        return {
            "adgbc_loss_w_div": loss_w,
            "adgbc_loss_scale_con": loss_scale,
            "adgbc_assignment_entropy": entropy,
            "adgbc_center_norm_mean": center_norm,
            "adgbc_scale_mean": scale_mean,
            "adgbc_scale_std": scale_std,
        }


@BACKBONES.register("timm_adgbc")
class TimmADGBCBackbone(nn.Module):
    """timm feature-map backbone with an AD-GBC cleaner before global pooling."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        input_size: int = 408,
        adgbc_k: int = 16,
        adgbc_dim: int | None = 256,
        adgbc_tau: float = 1.0,
        freeze_base: bool = False,
        freeze_adgbc: bool = False,
        normalize_region_desc: bool = True,
        use_refine: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        feature_channels = self._infer_feature_channels(int(input_size))
        # AD-GBC 净化器插在 timm backbone 的 feature map 和全局池化之间。
        self.cleaner = ADGBCModule(
            feature_channels,
            num_balls=int(adgbc_k),
            projection_dim=adgbc_dim,
            tau=float(adgbc_tau),
            normalize_region_desc=normalize_region_desc,
            use_refine=use_refine,
        )
        self.out_features = int(feature_channels)
        # staged_training 会把 plugin_modules 视为可单独训练的新增模块。
        self.plugin_modules = nn.ModuleList([self.cleaner])
        self.freeze_base = bool(freeze_base)
        self.freeze_adgbc = bool(freeze_adgbc)
        self._apply_trainability()

    def _apply_trainability(self) -> None:
        if self.freeze_base:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        if self.freeze_adgbc:
            for parameter in self.cleaner.parameters():
                parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self._extract_feature_map(x)
        cleaned = self.cleaner(feature_map)
        return torch.flatten(F.adaptive_avg_pool2d(cleaned, 1), 1)

    def get_auxiliary_outputs(self) -> dict[str, torch.Tensor]:
        return self.cleaner.auxiliary_outputs()

    def _extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(x)
        else:
            features = self.model(x)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.ndim == 4:
            num_features = getattr(self.model, "num_features", None)
            if num_features is not None and features.shape[-1] == int(num_features):
                features = features.permute(0, 3, 1, 2).contiguous()
            return features
        if features.ndim == 3:
            # Token features: use the token axis as a 1D spatial map.
            return features.transpose(1, 2).unsqueeze(-1).contiguous()
        raise ValueError(
            "timm_adgbc requires a spatial or token feature tensor from forward_features, "
            f"got shape {tuple(features.shape)}."
        )

    def _infer_feature_channels(self, input_size: int) -> int:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, input_size, input_size)
                features = self._extract_feature_map(dummy)
        finally:
            self.model.train(was_training)
        if features.ndim != 4 or features.shape[1] <= 0:
            raise ValueError(f"Invalid AD-GBC feature map shape: {tuple(features.shape)}")
        return int(features.shape[1])
