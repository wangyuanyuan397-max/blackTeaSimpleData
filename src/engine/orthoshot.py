"""Training-only Ortho-Shot adaptations for supervised MixNet-S.

The module deliberately does not alter the inference graph.  It provides:

* branch-aware MixNet-S pointwise/kernel/DBT regularization;
* SelfMix, CutMix and MixUp candidates;
* per-sample MaxUp over a configurable candidate pool.

The few-shot support/query/task protocol from Ortho-Shot is intentionally not
implemented because this project uses fixed-class supervised classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


VALID_ROLES = ("expansion", "projection", "depthwise")
VALID_ORTH_METHODS = {"dbt", "kernel"}
VALID_AUGMENTATIONS = {
    "none",
    "normal",
    "selfmix",
    "cutmix",
    "mixup",
    "random_choice",
}
VALID_CANDIDATES = {
    "normal",
    "selfmix",
    "cutmix",
    "mixup",
    "rotation",
    "horizontal_flip",
}
IMAGENET_NORMALIZED_BLACK = (
    -0.485 / 0.229,
    -0.456 / 0.224,
    -0.406 / 0.225,
)


@dataclass(frozen=True)
class MixNetOrthTarget:
    name: str
    role: str
    stage_index: int
    block_index: int
    branch_index: int
    conv: nn.Conv2d


@dataclass
class AugCandidate:
    name: str
    images: torch.Tensor
    target_a: torch.Tensor
    target_b: torch.Tensor | None = None
    lam: float = 1.0


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"train.orthoshot.{name} must be a mapping.")
    return dict(value)


def _unwrap_mixnet(model: nn.Module) -> nn.Module:
    top = model.module if hasattr(model, "module") else model
    backbone = getattr(top, "backbone", top)
    raw = getattr(backbone, "model", backbone)
    architecture = (getattr(raw, "default_cfg", {}) or {}).get("architecture")
    if architecture != "mixnet_s" or not hasattr(raw, "blocks"):
        raise ValueError("Ortho-Shot DBT currently requires official timm mixnet_s.")
    return raw


def _leaf_convs(module: nn.Module | None) -> list[nn.Conv2d]:
    if module is None:
        return []
    if isinstance(module, nn.Conv2d):
        return [module]
    return [child for child in module.modules() if isinstance(child, nn.Conv2d)]


def discover_mixnet_orth_targets(model: nn.Module) -> list[MixNetOrthTarget]:
    """Discover only inverted-residual feature convolutions, never stem/SE/FC."""
    raw = _unwrap_mixnet(model)
    targets: list[MixNetOrthTarget] = []
    role_attributes = (
        ("expansion", "conv_pw"),
        ("projection", "conv_pwl"),
        ("depthwise", "conv_dw"),
    )
    for stage_index, stage in enumerate(raw.blocks):
        for block_index, block in enumerate(stage):
            # The stem-adjacent DepthwiseSeparableConv has different semantics:
            # its conv_pw is a projection, not an MBConv expansion.  Excluding
            # that block keeps the target definition unambiguous.
            if block.__class__.__name__ != "InvertedResidual":
                continue
            for role, attribute in role_attributes:
                branches = _leaf_convs(getattr(block, attribute, None))
                for branch_index, conv in enumerate(branches):
                    kernel_size = tuple(int(value) for value in conv.kernel_size)
                    if role in {"expansion", "projection"}:
                        if kernel_size != (1, 1) or int(conv.groups) != 1:
                            raise ValueError(
                                f"{role} target must be an ungrouped 1x1 Conv2d: "
                                f"S{stage_index}B{block_index} branch {branch_index}."
                            )
                    else:
                        if not (
                            int(conv.groups) == int(conv.in_channels)
                            and int(conv.groups) == int(conv.out_channels)
                            and int(conv.weight.shape[1]) == 1
                        ):
                            raise ValueError(
                                f"depthwise target is not depthwise: "
                                f"S{stage_index}B{block_index} branch {branch_index}."
                            )
                    targets.append(
                        MixNetOrthTarget(
                            name=(
                                f"blocks.{stage_index}.{block_index}.{attribute}."
                                f"branch{branch_index}"
                            ),
                            role=role,
                            stage_index=stage_index,
                            block_index=block_index,
                            branch_index=branch_index,
                            conv=conv,
                        )
                    )
    if not targets:
        raise ValueError("No MixNet-S orthogonal-regularization targets were found.")
    return targets


def matrix_orthogonal_loss(weight: torch.Tensor) -> torch.Tensor:
    """Mean-normalized feasible row/column orthogonality penalty."""
    matrix = weight.float().flatten(1)
    out_dim, in_dim = matrix.shape
    if out_dim <= in_dim:
        gram = matrix @ matrix.t()
        size = out_dim
    else:
        gram = matrix.t() @ matrix
        size = in_dim
    identity = torch.eye(size, device=gram.device, dtype=gram.dtype)
    return (gram - identity).square().mean()


def depthwise_dbt_loss(conv: nn.Conv2d) -> torch.Tensor:
    """DBT-style spatial autocorrelation penalty for a depthwise branch.

    This is the grouped depthwise analogue of the official OCNN
    ``deconv_orth_dist`` construction: kernel self-convolution should equal a
    center impulse.  It does not compare kernels from unrelated input channels.
    """
    weight = conv.weight.float()
    channels = int(weight.shape[0])
    kernel_as_input = weight.squeeze(1).unsqueeze(0)
    output = F.conv2d(
        kernel_as_input,
        weight,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=channels,
    )
    target = torch.zeros_like(output)
    center_y = int(output.shape[-2] // 2)
    center_x = int(output.shape[-1] // 2)
    target[0, :, center_y, center_x] = 1.0
    return (output - target).square().mean()


@torch.no_grad()
def mean_absolute_filter_cosine(targets: Sequence[MixNetOrthTarget]) -> float:
    per_filter_scores: list[torch.Tensor] = []
    for target in targets:
        matrix = target.conv.weight.detach().float().flatten(1)
        if int(matrix.shape[0]) <= 1:
            continue
        normalized = F.normalize(matrix, p=2, dim=1, eps=1e-12)
        gram = torch.abs(normalized @ normalized.t())
        gram.fill_diagonal_(0.0)
        per_filter_scores.append(gram.sum(dim=1) / float(matrix.shape[0] - 1))
    if not per_filter_scores:
        return 0.0
    return float(torch.cat(per_filter_scores).mean().item())


class MixNetOrthogonalRegularizer:
    """Branch-aware training regularizer for the official timm MixNet-S."""

    def __init__(self, model: nn.Module, config: Mapping[str, Any]) -> None:
        cfg = dict(config)
        unknown = sorted(
            set(cfg)
            - {
                "enabled",
                "method",
                "lambda",
                "warmup_epochs",
                "scope",
                "reduction",
            }
        )
        if unknown:
            raise ValueError(f"Unknown orthogonal options: {unknown}")
        self.enabled = bool(cfg.get("enabled", False))
        self.method = str(cfg.get("method", "dbt")).lower()
        self.max_lambda = float(cfg.get("lambda", 0.001))
        self.warmup_epochs = int(cfg.get("warmup_epochs", 0))
        self.reduction = str(cfg.get("reduction", "mean")).lower()
        scope = _as_mapping(cfg.get("scope"), "orthogonal.scope")
        self.scope = {
            "expansion": bool(scope.get("expansion", False)),
            "projection": bool(scope.get("projection", True)),
            "depthwise": bool(scope.get("depthwise", False)),
        }
        if not self.enabled:
            raise ValueError("MixNetOrthogonalRegularizer requires enabled=true.")
        if self.method not in VALID_ORTH_METHODS:
            raise ValueError(f"orthogonal.method must be one of {sorted(VALID_ORTH_METHODS)}")
        if self.max_lambda < 0.0:
            raise ValueError("orthogonal.lambda must be non-negative.")
        if self.warmup_epochs < 0:
            raise ValueError("orthogonal.warmup_epochs must be non-negative.")
        if self.reduction != "mean":
            raise ValueError("Only mean-normalized orthogonal reduction is supported.")
        if not any(self.scope.values()):
            raise ValueError("At least one orthogonal scope must be enabled.")

        self.all_targets = discover_mixnet_orth_targets(model)
        self.targets = [target for target in self.all_targets if self.scope[target.role]]
        if not self.targets:
            raise ValueError("The requested orthogonal scope selected no layers.")

    def effective_lambda(self, epoch: int) -> float:
        if self.warmup_epochs <= 0 or int(epoch) >= self.warmup_epochs:
            return self.max_lambda
        return self.max_lambda * float(int(epoch) + 1) / float(self.warmup_epochs)

    def __call__(self) -> tuple[torch.Tensor, dict[str, float]]:
        role_losses: dict[str, list[torch.Tensor]] = {role: [] for role in VALID_ROLES}
        all_losses: list[torch.Tensor] = []
        for target in self.targets:
            if target.role == "depthwise" and self.method == "dbt":
                loss = depthwise_dbt_loss(target.conv)
            else:
                loss = matrix_orthogonal_loss(target.conv.weight)
            role_losses[target.role].append(loss)
            all_losses.append(loss)
        total = torch.stack(all_losses).mean()
        stats = {
            f"orthogonal_{role}_loss": float(torch.stack(losses).mean().detach().item())
            for role, losses in role_losses.items()
            if losses
        }
        return total, stats

    def epoch_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for role in ("expansion", "projection"):
            targets = [target for target in self.all_targets if target.role == role]
            metrics[f"mean_{role}_filter_cosine"] = mean_absolute_filter_cosine(targets)
        return metrics

    def summary(self) -> dict[str, Any]:
        counts = {
            role: sum(1 for target in self.targets if target.role == role)
            for role in VALID_ROLES
        }
        all_counts = {
            role: sum(1 for target in self.all_targets if target.role == role)
            for role in VALID_ROLES
        }
        return {
            "enabled": True,
            "method": self.method,
            "lambda": self.max_lambda,
            "warmup_epochs": self.warmup_epochs,
            "scope": dict(self.scope),
            "target_layer_counts": counts,
            "available_layer_counts": all_counts,
            "target_layer_count": len(self.targets),
            "target_layers": [
                {
                    "name": target.name,
                    "role": target.role,
                    "weight_shape": list(target.conv.weight.shape),
                    "kernel_size": list(target.conv.kernel_size),
                    "stride": list(target.conv.stride),
                    "groups": int(target.conv.groups),
                }
                for target in self.targets
            ],
            **self.epoch_metrics(),
        }


def selfmix_batch(
    images: torch.Tensor,
    probability: float = 0.5,
    patch_ratio: float = 0.5,
) -> torch.Tensor:
    """Copy a same-sized source region within each selected image."""
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("SelfMix probability must be in [0, 1].")
    if not 0.0 < float(patch_ratio) < 1.0:
        raise ValueError("SelfMix patch_ratio must be in (0, 1).")
    output = images.clone()
    _, _, height, width = images.shape
    patch_h = max(1, min(height - 1, int(round(height * float(patch_ratio)))))
    patch_w = max(1, min(width - 1, int(round(width * float(patch_ratio)))))
    max_y = height - patch_h
    max_x = width - patch_w
    for index in range(int(images.shape[0])):
        if float(torch.rand((), device=images.device).item()) >= float(probability):
            continue
        src_y = int(torch.randint(max_y + 1, (1,), device=images.device).item())
        src_x = int(torch.randint(max_x + 1, (1,), device=images.device).item())
        dst_y = int(torch.randint(max_y + 1, (1,), device=images.device).item())
        dst_x = int(torch.randint(max_x + 1, (1,), device=images.device).item())
        if dst_y == src_y and dst_x == src_x:
            dst_x = (dst_x + 1) % (max_x + 1) if max_x > 0 else dst_x
            if dst_x == src_x and max_y > 0:
                dst_y = (dst_y + 1) % (max_y + 1)
        patch = images[
            index,
            :,
            src_y : src_y + patch_h,
            src_x : src_x + patch_w,
        ].clone()
        output[
            index,
            :,
            dst_y : dst_y + patch_h,
            dst_x : dst_x + patch_w,
        ] = patch
    return output


def _sample_beta(alpha: float, device: torch.device) -> float:
    if float(alpha) <= 0.0:
        return 1.0
    concentration = torch.tensor(float(alpha), device=device)
    return float(torch.distributions.Beta(concentration, concentration).sample().item())


def _rand_bbox(images: torch.Tensor, lam: float) -> tuple[int, int, int, int]:
    height = int(images.shape[-2])
    width = int(images.shape[-1])
    ratio = float(max(0.0, 1.0 - lam) ** 0.5)
    cut_h = int(height * ratio)
    cut_w = int(width * ratio)
    center_y = int(torch.randint(height, (1,), device=images.device).item())
    center_x = int(torch.randint(width, (1,), device=images.device).item())
    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + cut_h // 2, height)
    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    return x1, y1, x2, y2


def cutmix_candidate(images: torch.Tensor, labels: torch.Tensor, alpha: float) -> AugCandidate:
    permutation = torch.randperm(images.shape[0], device=images.device)
    lam = _sample_beta(alpha, images.device)
    x1, y1, x2, y2 = _rand_bbox(images, lam)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    area = float((x2 - x1) * (y2 - y1))
    lam = 1.0 - area / float(images.shape[-1] * images.shape[-2])
    return AugCandidate("cutmix", mixed, labels, labels[permutation], lam)


def mixup_candidate(images: torch.Tensor, labels: torch.Tensor, alpha: float) -> AugCandidate:
    permutation = torch.randperm(images.shape[0], device=images.device)
    lam = _sample_beta(alpha, images.device)
    mixed = images.mul(lam).add(images[permutation], alpha=1.0 - lam)
    return AugCandidate("mixup", mixed, labels, labels[permutation], lam)


class OrthoShotObjective:
    """Optional classification objective; absent configs use the old Trainer path."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Mapping[str, Any],
        logger: Any = None,
    ) -> None:
        cfg = dict(config)
        unknown = sorted(set(cfg) - {"enabled", "orthogonal", "augmentation", "maxup"})
        if unknown:
            raise ValueError(f"Unknown train.orthoshot options: {unknown}")
        if cfg.get("enabled", True) is False:
            raise ValueError("OrthoShotObjective requires enabled=true.")

        criterion = getattr(loss_fn, "criterion", loss_fn)
        if not isinstance(criterion, nn.CrossEntropyLoss):
            raise ValueError(
                "Ortho-Shot augmentation/MaxUp currently requires cross_entropy loss."
            )
        self.criterion = criterion
        self.augmentation = _as_mapping(cfg.get("augmentation"), "augmentation")
        self.maxup = _as_mapping(cfg.get("maxup"), "maxup")
        orthogonal = _as_mapping(cfg.get("orthogonal"), "orthogonal")
        self.regularizer = (
            MixNetOrthogonalRegularizer(model, orthogonal)
            if bool(orthogonal.get("enabled", False))
            else None
        )
        self._validate_augmentation_config()
        self._validate_maxup_config()

        if logger:
            logger.info(
                "orthoshot_objective_initialized",
                augmentation_mode=self.augmentation_mode,
                maxup_candidates=list(self.maxup_candidates),
                orthogonal=(self.regularizer.summary() if self.regularizer else None),
            )

    @property
    def augmentation_mode(self) -> str:
        return str(self.augmentation.get("mode", "none")).lower()

    @property
    def maxup_enabled(self) -> bool:
        return bool(self.maxup.get("enabled", False))

    @property
    def maxup_candidates(self) -> tuple[str, ...]:
        if not self.maxup_enabled:
            return ()
        return tuple(str(value).lower() for value in self.maxup.get("candidates", ()))

    def _validate_augmentation_config(self) -> None:
        allowed = {
            "mode",
            "probability",
            "choices",
            "selfmix_probability",
            "selfmix_patch_ratio",
            "cutmix_alpha",
            "mixup_alpha",
            "rotation_degrees",
        }
        unknown = sorted(set(self.augmentation) - allowed)
        if unknown:
            raise ValueError(f"Unknown orthoshot augmentation options: {unknown}")
        if self.augmentation_mode not in VALID_AUGMENTATIONS:
            raise ValueError(
                f"augmentation.mode must be one of {sorted(VALID_AUGMENTATIONS)}"
            )
        probability = float(self.augmentation.get("probability", 1.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError("augmentation.probability must be in [0, 1].")
        selfmix_probability = float(
            self.augmentation.get("selfmix_probability", 0.5)
        )
        if not 0.0 <= selfmix_probability <= 1.0:
            raise ValueError("augmentation.selfmix_probability must be in [0, 1].")
        selfmix_patch_ratio = float(
            self.augmentation.get("selfmix_patch_ratio", 0.5)
        )
        if not 0.0 < selfmix_patch_ratio < 1.0:
            raise ValueError("augmentation.selfmix_patch_ratio must be in (0, 1).")
        for option_name, default in (("cutmix_alpha", 1.0), ("mixup_alpha", 0.2)):
            if float(self.augmentation.get(option_name, default)) < 0.0:
                raise ValueError(f"augmentation.{option_name} must be non-negative.")
        if float(self.augmentation.get("rotation_degrees", 15.0)) < 0.0:
            raise ValueError("augmentation.rotation_degrees must be non-negative.")
        choices = tuple(
            str(value).lower() for value in self.augmentation.get("choices", ())
        )
        if self.augmentation_mode == "random_choice":
            if not choices or any(choice not in VALID_CANDIDATES - {"normal"} for choice in choices):
                raise ValueError("random_choice requires valid non-normal choices.")
            if len(set(choices)) != len(choices):
                raise ValueError("random_choice choices must be unique.")

    def _validate_maxup_config(self) -> None:
        allowed = {"enabled", "candidates"}
        unknown = sorted(set(self.maxup) - allowed)
        if unknown:
            raise ValueError(f"Unknown maxup options: {unknown}")
        if not self.maxup_enabled:
            return
        candidates = self.maxup_candidates
        if len(candidates) < 2:
            raise ValueError("MaxUp requires at least two candidates.")
        if len(set(candidates)) != len(candidates):
            raise ValueError("MaxUp candidates must be unique.")
        invalid = sorted(set(candidates) - VALID_CANDIDATES)
        if invalid:
            raise ValueError(f"Unknown MaxUp candidates: {invalid}")

    def _per_sample_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.criterion.weight,
            ignore_index=self.criterion.ignore_index,
            reduction="none",
            label_smoothing=float(getattr(self.criterion, "label_smoothing", 0.0)),
        )

    def _candidate_loss(self, logits: torch.Tensor, candidate: AugCandidate) -> torch.Tensor:
        loss_a = self._per_sample_ce(logits, candidate.target_a)
        if candidate.target_b is None:
            return loss_a
        loss_b = self._per_sample_ce(logits, candidate.target_b)
        return loss_a * float(candidate.lam) + loss_b * (1.0 - float(candidate.lam))

    def _build_candidate(
        self,
        name: str,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> AugCandidate:
        name = str(name).lower()
        if name == "normal":
            return AugCandidate(name, images, labels)
        if name == "selfmix":
            return AugCandidate(
                name,
                selfmix_batch(
                    images,
                    probability=float(self.augmentation.get("selfmix_probability", 0.5)),
                    patch_ratio=float(self.augmentation.get("selfmix_patch_ratio", 0.5)),
                ),
                labels,
            )
        if name == "cutmix":
            return cutmix_candidate(
                images,
                labels,
                alpha=float(self.augmentation.get("cutmix_alpha", 1.0)),
            )
        if name == "mixup":
            return mixup_candidate(
                images,
                labels,
                alpha=float(self.augmentation.get("mixup_alpha", 0.2)),
            )
        if name == "horizontal_flip":
            return AugCandidate(name, torch.flip(images, dims=(-1,)), labels)
        if name == "rotation":
            degrees = abs(float(self.augmentation.get("rotation_degrees", 15.0)))
            rotated = []
            for image in images:
                angle = float(
                    torch.empty((), device=images.device).uniform_(-degrees, degrees).item()
                )
                rotated.append(
                    TF.rotate(
                        image,
                        angle=angle,
                        interpolation=InterpolationMode.BILINEAR,
                        # Inputs have already been ImageNet-normalized. These
                        # values reproduce a black pre-normalization fill,
                        # matching the image-level RandomRotation candidate.
                        fill=list(IMAGENET_NORMALIZED_BLACK),
                    )
                )
            return AugCandidate(name, torch.stack(rotated, dim=0), labels)
        raise ValueError(f"Unsupported Ortho-Shot candidate: {name}")

    def _build_single_candidate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> AugCandidate:
        mode = self.augmentation_mode
        if mode in {"none", "normal"}:
            return self._build_candidate("normal", images, labels)
        probability = float(self.augmentation.get("probability", 1.0))
        if float(torch.rand((), device=images.device).item()) >= probability:
            return self._build_candidate("normal", images, labels)
        if mode == "random_choice":
            choices = tuple(
                str(value).lower() for value in self.augmentation.get("choices", ())
            )
            choice_index = int(torch.randint(len(choices), (1,), device=images.device).item())
            mode = choices[choice_index]
        return self._build_candidate(mode, images, labels)

    def compute(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        diagnostics: dict[str, float] = {}
        if self.maxup_enabled:
            candidates = [
                self._build_candidate(name, images, labels)
                for name in self.maxup_candidates
            ]
            combined = torch.cat([candidate.images for candidate in candidates], dim=0)
            combined_logits = model(combined)
            if not torch.is_tensor(combined_logits) or combined_logits.ndim != 2:
                raise TypeError("Ortho-Shot MaxUp requires 2D classification logits.")
            logits_by_candidate = combined_logits.split(int(images.shape[0]), dim=0)
            losses = torch.stack(
                [
                    self._candidate_loss(logits, candidate)
                    for logits, candidate in zip(logits_by_candidate, candidates)
                ],
                dim=0,
            )
            worst_loss, worst_index = losses.max(dim=0)
            classification_loss = worst_loss.mean()
            normal_index = next(
                (index for index, candidate in enumerate(candidates) if candidate.name == "normal"),
                None,
            )
            if normal_index is not None:
                metric_logits = logits_by_candidate[normal_index]
            else:
                stacked_logits = torch.stack(list(logits_by_candidate), dim=0)
                gather_index = worst_index.view(1, -1, 1).expand(
                    1, -1, stacked_logits.shape[-1]
                )
                metric_logits = stacked_logits.gather(0, gather_index).squeeze(0)
            for index, candidate in enumerate(candidates):
                diagnostics[f"maxup_win_{candidate.name}"] = float(
                    (worst_index == index).float().mean().detach().item()
                )
        else:
            candidate = self._build_single_candidate(images, labels)
            metric_logits = model(candidate.images)
            if not torch.is_tensor(metric_logits) or metric_logits.ndim != 2:
                raise TypeError("Ortho-Shot requires 2D classification logits.")
            classification_loss = self._candidate_loss(metric_logits, candidate).mean()
            diagnostics[f"augmentation_{candidate.name}"] = 1.0

        if self.regularizer is None:
            orthogonal_loss = classification_loss.new_zeros(())
            effective_lambda = 0.0
            role_stats: dict[str, float] = {}
        else:
            orthogonal_loss, role_stats = self.regularizer()
            effective_lambda = self.regularizer.effective_lambda(epoch)
        weighted_orthogonal = orthogonal_loss * float(effective_lambda)
        total_loss = classification_loss + weighted_orthogonal
        diagnostics.update(role_stats)
        diagnostics.update(
            {
                "ce_loss": float(classification_loss.detach().item()),
                "orthogonal_loss": float(orthogonal_loss.detach().item()),
                "weighted_orthogonal_loss": float(weighted_orthogonal.detach().item()),
                "orthogonal_lambda": float(effective_lambda),
                "orthogonal_ce_ratio": float(
                    weighted_orthogonal.detach().div(
                        classification_loss.detach().clamp_min(1e-12)
                    ).item()
                ),
            }
        )
        return total_loss, metric_logits, diagnostics

    def epoch_metrics(self) -> dict[str, float]:
        return self.regularizer.epoch_metrics() if self.regularizer else {}

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "augmentation": dict(self.augmentation),
            "maxup": {
                "enabled": self.maxup_enabled,
                "candidates": list(self.maxup_candidates),
            },
            "orthogonal": self.regularizer.summary() if self.regularizer else None,
        }
