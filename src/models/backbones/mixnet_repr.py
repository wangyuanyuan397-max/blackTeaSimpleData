"""MixNet-S trained with cyclic RePr on residual projection filters.

RePr is a training-time controller: the timm MixNet-S topology is left
unchanged, selected projection channels are temporarily masked after their
BatchNorm, and the selected filters are later reintroduced in directions that
are orthogonal to the active filters. Final inference uses the ordinary full
network with every mask restored to one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import BACKBONES


VALID_SCOPES = {"mid", "late", "all_residual"}


@dataclass(frozen=True)
class MixNetRePrConfig:
    enabled: bool = True
    scope: str = "all_residual"
    prune_ratio: float = 0.2
    full_epochs: int = 10
    sparse_epochs: int = 5
    cycles: int = 3
    per_layer_max_ratio: float = 0.4
    reinit_scale: float = 0.1

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "MixNetRePrConfig":
        values = dict(config or {})
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown mixnet_repr options: {unknown}")
        parsed = cls(**values)
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError(
                "mixnet_s_repr requires repr.enabled=true; use the timm backbone "
                "for the baseline."
            )
        if str(self.scope).lower() not in VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(VALID_SCOPES)}")
        if not 0.0 < float(self.prune_ratio) < 0.5:
            raise ValueError("prune_ratio must be in (0, 0.5).")
        if int(self.full_epochs) <= 0 or int(self.sparse_epochs) <= 0:
            raise ValueError("full_epochs and sparse_epochs must be positive.")
        if int(self.cycles) <= 0:
            raise ValueError("cycles must be positive.")
        if not float(self.prune_ratio) <= float(self.per_layer_max_ratio) < 1.0:
            raise ValueError(
                "per_layer_max_ratio must be at least prune_ratio and below 1."
            )
        if not 0.0 < float(self.reinit_scale) <= 1.0:
            raise ValueError("reinit_scale must be in (0, 1].")

    @property
    def final_restore_epoch(self) -> int:
        """Zero-based epoch whose training phase ends with the final restore."""

        return int(self.cycles) * (int(self.full_epochs) + int(self.sparse_epochs)) - 1

    @property
    def prune_epochs(self) -> tuple[int, ...]:
        cycle_length = int(self.full_epochs) + int(self.sparse_epochs)
        return tuple(
            int(self.full_epochs) + cycle * cycle_length
            for cycle in range(int(self.cycles))
        )

    @property
    def restore_epochs(self) -> tuple[int, ...]:
        return tuple(
            prune_epoch + int(self.sparse_epochs) - 1
            for prune_epoch in self.prune_epochs
        )


@dataclass
class ProjectionBranch:
    block_name: str
    stage_index: int
    block_index: int
    branch_index: int
    conv: nn.Conv2d
    bn: nn.Module
    bn_offset: int
    block_channels: int
    mask_buffer_name: str

    @property
    def name(self) -> str:
        return f"{self.block_name}.branch{self.branch_index}"


def _conv_branches(module: nn.Module) -> list[nn.Conv2d]:
    if isinstance(module, nn.Conv2d):
        branches = [module]
    elif hasattr(module, "values"):
        branches = list(module.values())
    else:
        branches = list(getattr(module, "branches", ()))
    if not branches or not all(isinstance(branch, nn.Conv2d) for branch in branches):
        raise TypeError(f"Unsupported MixNet projection module: {module.__class__.__name__}")
    return branches


@torch.no_grad()
def _filter_redundancy_scores(conv: nn.Conv2d) -> torch.Tensor:
    flat = conv.weight.detach().flatten(1)
    normalized = F.normalize(flat, p=2, dim=1, eps=1e-12)
    gram = torch.abs(normalized @ normalized.t())
    gram.fill_diagonal_(0.0)
    return gram.mean(dim=1)


@torch.no_grad()
def summarize_mixnet_projection_redundancy(
    backbone_or_model: nn.Module,
) -> dict[str, Any]:
    """Measure residual PW-projection overlap for baseline and RePr models."""
    model = getattr(backbone_or_model, "model", backbone_or_model)
    architecture = (getattr(model, "default_cfg", {}) or {}).get("architecture")
    if architecture != "mixnet_s" or not hasattr(model, "blocks"):
        raise ValueError("Projection redundancy summary requires timm mixnet_s.")

    scope_scores: dict[str, list[torch.Tensor]] = {
        "mid": [],
        "late": [],
        "all_residual": [],
    }
    scope_blocks: dict[str, dict[str, dict[str, Any]]] = {
        scope: {} for scope in scope_scores
    }
    for stage_index, stage in enumerate(model.blocks):
        for block_index, block in enumerate(stage):
            if not bool(getattr(block, "has_skip", False)):
                continue
            projection = getattr(block, "conv_pwl", None)
            if projection is None or getattr(block, "bn3", None) is None:
                continue
            branches = _conv_branches(projection)
            branch_channels = [int(branch.out_channels) for branch in branches]
            block_name = f"S{stage_index}B{block_index}"
            scopes = ["all_residual"]
            if stage_index == 4:
                scopes.append("mid")
            if stage_index == 5:
                scopes.append("late")
            branch_scores = [_filter_redundancy_scores(branch) for branch in branches]
            for scope in scopes:
                scope_scores[scope].extend(branch_scores)
                scope_blocks[scope][block_name] = {
                    "stage_index": stage_index,
                    "block_index": block_index,
                    "channels": sum(branch_channels),
                    "branch_channels": branch_channels,
                    "branch_count": len(branches),
                }

    summary: dict[str, Any] = {}
    for scope, scores in scope_scores.items():
        if not scores:
            raise ValueError(f"MixNet-S redundancy scope {scope!r} has no targets.")
        blocks = scope_blocks[scope]
        summary[scope] = {
            "mean_filter_redundancy": float(torch.cat(scores).mean().item()),
            "target_block_count": len(blocks),
            "target_branch_count": sum(
                int(values["branch_count"]) for values in blocks.values()
            ),
            "target_filter_count": sum(
                int(values["channels"]) for values in blocks.values()
            ),
            "applied_blocks": blocks,
        }
    return summary


@BACKBONES.register("mixnet_s_repr")
class MixNetSRePrBackbone(nn.Module):
    """Official timm MixNet-S with a residual PW-projection RePr controller."""

    def __init__(
        self,
        pretrained: bool = True,
        input_size: int = 408,
        model_name: str = "mixnet_s",
        repr: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if model_name != "mixnet_s":
            raise ValueError("mixnet_s_repr fixes model_name to mixnet_s.")
        self.input_size = int(input_size)
        self.model_name = model_name
        self.repr_config = MixNetRePrConfig.from_mapping(repr)
        self.model = timm.create_model(
            model_name,
            pretrained=bool(pretrained),
            num_classes=0,
            **kwargs,
        )
        self.out_features = int(getattr(self.model, "num_features", 0))
        if self.out_features <= 0:
            raise ValueError("timm MixNet-S did not expose a valid num_features value.")

        self.projection_branches: list[ProjectionBranch] = []
        self.applied_blocks: dict[str, dict[str, Any]] = {}
        self._mask_hook_handles: list[Any] = []
        self._selected_by_branch: dict[str, list[int]] = {}
        self._selected_bn_by_block: dict[str, list[int]] = {}
        self._sparse_active = False
        self._active_cycle = 0
        self._latest_metrics: dict[str, float | int | bool] = {}
        self._cycle_history: list[dict[str, Any]] = []
        self._current_cycle_record: dict[str, Any] | None = None
        self._discover_projection_targets()

    def _scope_matches(self, stage_index: int) -> bool:
        scope = str(self.repr_config.scope).lower()
        if scope == "mid":
            return stage_index == 4
        if scope == "late":
            return stage_index == 5
        return True

    def _discover_projection_targets(self) -> None:
        for stage_index, stage in enumerate(self.model.blocks):
            for block_index, block in enumerate(stage):
                if not bool(getattr(block, "has_skip", False)):
                    continue
                if not self._scope_matches(stage_index):
                    continue
                projection = getattr(block, "conv_pwl", None)
                bn = getattr(block, "bn3", None)
                if projection is None or bn is None:
                    # MixNet's stem-adjacent DepthwiseSeparableConv can use a
                    # residual skip but has no inverted-residual PW projection.
                    # RePr-PWProj intentionally leaves it untouched.
                    continue
                branches = _conv_branches(projection)
                branch_channels = [int(branch.out_channels) for branch in branches]
                block_channels = sum(branch_channels)
                bn_channels = int(getattr(bn, "num_features", 0))
                if bn_channels != block_channels:
                    raise ValueError(
                        f"S{stage_index}B{block_index} projection channels={block_channels}, "
                        f"but bn3 channels={bn_channels}."
                    )
                if any(tuple(branch.kernel_size) != (1, 1) for branch in branches):
                    raise ValueError("RePr-PWProj only supports 1x1 projection branches.")
                block_name = f"S{stage_index}B{block_index}"
                mask_buffer_name = f"_repr_mask_s{stage_index}_b{block_index}"
                self.register_buffer(
                    mask_buffer_name,
                    torch.ones(block_channels, dtype=branches[0].weight.dtype),
                    persistent=True,
                )
                self._mask_hook_handles.append(
                    bn.register_forward_hook(self._make_mask_hook(mask_buffer_name))
                )
                offset = 0
                for branch_index, branch in enumerate(branches):
                    self.projection_branches.append(
                        ProjectionBranch(
                            block_name=block_name,
                            stage_index=stage_index,
                            block_index=block_index,
                            branch_index=branch_index,
                            conv=branch,
                            bn=bn,
                            bn_offset=offset,
                            block_channels=block_channels,
                            mask_buffer_name=mask_buffer_name,
                        )
                    )
                    offset += int(branch.out_channels)
                self.applied_blocks[block_name] = {
                    "stage_index": stage_index,
                    "block_index": block_index,
                    "channels": block_channels,
                    "branch_channels": branch_channels,
                    "branch_count": len(branches),
                }

        if not self.projection_branches:
            raise ValueError(
                f"RePr scope={self.repr_config.scope!r} selected no residual projections."
            )

    def _make_mask_hook(self, buffer_name: str):
        def apply_mask(module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor):
            del module, inputs
            if not torch.is_tensor(output) or output.ndim != 4:
                raise TypeError("RePr projection BN hook expects a BCHW tensor.")
            if not self._sparse_active:
                return output
            mask = getattr(self, buffer_name).to(device=output.device, dtype=output.dtype)
            return output * mask.view(1, -1, 1, 1)

        return apply_mask

    @staticmethod
    @torch.no_grad()
    def _branch_scores(conv: nn.Conv2d) -> torch.Tensor:
        return _filter_redundancy_scores(conv)

    @torch.no_grad()
    def mean_filter_redundancy(self) -> float:
        scores = [self._branch_scores(branch.conv) for branch in self.projection_branches]
        if not scores:
            return 0.0
        return float(torch.cat(scores).mean().item())

    @torch.no_grad()
    def _select_redundant_filters(self) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        candidates: list[tuple[float, ProjectionBranch, int, int]] = []
        block_totals = {
            block_name: int(values["channels"])
            for block_name, values in self.applied_blocks.items()
        }
        for branch in self.projection_branches:
            scores = self._branch_scores(branch.conv)
            for local_index, score in enumerate(scores.tolist()):
                candidates.append(
                    (
                        float(score),
                        branch,
                        int(local_index),
                        int(branch.bn_offset + local_index),
                    )
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        target_count = max(
            1,
            int(len(candidates) * float(self.repr_config.prune_ratio)),
        )
        block_caps = {
            block_name: max(
                1,
                min(
                    channels - 1,
                    int(math.floor(channels * float(self.repr_config.per_layer_max_ratio))),
                ),
            )
            for block_name, channels in block_totals.items()
        }
        block_selected_counts = {block_name: 0 for block_name in block_totals}
        selected_by_branch: dict[str, list[int]] = {}
        selected_bn_by_block: dict[str, list[int]] = {}
        for _, branch, local_index, bn_index in candidates:
            if block_selected_counts[branch.block_name] >= block_caps[branch.block_name]:
                continue
            selected_by_branch.setdefault(branch.name, []).append(local_index)
            selected_bn_by_block.setdefault(branch.block_name, []).append(bn_index)
            block_selected_counts[branch.block_name] += 1
            if sum(len(indices) for indices in selected_by_branch.values()) >= target_count:
                break
        selected_count = sum(len(indices) for indices in selected_by_branch.values())
        if selected_count != target_count:
            raise RuntimeError(
                f"RePr selected {selected_count} filters, expected {target_count}; "
                "per-layer caps may be infeasible."
            )
        return selected_by_branch, selected_bn_by_block

    @torch.no_grad()
    def _apply_masks(self, selected_bn_by_block: Mapping[str, Sequence[int]]) -> None:
        for block_name, block_info in self.applied_blocks.items():
            stage_index = int(block_info["stage_index"])
            block_index = int(block_info["block_index"])
            mask = getattr(self, f"_repr_mask_s{stage_index}_b{block_index}")
            mask.fill_(1.0)
            indices = list(selected_bn_by_block.get(block_name, ()))
            if indices:
                mask[torch.as_tensor(indices, device=mask.device, dtype=torch.long)] = 0.0

    @torch.no_grad()
    def _restore_all_masks(self) -> None:
        for block_info in self.applied_blocks.values():
            mask = getattr(
                self,
                f"_repr_mask_s{int(block_info['stage_index'])}_b{int(block_info['block_index'])}",
            )
            mask.fill_(1.0)

    @staticmethod
    @torch.no_grad()
    def _orthogonal_reinitialize(
        conv: nn.Conv2d,
        indices: Sequence[int],
        scale: float,
    ) -> None:
        if not indices:
            return
        flat = conv.weight.data.flatten(1)
        index_tensor = torch.as_tensor(indices, device=flat.device, dtype=torch.long)
        keep_mask = torch.ones(flat.size(0), dtype=torch.bool, device=flat.device)
        keep_mask[index_tensor] = False
        active = flat[keep_mask]
        if active.numel() == 0:
            raise RuntimeError("RePr cannot reinitialize a branch with no active filters.")

        q_complete, _ = torch.linalg.qr(active.t(), mode="complete")
        rank = int(torch.linalg.matrix_rank(active).item())
        null_basis = q_complete[:, rank:]
        new_count = int(index_tensor.numel())
        if null_basis.size(1) >= new_count:
            new_filters = null_basis[:, :new_count].t()
        else:
            active_basis = q_complete[:, :rank]
            random_filters = torch.randn(
                new_count,
                flat.size(1),
                device=flat.device,
                dtype=flat.dtype,
            )
            random_filters -= (random_filters @ active_basis) @ active_basis.t()
            new_q, _ = torch.linalg.qr(random_filters.t(), mode="reduced")
            new_filters = new_q[:, :new_count].t()

        active_norm = active.norm(dim=1).median().clamp_min(1e-12)
        new_filters = F.normalize(new_filters, p=2, dim=1, eps=1e-12)
        new_filters = new_filters * active_norm * float(scale)
        flat[index_tensor] = new_filters
        conv.weight.data.copy_(flat.view_as(conv.weight.data))
        if conv.bias is not None:
            conv.bias.data[index_tensor] = 0.0

    @staticmethod
    @torch.no_grad()
    def _reset_optimizer_slices(
        optimizer: torch.optim.Optimizer,
        parameter: nn.Parameter | None,
        indices: Sequence[int],
    ) -> None:
        if parameter is None or not indices:
            return
        state = optimizer.state.get(parameter)
        if not state:
            return
        index_tensor = torch.as_tensor(indices, device=parameter.device, dtype=torch.long)
        for value in state.values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[index_tensor] = 0.0

    @torch.no_grad()
    def _reinitialize_selected(self, optimizer: torch.optim.Optimizer) -> None:
        branches_by_name = {branch.name: branch for branch in self.projection_branches}
        for branch_name, indices in self._selected_by_branch.items():
            branch = branches_by_name[branch_name]
            self._orthogonal_reinitialize(
                branch.conv,
                indices,
                scale=float(self.repr_config.reinit_scale),
            )
            self._reset_optimizer_slices(optimizer, branch.conv.weight, indices)
            self._reset_optimizer_slices(optimizer, branch.conv.bias, indices)

        branches_by_block: dict[str, ProjectionBranch] = {}
        for branch in self.projection_branches:
            branches_by_block.setdefault(branch.block_name, branch)
        for block_name, bn_indices in self._selected_bn_by_block.items():
            bn = branches_by_block[block_name].bn
            index_tensor = torch.as_tensor(
                bn_indices,
                device=bn.running_mean.device,
                dtype=torch.long,
            )
            if getattr(bn, "affine", False):
                bn.weight.data[index_tensor] = 1.0
                bn.bias.data[index_tensor] = 0.0
                self._reset_optimizer_slices(optimizer, bn.weight, bn_indices)
                self._reset_optimizer_slices(optimizer, bn.bias, bn_indices)
            if getattr(bn, "track_running_stats", False):
                bn.running_mean.data[index_tensor] = 0.0
                bn.running_var.data[index_tensor] = 1.0

    def training_controller_on_train_begin(
        self,
        *,
        total_epochs: int,
        optimizer: torch.optim.Optimizer,
        logger: Any = None,
    ) -> None:
        del optimizer
        required_epochs = int(self.repr_config.final_restore_epoch) + 1
        if int(total_epochs) < required_epochs:
            raise ValueError(
                f"RePr schedule needs at least {required_epochs} epochs, got {total_epochs}."
            )
        self._restore_all_masks()
        if logger:
            logger.info(
                "repr_controller_initialized",
                scope=str(self.repr_config.scope),
                prune_ratio=float(self.repr_config.prune_ratio),
                full_epochs=int(self.repr_config.full_epochs),
                sparse_epochs=int(self.repr_config.sparse_epochs),
                cycles=int(self.repr_config.cycles),
                target_blocks=len(self.applied_blocks),
                target_filters=sum(int(v["channels"]) for v in self.applied_blocks.values()),
            )

    def training_controller_epoch_start(
        self,
        *,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        logger: Any = None,
    ) -> None:
        del optimizer
        if int(epoch) not in self.repr_config.prune_epochs:
            return
        if self._sparse_active:
            raise RuntimeError("RePr attempted to start a sparse phase while one is active.")
        cycle_index = self.repr_config.prune_epochs.index(int(epoch)) + 1
        pre_redundancy = self.mean_filter_redundancy()
        selected_by_branch, selected_bn_by_block = self._select_redundant_filters()
        self._selected_by_branch = selected_by_branch
        self._selected_bn_by_block = selected_bn_by_block
        self._apply_masks(selected_bn_by_block)
        self._sparse_active = True
        self._active_cycle = cycle_index
        selected_count = sum(len(v) for v in selected_by_branch.values())
        self._current_cycle_record = {
            "cycle": cycle_index,
            "prune_epoch": int(epoch) + 1,
            "restore_epoch": int(self.repr_config.restore_epochs[cycle_index - 1]) + 1,
            "pre_prune_redundancy": pre_redundancy,
            "selected_filter_count": selected_count,
            "selected_ratio": selected_count
            / float(sum(int(v["channels"]) for v in self.applied_blocks.values())),
            "selected_by_branch": {
                name: list(indices) for name, indices in selected_by_branch.items()
            },
            "selected_bn_by_block": {
                name: list(indices) for name, indices in selected_bn_by_block.items()
            },
        }
        if logger:
            logger.info(
                "repr_sparse_phase_started",
                cycle=cycle_index,
                epoch=int(epoch) + 1,
                selected_filters=selected_count,
                pre_prune_redundancy=pre_redundancy,
            )

    def training_controller_epoch_end(
        self,
        *,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        logger: Any = None,
    ) -> None:
        if int(epoch) in self.repr_config.restore_epochs:
            if not self._sparse_active or self._current_cycle_record is None:
                raise RuntimeError("RePr restore epoch reached without an active sparse phase.")
            self._reinitialize_selected(optimizer)
            self._restore_all_masks()
            self._sparse_active = False
            post_reinit_redundancy = self.mean_filter_redundancy()
            self._current_cycle_record["post_reinit_redundancy"] = post_reinit_redundancy
            self._cycle_history.append(self._current_cycle_record)
            if logger:
                logger.info(
                    "repr_sparse_phase_restored",
                    cycle=self._active_cycle,
                    epoch=int(epoch) + 1,
                    post_reinit_redundancy=post_reinit_redundancy,
                )
            self._current_cycle_record = None

        current_redundancy = self.mean_filter_redundancy()
        selected_count = sum(len(v) for v in self._selected_by_branch.values())
        checkpoint_eligible = not self._sparse_active
        early_stopping_eligible = (
            checkpoint_eligible and int(epoch) >= int(self.repr_config.final_restore_epoch)
        )
        self._latest_metrics = {
            "repr_sparse_active": bool(self._sparse_active),
            "repr_cycle": int(self._active_cycle),
            "repr_pruned_filters": int(selected_count if self._sparse_active else 0),
            "repr_mean_filter_redundancy": float(current_redundancy),
            "checkpoint_eligible": bool(checkpoint_eligible),
            "early_stopping_eligible": bool(early_stopping_eligible),
        }
        if self._cycle_history:
            latest_cycle = self._cycle_history[-1]
            self._latest_metrics["repr_last_pre_prune_redundancy"] = float(
                latest_cycle["pre_prune_redundancy"]
            )
            self._latest_metrics["repr_last_post_reinit_redundancy"] = float(
                latest_cycle["post_reinit_redundancy"]
            )

    def training_controller_metrics(self) -> dict[str, float | int | bool]:
        return dict(self._latest_metrics)

    def training_controller_on_train_end(self) -> None:
        self._restore_all_masks()
        self._sparse_active = False

    def repr_summary(self) -> dict[str, Any]:
        redundancy_by_scope = summarize_mixnet_projection_redundancy(self)
        return {
            "model_name": self.model_name,
            "scope": str(self.repr_config.scope).lower(),
            "prune_ratio": float(self.repr_config.prune_ratio),
            "full_epochs": int(self.repr_config.full_epochs),
            "sparse_epochs": int(self.repr_config.sparse_epochs),
            "cycles": int(self.repr_config.cycles),
            "per_layer_max_ratio": float(self.repr_config.per_layer_max_ratio),
            "reinit_scale": float(self.repr_config.reinit_scale),
            "target_block_count": len(self.applied_blocks),
            "target_branch_count": len(self.projection_branches),
            "target_filter_count": sum(
                int(values["channels"]) for values in self.applied_blocks.values()
            ),
            "applied_blocks": self.applied_blocks,
            "prune_epochs": [epoch + 1 for epoch in self.repr_config.prune_epochs],
            "restore_epochs": [epoch + 1 for epoch in self.repr_config.restore_epochs],
            "completed_cycles": len(self._cycle_history),
            "cycle_history": list(self._cycle_history),
            "current_mean_filter_redundancy": self.mean_filter_redundancy(),
            "redundancy_by_scope": redundancy_by_scope,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features") and hasattr(self.model, "forward_head"):
            features = self.model.forward_features(x)
            try:
                features = self.model.forward_head(features, pre_logits=True)
            except TypeError:
                features = self.model.forward_head(features)
        else:
            features = self.model(x)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.ndim == 2:
            return features
        if features.ndim == 4:
            if features.shape[-1] == self.out_features:
                return features.mean(dim=(1, 2))
            return torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
        if features.ndim == 3:
            return features.mean(dim=1)
        return torch.flatten(features, 1)
