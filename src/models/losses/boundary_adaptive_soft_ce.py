"""基于阶段边界距离的自适应 soft-label 交叉熵损失。"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import LOSSES


@LOSSES.register("boundary_adaptive_soft_ce")
class BoundaryAdaptiveSoftCrossEntropyLoss(nn.Module):
    """根据样本发酵时间到相邻阶段边界的距离，动态生成 soft label。

    这个 loss 不改模型结构，只改监督信号：
    - 离阶段边界越近，相邻类别拿到的概率越高；
    - 离阶段边界越远，标签越接近 one-hot hard label；
    - 图片路径必须能从文件名前缀解析出时间，例如 00/05/10/.../60。
    """

    def __init__(
        self,
        num_classes: int = 4,
        eps_max: float = 0.20,
        tau: float = 0.5,
        max_total_soft: float = 0.30,
        boundaries: Sequence[float] | None = None,
        time_prefixes: Sequence[str] | None = None,
        check_label_consistency: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.eps_max = float(eps_max)
        self.tau = float(tau)
        self.max_total_soft = float(max_total_soft)
        self.check_label_consistency = bool(check_label_consistency)
        self.reduction = str(reduction).lower()

        if self.num_classes != 4:
            raise ValueError("boundary_adaptive_soft_ce 当前按 Pre/Slight/Moderate/Over 四阶段设计。")
        if self.eps_max < 0:
            raise ValueError("eps_max 必须大于等于 0。")
        if self.tau <= 0:
            raise ValueError("tau 必须大于 0。")
        if self.max_total_soft < 0:
            raise ValueError("max_total_soft 必须大于等于 0。")
        if self.reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction 只能是 mean、sum 或 none。")

        # 三个相邻阶段边界分别是：Pre/Slight、Slight/Moderate、Moderate/Over。
        boundary_values = list(boundaries) if boundaries is not None else [1.25, 2.75, 4.75]
        if len(boundary_values) != self.num_classes - 1:
            raise ValueError("boundaries 长度必须等于 num_classes - 1。")
        self.boundaries = [float(value) for value in boundary_values]

        # 文件名前缀和发酵小时的对应关系：00->0.0h，05->0.5h，...，60->6.0h。
        prefixes = list(time_prefixes) if time_prefixes is not None else [
            "00",
            "05",
            "10",
            "15",
            "20",
            "25",
            "30",
            "35",
            "40",
            "45",
            "50",
            "55",
            "60",
        ]
        self.time_prefix_to_hour = {str(prefix): int(prefix) / 10.0 for prefix in prefixes}
        # 按长度降序拼接，避免未来出现不同长度前缀时被短前缀提前截断。
        prefix_pattern = "|".join(re.escape(prefix) for prefix in sorted(prefixes, key=len, reverse=True))
        self.time_prefix_pattern = re.compile(rf"^({prefix_pattern})")

    def _extract_logits(self, outputs) -> torch.Tensor:
        """从普通 logits 或 (logits, auxiliary/features) 输出中取分类 logits。"""
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def _stage_from_time(self, time_h: float) -> int:
        """根据阶段边界把时间映射到 0/1/2/3 四个 hard stage。"""
        if time_h < self.boundaries[0]:
            return 0
        if time_h < self.boundaries[1]:
            return 1
        if time_h < self.boundaries[2]:
            return 2
        return 3

    def _parse_time_from_path(self, path_like) -> float:
        """从图片文件名前缀解析发酵时间小时数。"""
        stem = Path(str(path_like)).stem
        match = self.time_prefix_pattern.match(stem)
        if match is None:
            raise ValueError(
                "boundary_adaptive_soft_ce 无法从文件名前缀解析发酵时间："
                f"{path_like}。文件名应以 00/05/10/.../60 开头。"
            )
        return self.time_prefix_to_hour[match.group(1)]

    def _paths_from_extra_targets(self, extra_targets) -> Iterable:
        """从 Trainer/Evaluator 注入的 extra_targets 中取得当前 batch 的图片路径。"""
        if not isinstance(extra_targets, dict) or "sample_paths" not in extra_targets:
            raise ValueError(
                "boundary_adaptive_soft_ce 需要 extra_targets['sample_paths']，"
                "请确认 Dataset 返回图片路径且 Trainer/Evaluator 已注入 sample_paths。"
            )
        sample_paths = extra_targets["sample_paths"]
        if isinstance(sample_paths, (str, Path)):
            return [sample_paths]
        return list(sample_paths)

    def _build_soft_targets(
        self,
        targets: torch.Tensor,
        sample_paths: Iterable,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """为一个 batch 的样本动态生成 [B, C] soft target 矩阵。"""
        paths = list(sample_paths)
        if len(paths) != int(targets.numel()):
            raise ValueError(
                "sample_paths 数量和 targets 数量不一致："
                f"paths={len(paths)}, targets={int(targets.numel())}。"
            )

        soft_targets = torch.zeros(
            (int(targets.numel()), self.num_classes),
            device=device,
            dtype=dtype,
        )
        cpu_targets = targets.detach().cpu().long().tolist()

        for row_index, (hard_label, sample_path) in enumerate(zip(cpu_targets, paths)):
            if hard_label < 0 or hard_label >= self.num_classes:
                raise ValueError(f"标签编号越界：{hard_label}，num_classes={self.num_classes}。")

            time_h = self._parse_time_from_path(sample_path)
            time_stage = self._stage_from_time(time_h)
            if self.check_label_consistency and time_stage != int(hard_label):
                raise ValueError(
                    "文件名前缀解析出的阶段和文件夹标签不一致："
                    f"path={sample_path}, time_h={time_h}, "
                    f"time_stage={time_stage}, folder_label={hard_label}。"
                )

            alpha_left = 0.0
            alpha_right = 0.0

            if hard_label > 0:
                left_boundary = self.boundaries[hard_label - 1]
                distance_left = max(0.0, time_h - left_boundary)
                alpha_left = self.eps_max * math.exp(-distance_left / self.tau)

            if hard_label < self.num_classes - 1:
                right_boundary = self.boundaries[hard_label]
                distance_right = max(0.0, right_boundary - time_h)
                alpha_right = self.eps_max * math.exp(-distance_right / self.tau)

            total_soft = alpha_left + alpha_right
            if total_soft > self.max_total_soft and total_soft > 0.0:
                scale = self.max_total_soft / total_soft
                alpha_left *= scale
                alpha_right *= scale

            if hard_label > 0:
                soft_targets[row_index, hard_label - 1] = float(alpha_left)
            if hard_label < self.num_classes - 1:
                soft_targets[row_index, hard_label + 1] = float(alpha_right)
            soft_targets[row_index, hard_label] = float(1.0 - alpha_left - alpha_right)

        return soft_targets

    def forward(self, outputs, targets: torch.Tensor, extra_targets=None) -> torch.Tensor:
        """使用动态 soft target 计算交叉熵。"""
        logits = self._extract_logits(outputs)
        sample_paths = self._paths_from_extra_targets(extra_targets)
        soft_targets = self._build_soft_targets(
            targets=targets,
            sample_paths=sample_paths,
            device=logits.device,
            dtype=logits.dtype,
        )

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

    def preview_soft_label(self, time_h: float, hard_label: int | None = None) -> list[float]:
        """便于调试：返回某个时间点的 soft label 数值列表。"""
        label = self._stage_from_time(float(time_h)) if hard_label is None else int(hard_label)
        dummy_target = torch.tensor([label], dtype=torch.long)
        dummy_path = f"{int(round(float(time_h) * 10)):02d}_preview.bmp"
        soft = self._build_soft_targets(
            targets=dummy_target,
            sample_paths=[dummy_path],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )[0]
        return [float(value) for value in soft.tolist()]
