"""只在验证集选择尺度融合，并在冻结后评估一次测试集。

该脚本解决旧流程中的测试集选择偏差：候选单尺度/融合组合只根据验证集排序；
胜出组合确定后，代码才会构建测试集并进行一次冻结评估。
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
ARCHIVED_RESULTS_ROOT = Path(
    r"E:\docs\服务器跑模型结果备份\data01234\grid裁剪30+basic处理的模型结果"
    r"\problem_diagnosis_experiments\results"
)
# 当前电脑优先复用用户给出的服务器 checkpoint；换机后自动回退到本目录结果。
DEFAULT_RESULTS_ROOT = (
    ARCHIVED_RESULTS_ROOT
    if ARCHIVED_RESULTS_ROOT.is_dir()
    else EXPERIMENT_DIR / "results"
)
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from diagnosis_common import (  # noqa: E402
    DEFAULT_CLASSES,
    DiagnosticImageDataset,
    DiagnosticTransform,
    aggregate_by_parent,
    classification_metrics,
    evaluate_model,
    load_checkpoint,
    make_loader,
    resolve_device,
    write_csv,
    write_json,
)


# =============================================================================
# PyCharm 右键运行配置区
# 先运行尺度实验得到各 checkpoint，再直接右键运行本文件。
# =============================================================================
@dataclass(frozen=True)
class FrozenFusionConfig:
    """验证集选融合、测试集冻结评估配置。"""

    dataset_root: Path = PROJECT_ROOT / "datasets_01234_original_split"
    crop_results_dir: Path = DEFAULT_RESULTS_ROOT / "crop_scale"
    output_dir: Path = EXPERIMENT_DIR / "results" / "frozen_fusion"

    # 默认排除当前不公平的 Global；候选名与 crop_<name>/best.pth 对应。
    candidate_conditions: tuple[str, ...] = ("408", "306", "204", "102")
    combination_sizes: tuple[int, ...] = (1, 2)
    input_size: int = 224
    val_views: int = 5
    test_views: int = 9
    # 多视图会在网络前向时展开为 batch_size × views；8GB 显卡默认用 4 更稳妥。
    batch_size: int = 4
    num_workers: int = 4
    seed: int = 2026
    device: str = "auto"

    # 主排序指标；平局时依次比较 QWK、macro-F1、负 NLL。
    selection_metric: str = "accuracy"

    # 当前历史 test 已经被旧流程用于枚举融合；如换成从未查看的新测试集再修改此说明。
    test_status_note: str = (
        "本次脚本内在验证集冻结后才评估 test；但该 test 此前已被旧流程用于融合探索，"
        "因此结果属于回顾性纠偏，不等于全新盲测。"
    )


CONFIG = FrozenFusionConfig()


def condition_is_global(condition: str) -> bool:
    return str(condition).lower() == "global"


def checkpoint_path(config: FrozenFusionConfig, condition: str) -> Path:
    """返回某候选条件的最佳权重路径。"""

    return config.crop_results_dir / f"crop_{condition}" / "best.pth"


def build_dataset(
    config: FrozenFusionConfig,
    split: str,
    condition: str,
    views: int,
) -> DiagnosticImageDataset:
    """根据条件构建整图或局部多视图数据集。"""

    full_image = condition_is_global(condition)
    crop_size = 1 if full_image else int(condition)
    return DiagnosticImageDataset(
        config.dataset_root,
        split,
        DiagnosticTransform(
            crop_size=crop_size,
            input_size=config.input_size,
            training=False,
            view_count=1 if full_image else views,
            seed=config.seed,
            full_image=full_image,
        ),
        DEFAULT_CLASSES,
    )


def prediction_arrays(result: dict[str, Any], classes: Sequence[str]):
    """从逐样本预测恢复路径、标签和概率矩阵。"""

    rows = result["sample_predictions"]
    paths = [row["path"] for row in rows]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[row[f"prob_{name}"] for name in classes] for row in rows],
        dtype=np.float64,
    )
    return paths, labels, probabilities


def fuse_results(
    named_results: Sequence[tuple[str, DiagnosticImageDataset, dict[str, Any]]],
) -> tuple[DiagnosticImageDataset, dict[str, Any]]:
    """按样本配对平均概率，并重新计算两级指标。"""

    reference_name, reference_dataset, reference_result = named_results[0]
    del reference_name
    reference_paths, labels, first_probabilities = prediction_arrays(
        reference_result,
        reference_dataset.classes,
    )
    probability_list = [first_probabilities]
    for _, dataset, result in named_results[1:]:
        paths, current_labels, probabilities = prediction_arrays(result, dataset.classes)
        if paths != reference_paths or not np.array_equal(labels, current_labels):
            raise RuntimeError("候选模型的样本/标签顺序不一致，不能进行配对融合。")
        probability_list.append(probabilities)
    fused = np.mean(probability_list, axis=0)
    parent_labels, parent_probabilities, parent_rows = aggregate_by_parent(
        reference_dataset,
        labels,
        fused,
    )
    sample_predictions: list[dict[str, Any]] = []
    predictions = fused.argmax(axis=1)
    for index, ((path, _), label, prediction, probability) in enumerate(
        zip(reference_dataset.samples, labels, predictions, fused)
    ):
        row: dict[str, Any] = {
            "path": str(path),
            "parent_id": reference_dataset.parent_ids[index],
            "label": int(label),
            "prediction": int(prediction),
            "correct": int(label == prediction),
            "confidence": float(probability.max()),
        }
        row.update(
            {
                f"prob_{name}": float(probability[class_index])
                for class_index, name in enumerate(reference_dataset.classes)
            }
        )
        sample_predictions.append(row)
    return reference_dataset, {
        "sample": classification_metrics(labels, fused, reference_dataset.classes),
        "parent": classification_metrics(
            parent_labels,
            parent_probabilities,
            reference_dataset.classes,
        ),
        "sample_predictions": sample_predictions,
        "parent_predictions": parent_rows,
    }


def candidate_key(row: dict[str, Any], metric: str) -> tuple[float, ...]:
    """验证集排序键；最后偏好更少模型，保持方案克制。"""

    metrics = row["metrics"]
    if metric not in metrics:
        raise ValueError(f"未知 selection_metric：{metric}")
    return (
        float(metrics[metric]),
        float(metrics["qwk"]),
        float(metrics["macro_f1"]),
        -float(metrics["nll"]),
        -float(row["component_count"]),
    )


def save_evaluation(directory: Path, result: dict[str, Any]) -> None:
    """保存指标和逐原图预测。"""

    write_json(
        directory / "metrics.json",
        {"sample": result["sample"], "parent": result["parent"]},
    )
    write_csv(directory / "sample_predictions.csv", result["sample_predictions"])
    write_csv(directory / "parent_predictions.csv", result["parent_predictions"])


def paired_correctness_comparison(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """在同一批原图上比较正确/错误变化，并计算精确 McNemar p 值。"""

    baseline = {row["parent_id"]: bool(row["correct"]) for row in baseline_rows}
    candidate = {row["parent_id"]: bool(row["correct"]) for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise RuntimeError("基线与融合结果的原图集合不一致，不能做配对比较。")
    gained = sum(not baseline[key] and candidate[key] for key in baseline)
    lost = sum(baseline[key] and not candidate[key] for key in baseline)
    discordant = gained + lost
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(gained, lost) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "paired_sample_count": len(baseline),
        "fusion_gained_correct": gained,
        "fusion_lost_correct": lost,
        "discordant_count": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    }


def main() -> None:
    config = CONFIG
    device = resolve_device(config.device)
    if config.selection_metric not in {"accuracy", "qwk", "macro_f1"}:
        raise ValueError("selection_metric 只支持 accuracy、qwk 或 macro_f1。")
    if not config.candidate_conditions:
        raise ValueError("candidate_conditions 不能为空。")
    if any(size <= 0 or size > len(config.candidate_conditions) for size in config.combination_sizes):
        raise ValueError("combination_sizes 超出候选数量范围。")
    for condition in config.candidate_conditions:
        if not checkpoint_path(config, condition).is_file():
            raise FileNotFoundError(
                f"缺少候选权重：{checkpoint_path(config, condition)}"
            )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        config.output_dir / "selection_config.json",
        {
            **vars(config),
            "rule": "仅验证集选择；胜出组合确定后才构建测试集",
        },
    )

    # 第一阶段：只接触验证集。
    validation_components: dict[
        str, tuple[DiagnosticImageDataset, dict[str, Any]]
    ] = {}
    for condition in config.candidate_conditions:
        print(f"验证集评估候选尺度：{condition}")
        dataset = build_dataset(config, "val", condition, config.val_views)
        loader = make_loader(
            dataset,
            config.batch_size,
            config.num_workers,
            False,
            config.seed,
        )
        model, _ = load_checkpoint(checkpoint_path(config, condition), device)
        result = evaluate_model(model, loader, dataset, device)
        validation_components[condition] = (dataset, result)
        del model, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidates: list[dict[str, Any]] = []
    validation_results: dict[str, tuple[DiagnosticImageDataset, dict[str, Any]]] = {}
    for size in sorted(set(config.combination_sizes)):
        for combination in itertools.combinations(config.candidate_conditions, size):
            name = "+".join(combination)
            selected = [
                (condition, *validation_components[condition])
                for condition in combination
            ]
            dataset, result = fuse_results(selected)
            validation_results[name] = (dataset, result)
            candidates.append(
                {
                    "candidate": name,
                    "components": list(combination),
                    "component_count": len(combination),
                    "metrics": result["parent"],
                }
            )

    ranked = sorted(
        candidates,
        key=lambda row: candidate_key(row, config.selection_metric),
        reverse=True,
    )
    ranking_rows = []
    for rank, row in enumerate(ranked, start=1):
        metrics = row["metrics"]
        ranking_rows.append(
            {
                "rank": rank,
                "candidate": row["candidate"],
                "component_count": row["component_count"],
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "val_mae": metrics["mae"],
                "val_qwk": metrics["qwk"],
                "val_nll": metrics["nll"],
            }
        )
    write_csv(config.output_dir / "validation_ranking.csv", ranking_rows)
    write_json(config.output_dir / "validation_ranking.json", ranking_rows)

    winner = ranked[0]
    winner_name = winner["candidate"]
    winner_components = tuple(winner["components"])
    single_candidates = [row for row in ranked if row["component_count"] == 1]
    if not single_candidates:
        raise ValueError("combination_sizes 必须包含 1，才能确定公平的单尺度基线。")
    best_single_name = single_candidates[0]["candidate"]
    print(f"验证集冻结胜出组合：{winner_name}")
    save_evaluation(
        config.output_dir / "winner_validation",
        validation_results[winner_name][1],
    )

    # 第二阶段：组合已经冻结，现在才读取一次测试集。
    test_components_by_name: dict[
        str, tuple[str, DiagnosticImageDataset, dict[str, Any]]
    ] = {}
    required_test_conditions = tuple(
        dict.fromkeys((*winner_components, best_single_name))
    )
    for condition in required_test_conditions:
        print(f"冻结测试评估组成尺度：{condition}")
        dataset = build_dataset(config, "test", condition, config.test_views)
        loader = make_loader(
            dataset,
            config.batch_size,
            config.num_workers,
            False,
            config.seed,
        )
        model, _ = load_checkpoint(checkpoint_path(config, condition), device)
        result = evaluate_model(model, loader, dataset, device)
        test_components_by_name[condition] = (condition, dataset, result)
        del model, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    test_components = [test_components_by_name[name] for name in winner_components]
    _, frozen_test_result = fuse_results(test_components)
    save_evaluation(config.output_dir / "frozen_test", frozen_test_result)
    best_single_test_result = test_components_by_name[best_single_name][2]
    save_evaluation(
        config.output_dir / "frozen_test_best_single",
        best_single_test_result,
    )
    paired_comparison = paired_correctness_comparison(
        best_single_test_result["parent_predictions"],
        frozen_test_result["parent_predictions"],
    )

    final = {
        "winner": winner_name,
        "components": list(winner_components),
        "selected_is_fusion": len(winner_components) > 1,
        "validation_selected_best_single": best_single_name,
        "selection_metric": config.selection_metric,
        "validation_metrics": validation_results[winner_name][1]["parent"],
        "frozen_test_metrics": frozen_test_result["parent"],
        "best_single_frozen_test_metrics": best_single_test_result["parent"],
        "fusion_vs_best_single_paired": paired_comparison,
        "test_evaluated_after_selection": True,
        "test_status_note": config.test_status_note,
    }
    write_json(config.output_dir / "FROZEN_FUSION_RESULT.json", final)
    comparison_line = (
        f"- 融合相对单尺度：新增答对 {paired_comparison['fusion_gained_correct']} 张，"
        f"损失答对 {paired_comparison['fusion_lost_correct']} 张，"
        f"精确 McNemar p={paired_comparison['mcnemar_exact_two_sided_p']:.4f}"
        if len(winner_components) > 1
        else "- 验证集未选中融合；冻结方案就是最佳单尺度，不存在融合增益。"
    )
    report = [
        "# 验证集选择、冻结测试的融合结果",
        "",
        f"- 验证集胜出组合：`{winner_name}`",
        f"- 验证集选出的最佳单尺度：`{best_single_name}`",
        f"- 选择指标：`{config.selection_metric}`",
        f"- 验证集准确率：{final['validation_metrics']['accuracy']:.2%}",
        f"- 冻结测试准确率：{final['frozen_test_metrics']['accuracy']:.2%}",
        f"- 冻结测试 Macro-F1：{final['frozen_test_metrics']['macro_f1']:.4f}",
        f"- 冻结测试 MAE：{final['frozen_test_metrics']['mae']:.4f}",
        f"- 冻结测试 QWK：{final['frozen_test_metrics']['qwk']:.4f}",
        f"- 最佳单尺度冻结测试准确率：{final['best_single_frozen_test_metrics']['accuracy']:.2%}",
        comparison_line,
        "",
        "> 候选组合只在验证集排序；脚本在确定胜出组合后才构建测试集。",
        f"> 测试集状态：{config.test_status_note}",
        "",
    ]
    (config.output_dir / "FROZEN_FUSION_RESULT.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    print(f"冻结融合实验完成：{config.output_dir}")


if __name__ == "__main__":
    main()
