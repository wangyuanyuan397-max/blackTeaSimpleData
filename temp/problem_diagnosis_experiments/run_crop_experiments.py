"""不同物理视野的独立训练，以及多尺度后验融合诊断。"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from diagnosis_common import (  # noqa: E402
    DEFAULT_CLASSES,
    DiagnosticImageDataset,
    DiagnosticTransform,
    RepeatedDataset,
    TrainingOptions,
    aggregate_by_parent,
    classification_metrics,
    create_model,
    evaluate_model,
    load_checkpoint,
    make_loader,
    metrics_row,
    resolve_device,
    set_random_seed,
    train_model,
    write_csv,
    write_json,
)


# =============================================================================
# PyCharm 右键运行配置区
# 只需要修改这里的值，然后右键本文件 -> Run 'run_crop_experiments'。
# 本脚本不再读取任何命令行参数。
# =============================================================================
@dataclass(frozen=True)
class CropExperimentConfig:
    """Global/Local 尺度诊断的全部运行参数。"""

    # 数据与结果目录均根据本文件位置解析，不依赖 PyCharm 的 Working directory。
    dataset_root: Path = PROJECT_ROOT / "datasets_01234_original_split"
    output_dir: Path = EXPERIMENT_DIR / "results" / "crop_scale"

    # 默认运行真正的整图 Global，以及四种局部物理视野。
    crop_sizes: tuple[int, ...] = (408, 306, 204, 102)
    include_global: bool = True
    input_size: int = 224
    model_name: str = "mixnet_s"

    # 正式训练参数。
    epochs: int = 150
    batch_size: int = 32
    eval_batch_size: int = 16
    num_workers: int = 4
    train_repeats: int = 30
    val_views: int = 5
    test_views: int = 9
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    patience: int = 30
    label_smoothing: float = 0.0
    seed: int = 2026
    device: str = "auto"
    use_pretrained: bool = True
    use_amp: bool = True
    # 默认禁止在测试集上枚举融合组合，避免把测试集当成模型选择集。
    # 旧结果复现时才临时设为 True；正式流程使用 select_fusion_on_validation.py。
    evaluate_fusions: bool = False

    # 正式实验保持 None 和 False。调试时可设为 1 和 True，只检查前向流程。
    max_samples_per_class: int | None = None
    dry_run: bool = False


CONFIG = CropExperimentConfig()


def build_dataset(args, split: str, condition: str | int, training: bool, views: int):
    """所有尺度共用同一源数据和类别顺序，仅改变 crop 物理范围。"""

    return DiagnosticImageDataset(
        args.dataset_root,
        split,
        DiagnosticTransform(
            crop_size=1 if condition == "global" else int(condition),
            input_size=args.input_size,
            training=training,
            view_count=views,
            seed=args.seed,
            full_image=condition == "global",
        ),
        DEFAULT_CLASSES,
        args.max_samples_per_class,
    )


def save_evaluation(directory: Path, result: dict) -> None:
    """指标与逐样本预测分开保存。"""

    write_json(directory / "metrics.json", {"sample": result["sample"], "parent": result["parent"]})
    write_csv(directory / "sample_predictions.csv", result["sample_predictions"])
    write_csv(directory / "parent_predictions.csv", result["parent_predictions"])


def prediction_arrays(result: dict, classes: list[str]):
    """从结果行恢复标签与类别概率矩阵。"""

    rows = result["sample_predictions"]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[row[f"prob_{name}"] for name in classes] for row in rows],
        dtype=np.float64,
    )
    return labels, probabilities


def evaluate_fusions(scale_results: dict, output_dir: Path) -> list[dict]:
    """融合两尺度及全部尺度，判断是否存在互补信息。"""

    # 保留训练顺序，确保 global 条件与整数尺度可以同时参与组合。
    scales = list(scale_results)
    if len(scales) < 2:
        return []
    reference_dataset, reference_result = scale_results[scales[0]]
    reference_paths = [str(path) for path, _ in reference_dataset.samples]
    labels, _ = prediction_arrays(reference_result, reference_dataset.classes)
    combinations = list(itertools.combinations(scales, 2))
    if len(scales) > 2:
        combinations.append(tuple(scales))
    summary = []
    for combination in combinations:
        probability_list = []
        for scale in combination:
            dataset, result = scale_results[scale]
            if [str(path) for path, _ in dataset.samples] != reference_paths:
                raise RuntimeError("不同尺度样本顺序不一致，无法进行配对融合。")
            current_labels, probabilities = prediction_arrays(result, dataset.classes)
            if not np.array_equal(labels, current_labels):
                raise RuntimeError("不同尺度标签不一致。")
            probability_list.append(probabilities)
        fused = np.mean(probability_list, axis=0)
        parent_labels, parent_probabilities, parent_rows = aggregate_by_parent(
            reference_dataset,
            labels,
            fused,
        )
        condition = "fusion_" + "_".join(map(str, combination))
        result = {
            "sample": classification_metrics(labels, fused, reference_dataset.classes),
            "parent": classification_metrics(
                parent_labels,
                parent_probabilities,
                reference_dataset.classes,
            ),
        }
        row = metrics_row(condition, result)
        row["scales"] = "+".join(map(str, combination))
        summary.append(row)
        write_json(output_dir / condition / "metrics.json", result)
        write_csv(output_dir / condition / "parent_predictions.csv", parent_rows)
    return summary


def main() -> None:
    args = CONFIG
    crop_sizes = list(args.crop_sizes)
    if not crop_sizes or any(size <= 0 for size in crop_sizes):
        raise ValueError(f"CONFIG.crop_sizes 必须包含正整数，当前为：{args.crop_sizes}")
    conditions: list[str | int] = (["global"] if args.include_global else []) + crop_sizes
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    write_json(
        output_dir / "experiment_config.json",
        {
            **vars(args),
            "crop_sizes_parsed": crop_sizes,
            "conditions": conditions,
            "device_resolved": str(device),
        },
    )
    summary, scale_results = [], {}
    for condition in conditions:
        # 每个尺度重新设为同一种子，让初始化及采样差异尽可能受控。
        set_random_seed(args.seed)
        print(f"\n===== condition={condition}, input_size={args.input_size} =====")
        train_dataset = build_dataset(args, "train", condition, True, 1)
        val_dataset = build_dataset(args, "val", condition, False, args.val_views)
        test_single_dataset = build_dataset(args, "test", condition, False, 1)
        test_dataset = build_dataset(args, "test", condition, False, args.test_views)
        train_loader = make_loader(
            RepeatedDataset(train_dataset, args.train_repeats),
            args.batch_size,
            args.num_workers,
            True,
            args.seed,
        )
        val_loader = make_loader(
            val_dataset,
            args.eval_batch_size,
            args.num_workers,
            False,
            args.seed,
        )
        test_loader = make_loader(
            test_dataset,
            args.eval_batch_size,
            args.num_workers,
            False,
            args.seed,
        )
        test_single_loader = make_loader(
            test_single_dataset,
            args.eval_batch_size,
            args.num_workers,
            False,
            args.seed,
        )
        model = create_model(args.model_name, len(DEFAULT_CLASSES), args.use_pretrained)
        if args.dry_run:
            images, labels, _ = next(iter(train_loader))
            with torch.inference_mode():
                logits = model.to(device)(images.to(device))
            print(
                f"dry-run 通过：train={len(train_dataset)}, val={len(val_dataset)}, "
                f"test={len(test_dataset)}, batch={tuple(images.shape)}, "
                f"labels={tuple(labels.shape)}, logits={tuple(logits.shape)}"
            )
            continue

        condition_dir = output_dir / f"crop_{condition}"
        metadata = {
            "experiment": "crop_scale_diagnosis",
            "model_name": args.model_name,
            "num_classes": len(DEFAULT_CLASSES),
            "class_names": list(DEFAULT_CLASSES),
            "crop_size": None if condition == "global" else condition,
            "full_image": condition == "global",
            "input_size": args.input_size,
            "seed": args.seed,
        }
        checkpoint = train_model(
            model,
            train_loader,
            val_loader,
            val_dataset,
            device,
            condition_dir,
            TrainingOptions(
                args.epochs,
                args.learning_rate,
                args.weight_decay,
                args.patience,
                args.label_smoothing,
                args.use_amp,
            ),
            metadata,
        )
        best_model, _ = load_checkpoint(checkpoint, device)
        single_result = evaluate_model(
            best_model,
            test_single_loader,
            test_single_dataset,
            device,
        )
        save_evaluation(condition_dir / "test_single", single_result)
        single_name = "global" if condition == "global" else f"crop_{condition}_single"
        single_row = metrics_row(single_name, single_result)
        single_row.update(
            {"crop_size": condition, "view_count": 1, "checkpoint": str(checkpoint)}
        )
        summary.append(single_row)

        if condition == "global":
            # 整图没有不同 crop 位置，重复同一视图不会带来新信息。
            result = single_result
            fusion_dataset = test_single_dataset
        else:
            result = evaluate_model(best_model, test_loader, test_dataset, device)
            save_evaluation(condition_dir / f"test_multi{args.test_views}", result)
            multi_row = metrics_row(f"crop_{condition}_multi{args.test_views}", result)
            multi_row.update(
                {
                    "crop_size": condition,
                    "view_count": args.test_views,
                    "checkpoint": str(checkpoint),
                }
            )
            summary.append(multi_row)
            fusion_dataset = test_dataset
        scale_results[condition] = (fusion_dataset, result)
        print(
            f"测试：patch_acc={result['sample']['accuracy']:.4f}, "
            f"parent_acc={result['parent']['accuracy']:.4f}, "
            f"parent_qwk={result['parent']['qwk']:.4f}"
        )
    if args.dry_run:
        print("全部 dry-run 已通过。")
        return
    if args.evaluate_fusions:
        summary.extend(evaluate_fusions(scale_results, output_dir))
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", summary)
    print(f"结果已保存：{output_dir}")


if __name__ == "__main__":
    main()
