"""精选尺度的多随机种子复现实验。

默认只复现 408、204、102 三个尺度。每个种子独立训练并报告单中心 crop 与
多区域聚合结果，不在测试集枚举融合组合；融合组合应由
``select_fusion_on_validation.py`` 在验证集选择。
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_crop_experiments  # noqa: E402


# =============================================================================
# PyCharm 右键运行配置区
# 正式复现直接右键运行本文件；无命令行参数。
# =============================================================================
@dataclass(frozen=True)
class MultiSeedConfig:
    """精选尺度多种子配置。"""

    dataset_root: Path = PROJECT_ROOT / "datasets_01234_original_split"
    output_dir: Path = EXPERIMENT_DIR / "results" / "selected_scales_multiseed"
    scales: tuple[int, ...] = (408, 204, 102)
    seeds: tuple[int, ...] = (2026, 3407, 42)

    input_size: int = 224
    model_name: str = "mixnet_s"
    epochs: int = 150
    batch_size: int = 32
    # 9视图评估时实际前向图像数约为 eval_batch_size×9；4 可适配当前 8GB 显存。
    # 评估 batch 大小不改变权重和逐图聚合定义，因此可与归档 seed 直接比较。
    eval_batch_size: int = 4
    num_workers: int = 4
    train_repeats: int = 30
    val_views: int = 5
    test_views: int = 9
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    patience: int = 30
    label_smoothing: float = 0.0
    device: str = "auto"
    use_pretrained: bool = True
    use_amp: bool = True

    # 2026 已经按完全相同训练设置正式跑完，直接复用归档结果，避免重复耗时训练。
    # 如果归档文件不存在，脚本会自动退回到本地重新训练该 seed。
    archived_seed_summaries: tuple[tuple[int, Path], ...] = (
        (
            2026,
            Path(
                r"E:\docs\服务器跑模型结果备份\data01234\grid裁剪30+basic处理的模型结果"
                r"\problem_diagnosis_experiments\results\crop_scale\summary.csv"
            ),
        ),
    )

    # True 时，已有完整 summary.csv 的种子/尺度不会重复训练。
    # 当前断点粒度是“单个 seed × 单个尺度”，中断后最多重跑一个尺度。
    skip_completed_seeds: bool = True

    # 仅供流程测试；正式实验必须保持 None 和 False。
    max_samples_per_class: int | None = None
    dry_run: bool = False


CONFIG = MultiSeedConfig()


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取结果表。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """保存 UTF-8-BOM 汇总表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def expected_conditions(scales: tuple[int, ...], test_views: int) -> set[str]:
    """返回本轮正式复现必须具备的结果条件。"""

    return {
        condition
        for scale in scales
        for condition in (f"crop_{scale}_single", f"crop_{scale}_multi{test_views}")
    }


def seed_complete(summary_path: Path, scales: tuple[int, ...], test_views: int) -> bool:
    """检查一个种子的所有目标条件是否已经完整落盘。"""

    if not summary_path.is_file():
        return False
    conditions = {row["condition"] for row in read_csv(summary_path)}
    expected = expected_conditions(scales, test_views)
    return expected.issubset(conditions)


def scale_complete(summary_path: Path, scale: int, test_views: int) -> bool:
    """检查单个 seed×尺度是否已经完整产生 single/multi-view 两行结果。"""

    return seed_complete(summary_path, (scale,), test_views)


def collect_seed_summary(config: MultiSeedConfig, seed: int) -> Path:
    """把三个尺度的独立结果合并为该 seed 的 summary，便于统一汇总。"""

    seed_dir = config.output_dir / f"seed_{seed}"
    combined_rows: list[dict[str, Any]] = []
    for scale in config.scales:
        scale_summary = seed_dir / f"scale_{scale}" / "summary.csv"
        if not scale_complete(scale_summary, scale, config.test_views):
            raise RuntimeError(f"seed={seed}, scale={scale} 的结果不完整：{scale_summary}")
        allowed = expected_conditions((scale,), config.test_views)
        combined_rows.extend(
            row for row in read_csv(scale_summary) if row["condition"] in allowed
        )
    summary_path = seed_dir / "summary.csv"
    write_csv(summary_path, combined_rows)
    (seed_dir / "summary.json").write_text(
        json.dumps(combined_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def train_seed(config: MultiSeedConfig, seed: int) -> Path:
    """调用已验证的尺度训练器，但关闭 Global 和测试集融合枚举。"""

    seed_dir = config.output_dir / f"seed_{seed}"
    summary_path = seed_dir / "summary.csv"
    if config.skip_completed_seeds and seed_complete(
        summary_path,
        config.scales,
        config.test_views,
    ):
        print(f"种子 {seed} 已完整完成，跳过训练。")
        return summary_path

    for scale in config.scales:
        scale_dir = seed_dir / f"scale_{scale}"
        scale_summary = scale_dir / "summary.csv"
        if config.skip_completed_seeds and scale_complete(
            scale_summary,
            scale,
            config.test_views,
        ):
            print(f"种子 {seed}、尺度 {scale} 已完成，跳过。")
            continue

        # 每个尺度单独落盘，使长任务中断后只需重跑未完成的尺度。
        run_crop_experiments.CONFIG = run_crop_experiments.CropExperimentConfig(
            dataset_root=config.dataset_root,
            output_dir=scale_dir,
            crop_sizes=(scale,),
            include_global=False,
            input_size=config.input_size,
            model_name=config.model_name,
            epochs=config.epochs,
            batch_size=config.batch_size,
            eval_batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            train_repeats=config.train_repeats,
            val_views=config.val_views,
            test_views=config.test_views,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=config.patience,
            label_smoothing=config.label_smoothing,
            seed=seed,
            device=config.device,
            use_pretrained=config.use_pretrained,
            use_amp=config.use_amp,
            evaluate_fusions=False,
            max_samples_per_class=config.max_samples_per_class,
            dry_run=config.dry_run,
        )
        run_crop_experiments.main()

    if config.dry_run:
        return summary_path
    return collect_seed_summary(config, seed)


def aggregate(config: MultiSeedConfig, summary_paths: dict[int, Path]) -> None:
    """汇总逐种子结果，并计算均值、样本标准差和最小/最大值。"""

    allowed_conditions = expected_conditions(config.scales, config.test_views)
    raw_rows: list[dict[str, Any]] = []
    for seed, path in summary_paths.items():
        if not path.is_file():
            if config.dry_run:
                continue
            raise FileNotFoundError(f"种子 {seed} 缺少汇总：{path}")
        for row in read_csv(path):
            if row["condition"] not in allowed_conditions:
                continue
            raw_rows.append({"seed": seed, "summary_source": str(path), **row})
    write_csv(config.output_dir / "per_seed_results.csv", raw_rows)

    metric_names = (
        "parent_accuracy",
        "parent_macro_f1",
        "parent_mae",
        "parent_qwk",
        "parent_nll",
    )
    conditions = sorted({row["condition"] for row in raw_rows})
    if set(conditions) != allowed_conditions:
        missing = sorted(allowed_conditions - set(conditions))
        raise RuntimeError(f"多seed汇总缺少条件：{missing}")
    aggregate_rows: list[dict[str, Any]] = []
    for condition in conditions:
        subset = [row for row in raw_rows if row["condition"] == condition]
        if len(subset) != len(config.seeds):
            raise RuntimeError(
                f"{condition} 只有 {len(subset)} 个seed，预期 {len(config.seeds)} 个。"
            )
        record: dict[str, Any] = {
            "condition": condition,
            "seed_count": len(subset),
            "seeds": "+".join(str(row["seed"]) for row in subset),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in subset]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            record[f"{metric}_min"] = min(values)
            record[f"{metric}_max"] = max(values)
        aggregate_rows.append(record)
    write_csv(config.output_dir / "multiseed_summary.csv", aggregate_rows)
    (config.output_dir / "multiseed_summary.json").write_text(
        json.dumps(aggregate_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# 精选尺度多随机种子复现",
        "",
        f"- 尺度：{', '.join(map(str, config.scales))}",
        f"- 种子：{', '.join(map(str, config.seeds))}",
        "- seed 2026 复用完全相同设置的既有正式结果；seed 3407、42 在本目录新训练。",
        "- 测试集不搜索融合组合；仅报告冻结的单尺度 single/multi-view 结果。",
        "",
        "| 条件 | Seed 数 | Acc 均值±SD | Macro-F1 均值±SD | MAE 均值±SD | QWK 均值±SD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        report.append(
            "| {condition} | {seed_count} | {parent_accuracy_mean:.2%}±{parent_accuracy_std:.2%} | "
            "{parent_macro_f1_mean:.4f}±{parent_macro_f1_std:.4f} | "
            "{parent_mae_mean:.4f}±{parent_mae_std:.4f} | "
            "{parent_qwk_mean:.4f}±{parent_qwk_std:.4f} |".format(**row)
        )
    report.extend(
        [
            "",
            "> 测试集仅 25 张原图；多种子主要衡量训练随机性，不能替代增加独立样本。",
            "> 评估 batch_size=4 仅为适配显存，不改变 checkpoint、视图数量或原图级聚合定义。",
            "",
        ]
    )
    (config.output_dir / "MULTISEED_REPORT.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def main() -> None:
    config = CONFIG
    if not config.scales or any(scale <= 0 for scale in config.scales):
        raise ValueError("CONFIG.scales 必须是非空正整数元组。")
    if len(set(config.seeds)) != len(config.seeds) or not config.seeds:
        raise ValueError("CONFIG.seeds 必须非空且不能重复。")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        **vars(config),
        "archived_seed_summaries": [
            {"seed": seed, "summary_path": str(path)}
            for seed, path in config.archived_seed_summaries
        ],
        "evaluate_test_fusions": False,
        "resume_granularity": "seed_x_scale",
    }
    (config.output_dir / "multiseed_config.json").write_text(
        json.dumps(
            config_payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    archived = {
        seed: path
        for seed, path in config.archived_seed_summaries
        if seed in config.seeds and seed_complete(path, config.scales, config.test_views)
    }
    for seed, path in archived.items():
        print(f"种子 {seed} 复用既有正式结果：{path}")
    summary_paths = {
        seed: archived[seed] if seed in archived else train_seed(config, seed)
        for seed in config.seeds
    }
    if not config.dry_run:
        aggregate(config, summary_paths)
        print(f"多种子复现完成：{config.output_dir / 'MULTISEED_REPORT.md'}")
    else:
        print("多种子 dry-run 全部通过；未生成正式统计汇总。")


if __name__ == "__main__":
    main()
