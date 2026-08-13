"""把综合颜色扰动拆成可单独归因的颜色分量实验。

该脚本分别改变亮度、对比度、饱和度、色相和冷暖白平衡；每个条件只启用一种
变换。默认先在验证集运行，选择强度后再决定是否进行一次冻结测试。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
ARCHIVED_RESULTS_ROOT = Path(
    r"E:\docs\服务器跑模型结果备份\data01234\grid裁剪30+basic处理的模型结果"
    r"\problem_diagnosis_experiments\results"
)
# 当前电脑优先复用用户给出的服务器 checkpoint；换机后回退到本目录结果。
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
    PerturbationSpec,
    evaluate_model,
    load_checkpoint,
    make_loader,
    metrics_row,
    resolve_device,
    set_random_seed,
    write_csv,
    write_json,
)


# =============================================================================
# PyCharm 右键运行配置区
# 默认直接读取已有 crop_408 权重并先跑验证集，不需要任何命令行参数。
# =============================================================================
@dataclass(frozen=True)
class ColorComponentConfig:
    """颜色分量诊断的全部配置。"""

    checkpoint: Path = DEFAULT_RESULTS_ROOT / "crop_scale" / "crop_408" / "best.pth"
    dataset_root: Path = PROJECT_ROOT / "datasets_01234_original_split"
    output_dir: Path = EXPERIMENT_DIR / "results" / "color_components"

    # 正式流程先保持 val；在验证集固定强度和判读规则后，才改成 test 评估一次。
    split: str = "val"
    val_views: int = 5
    test_views: int = 9

    # 1.0 表示原图不变；上下两个方向各设置轻度和中度水平，便于观察剂量趋势。
    brightness_factors: tuple[float, ...] = (0.70, 0.85, 1.15, 1.30)
    contrast_factors: tuple[float, ...] = (0.70, 0.85, 1.15, 1.30)
    saturation_factors: tuple[float, ...] = (0.50, 0.75, 1.25, 1.50)

    # torchvision 的 hue 范围是 [-0.5, 0.5]；0.03/0.06 约对应 11°/22°。
    hue_shifts: tuple[float, ...] = (-0.06, -0.03, 0.03, 0.06)

    # 正值偏暖，负值偏冷；实现会尽量保持整图平均亮度不变。
    temperature_strengths: tuple[float, ...] = (-0.20, -0.10, 0.10, 0.20)

    # 仅当 split="test" 时使用。必须先根据 val 预先填入要冻结评估的条件，
    # 例如 (("brightness", 0.70), ("hue", -0.06))；空元组会拒绝访问 test。
    frozen_test_conditions: tuple[tuple[str, float], ...] = ()

    model_name: str | None = None
    num_classes: int | None = None
    crop_size: int | None = None
    input_size: int | None = None
    full_image: bool = False
    batch_size: int = 4
    num_workers: int = 4
    seed: int = 2026
    device: str = "auto"

    # 调试时可以改成 1 和 True；正式结果必须保持 None 和 False。
    max_samples_per_class: int | None = None
    dry_run: bool = False


CONFIG = ColorComponentConfig()


def validate_config(config: ColorComponentConfig) -> None:
    """提前拦截无效强度，避免长时间推理后才报错。"""

    if config.split not in {"val", "test"}:
        raise ValueError("CONFIG.split 只能是 'val' 或 'test'。")
    if config.split == "test" and not config.frozen_test_conditions:
        raise ValueError(
            "split='test' 时必须先填写 frozen_test_conditions，禁止在 test 枚举全部强度。"
        )
    factor_groups = {
        "brightness_factors": config.brightness_factors,
        "contrast_factors": config.contrast_factors,
        "saturation_factors": config.saturation_factors,
    }
    for name, values in factor_groups.items():
        if not values or any(value < 0 or value == 1 for value in values):
            raise ValueError(f"{name} 必须非空、非负且不包含无变化因子 1.0。")
    if not config.hue_shifts or any(
        value == 0 or not -0.5 <= value <= 0.5 for value in config.hue_shifts
    ):
        raise ValueError("hue_shifts 必须非空、非零且位于 [-0.5, 0.5]。")
    if not config.temperature_strengths or any(
        value == 0 or not -0.5 <= value <= 0.5
        for value in config.temperature_strengths
    ):
        raise ValueError("temperature_strengths 必须非空、非零且位于 [-0.5, 0.5]。")
    if min(config.val_views, config.test_views, config.batch_size) <= 0:
        raise ValueError("val_views、test_views 和 batch_size 必须为正数。")


def build_conditions(config: ColorComponentConfig) -> list[PerturbationSpec]:
    """构造单因素条件；每个条件只改变一种颜色属性。"""

    conditions = [PerturbationSpec("original")]
    conditions.extend(PerturbationSpec("brightness", value) for value in config.brightness_factors)
    conditions.extend(PerturbationSpec("contrast", value) for value in config.contrast_factors)
    conditions.extend(PerturbationSpec("saturation", value) for value in config.saturation_factors)
    conditions.extend(PerturbationSpec("hue", value) for value in config.hue_shifts)
    conditions.extend(
        PerturbationSpec("white_balance_temperature", value)
        for value in config.temperature_strengths
    )
    return conditions


def conditions_for_split(config: ColorComponentConfig) -> list[PerturbationSpec]:
    """val 运行完整诊断；test 只运行预先冻结的条件。"""

    all_conditions = build_conditions(config)
    if config.split == "val":
        return all_conditions
    available = {
        (condition.name, float(condition.value)): condition
        for condition in all_conditions
        if condition.name != "original"
    }
    requested = [
        (str(name), float(value))
        for name, value in config.frozen_test_conditions
    ]
    if len(set(requested)) != len(requested):
        raise ValueError("frozen_test_conditions 不能包含重复条件。")
    missing = [condition for condition in requested if condition not in available]
    if missing:
        raise ValueError(
            "冻结测试条件不在 val 候选配置中，请先同步对应强度："
            f"{missing}"
        )
    return [all_conditions[0], *(available[condition] for condition in requested)]


def value_token(value: float) -> str:
    """把带符号小数变成稳定的目录名片段。"""

    prefix = "m" if value < 0 else "p" if value > 0 else "z"
    return prefix + f"{abs(value):.2f}".replace(".", "p")


def condition_name(spec: PerturbationSpec) -> str:
    """生成明确包含物理含义和强度的条件名。"""

    if spec.name == "original":
        return "original"
    if spec.name in {"brightness", "contrast", "saturation"}:
        return f"{spec.name}_factor_{spec.value:.2f}".replace(".", "p")
    if spec.name == "hue":
        return f"hue_shift_{value_token(spec.value)}"
    if spec.name == "white_balance_temperature":
        direction = "warm" if spec.value > 0 else "cool"
        return f"temperature_{direction}_{abs(spec.value):.2f}".replace(".", "p")
    raise ValueError(f"未知颜色条件：{spec.name}")


def exact_mcnemar(baseline: dict[str, bool], current: dict[str, bool]) -> dict[str, Any]:
    """计算同一批原图扰动前后的配对得失与精确 McNemar p 值。"""

    if set(baseline) != set(current):
        raise RuntimeError("原始与扰动条件的原图集合不一致，不能做配对比较。")
    lost = sum(baseline[key] and not current[key] for key in baseline)
    gained = sum(not baseline[key] and current[key] for key in baseline)
    discordant = lost + gained
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(lost, gained) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "paired_lost_correct": lost,
        "paired_gained_correct": gained,
        "paired_discordant": discordant,
        "paired_mcnemar_exact_p": p_value,
    }


def build_report(
    config: ColorComponentConfig,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    """生成带保守判读边界的中文结果报告。"""

    baseline = rows[0]
    lines = [
        f"# 颜色分量独立扰动结果（{config.split}）",
        "",
        f"- Checkpoint：`{config.checkpoint}`",
        f"- 原始准确率：{baseline['parent_accuracy']:.2%}",
        f"- 原始 NLL：{baseline['parent_nll']:.4f}",
        "- 每个条件只改变一种颜色属性；白平衡实验对整图平均亮度进行了近似保持。",
        "",
        "| 条件 | 参数 | Accuracy | ΔAccuracy | NLL | ΔNLL | QWK | 丢失/新增答对 | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {value:+.2f} | {parent_accuracy:.2%} | "
            "{parent_accuracy_drop:+.2%} | {parent_nll:.4f} | {parent_nll_increase:+.4f} | "
            "{parent_qwk:.4f} | {paired_lost_correct}/{paired_gained_correct} | "
            "{paired_mcnemar_exact_p:.4f} |".format(**row)
        )

    component_labels = {
        "brightness": "亮度",
        "contrast": "对比度",
        "saturation": "饱和度",
        "hue": "色相",
        "white_balance_temperature": "白平衡/色温",
    }
    lines.extend(["", "## 分量级线索", ""])
    for component, label in component_labels.items():
        subset = [row for row in rows if row["perturbation"] == component]
        worst = max(subset, key=lambda row: (row["parent_accuracy_drop"], row["parent_nll_increase"]))
        lines.append(
            f"- {label}：最大准确率下降 {worst['parent_accuracy_drop']:+.2%}，"
            f"最大受影响条件为 `{worst['condition']}`；该条件 NLL 变化 "
            f"{worst['parent_nll_increase']:+.4f}。"
        )
    lines.extend(
        [
            "",
            "## 判读边界",
            "",
            "- 当前默认 val 只有 20 张原图，Accuracy 每张对应 5 个百分点；同时查看 NLL 和剂量趋势。",
            "- 一个方向单次掉点不能证明模型依赖该属性；更可信的是正负方向或强度增加时出现一致趋势。",
            "- 对比度、饱和度与色温在图像统计上不可能完全正交，本实验表示算子级单因素控制，而非生理机制分离。",
            "- 先在 val 固定最终强度和判读规则；不要根据 test 结果继续调强度。",
            "",
            f"逐条件指标和预测保存在：`{output_dir}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    config = CONFIG
    validate_config(config)
    if not config.checkpoint.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{config.checkpoint}")
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    model, metadata = load_checkpoint(
        config.checkpoint,
        device,
        model_name=config.model_name,
        num_classes=config.num_classes,
    )
    crop_size = config.crop_size or metadata.get("crop_size")
    input_size = config.input_size or metadata.get("input_size", 224)
    full_image = bool(config.full_image or metadata.get("full_image", False))
    if not crop_size and not full_image:
        raise ValueError("checkpoint 未记录 crop_size，请在 CONFIG 中手动填写。")
    views = config.val_views if config.split == "val" else config.test_views
    output_dir = config.output_dir.expanduser().resolve() / config.split
    write_json(
        output_dir / "experiment_config.json",
        {
            **vars(config),
            "views_resolved": views,
            "crop_size_resolved": crop_size,
            "input_size_resolved": input_size,
            "full_image_resolved": full_image,
            "device_resolved": str(device),
            "checkpoint_metadata": metadata,
        },
    )

    conditions = conditions_for_split(config)
    # dry-run 会覆盖五类算子，但每类只保留第一个条件和每类第一张图。
    if config.dry_run:
        first_by_name: dict[str, PerturbationSpec] = {}
        for condition in conditions:
            first_by_name.setdefault(condition.name, condition)
        conditions = list(first_by_name.values())

    rows: list[dict[str, Any]] = []
    baseline_accuracy = baseline_nll = None
    baseline_correctness: dict[str, bool] | None = None
    for spec in conditions:
        name = condition_name(spec)
        print(f"评估颜色条件：{name}")
        dataset = DiagnosticImageDataset(
            config.dataset_root,
            config.split,
            DiagnosticTransform(
                crop_size=1 if full_image else int(crop_size),
                input_size=int(input_size),
                training=False,
                view_count=views,
                perturbation=spec,
                seed=config.seed,
                full_image=full_image,
            ),
            DEFAULT_CLASSES,
            config.max_samples_per_class,
        )
        loader = make_loader(
            dataset,
            config.batch_size,
            config.num_workers,
            False,
            config.seed,
        )
        result = evaluate_model(model, loader, dataset, device)
        correctness = {
            row["parent_id"]: bool(row["correct"])
            for row in result["parent_predictions"]
        }
        if spec.name == "original":
            baseline_accuracy = result["parent"]["accuracy"]
            baseline_nll = result["parent"]["nll"]
            baseline_correctness = correctness
        assert baseline_accuracy is not None
        assert baseline_nll is not None
        assert baseline_correctness is not None
        row = metrics_row(name, result)
        row.update(
            {
                "perturbation": spec.name,
                "value": spec.value,
                "parent_accuracy_drop": baseline_accuracy - result["parent"]["accuracy"],
                "parent_nll_increase": result["parent"]["nll"] - baseline_nll,
                **exact_mcnemar(baseline_correctness, correctness),
            }
        )
        rows.append(row)
        condition_dir = output_dir / name
        write_json(
            condition_dir / "metrics.json",
            {"sample": result["sample"], "parent": result["parent"]},
        )
        write_csv(condition_dir / "sample_predictions.csv", result["sample_predictions"])
        write_csv(condition_dir / "parent_predictions.csv", result["parent_predictions"])

    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "summary.json", rows)
    report = build_report(config, rows, output_dir)
    (output_dir / "COLOR_COMPONENT_REPORT.md").write_text(report, encoding="utf-8")
    print(f"颜色分量实验完成：{output_dir / 'COLOR_COMPONENT_REPORT.md'}")


if __name__ == "__main__":
    main()
