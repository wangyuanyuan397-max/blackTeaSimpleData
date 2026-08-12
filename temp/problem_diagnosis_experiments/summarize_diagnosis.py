"""汇总 crop 和扰动实验，生成带保守判读规则的中文报告。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


EXPERIMENT_DIR = Path(__file__).resolve().parent


# =============================================================================
# PyCharm 右键运行配置区
# 默认读取前两个脚本的标准输出目录，无需任何命令行参数。
# =============================================================================
CROP_SUMMARY = EXPERIMENT_DIR / "results" / "crop_scale" / "summary.csv"
PERTURBATION_SUMMARY = EXPERIMENT_DIR / "results" / "perturbation" / "summary.csv"
REPORT_OUTPUT = EXPERIMENT_DIR / "results" / "DIAGNOSIS_REPORT.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 UTF-8/UTF-8-BOM CSV。"""

    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    """读取数值列，缺失时返回 NaN。"""

    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """生成紧凑 Markdown 表。"""

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def summarize_crop(rows: list[dict[str, str]]) -> tuple[list[str], dict]:
    """分析单尺度结果；旧测试集枚举融合只标为探索性结果。"""

    lines = ["## 1. Crop 尺度与尺度互补", ""]
    if not rows:
        lines.append("未找到 crop 实验结果。")
        return lines, {}
    table_rows = []
    single_views, multi_views, fusions = [], [], []
    for row in rows:
        accuracy = number(row, "parent_accuracy")
        table_rows.append(
            [
                row.get("condition", ""),
                percent(number(row, "sample_accuracy")),
                percent(accuracy),
                f"{number(row, 'parent_qwk'):.3f}",
                f"{number(row, 'parent_mae'):.3f}",
            ]
        )
        condition = row.get("condition", "")
        if condition.startswith("fusion_"):
            fusions.append(row)
        elif "_multi" in condition:
            multi_views.append(row)
        else:
            single_views.append(row)
    lines.append(markdown_table(["条件", "Patch Acc", "原图 Acc", "原图 QWK", "原图 MAE"], table_rows))
    lines.append("")
    diagnosis = {}
    if single_views:
        best_single = max(single_views, key=lambda row: number(row, "parent_accuracy"))
        diagnosis["best_single"] = best_single.get("condition")
        diagnosis["best_single_accuracy"] = number(best_single, "parent_accuracy")
        lines.append(
            f"最佳单尺度是 `{best_single.get('condition')}`，原图级准确率为 "
            f"{percent(diagnosis['best_single_accuracy'])}。"
        )
    if multi_views:
        best_multi = max(multi_views, key=lambda row: number(row, "parent_accuracy"))
        diagnosis["best_multi"] = best_multi.get("condition")
        diagnosis["best_multi_accuracy"] = number(best_multi, "parent_accuracy")
        lines.append(
            f"最佳多局部条件是 `{best_multi.get('condition')}`，原图级准确率为 "
            f"{percent(diagnosis['best_multi_accuracy'])}。"
        )
    standalone = single_views + multi_views
    if fusions and standalone:
        best_fusion = max(fusions, key=lambda row: number(row, "parent_accuracy"))
        fusion_accuracy = number(best_fusion, "parent_accuracy")
        best_standalone_accuracy = max(number(row, "parent_accuracy") for row in standalone)
        gain = fusion_accuracy - best_standalone_accuracy
        diagnosis.update(
            {
                "best_fusion": best_fusion.get("condition"),
                "best_fusion_accuracy": fusion_accuracy,
                "fusion_gain": gain,
            }
        )
        if gain > 0:
            judgment = "存在待复核的多尺度互补线索"
        else:
            judgment = "本次探索未观察到正向融合增益"
        lines.append(
            f"旧流程在测试集枚举得到的最佳融合 `{best_fusion.get('condition')}` "
            f"相对最佳非融合条件变化 {gain * 100:+.2f} 个百分点：**{judgment}**。"
        )
        lines.append(
            "该组合由测试集反向选出，不能作为无偏性能估计；正式结论必须改用 "
            "`select_fusion_on_validation.py` 的验证集冻结流程。"
        )
        diagnosis["fusion_selection_warning"] = "legacy_test_selected_exploratory_only"
        diagnosis["judgment"] = judgment
    return lines, diagnosis


def summarize_perturbation(rows: list[dict[str, str]]) -> tuple[list[str], dict]:
    """按扰动参数聚合随机重复，避免只挑最好或最差一次。"""

    lines = ["## 2. 受控破坏实验", ""]
    if not rows:
        lines.append("未找到扰动实验结果。")
        return lines, {}
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("perturbation", ""),
            row.get("value", ""),
            row.get("grid", ""),
            row.get("micro_grid", ""),
        )
        grouped[key].append(row)
    table_rows, diagnosis = [], {}
    for key, group in grouped.items():
        accuracies = [number(row, "parent_accuracy") for row in group]
        drops = [number(row, "parent_accuracy_drop") for row in group]
        label = key[0]
        if key[0] in {"occlusion", "blur", "color_jitter"}:
            label += f"({key[1]})"
        elif key[0] == "patch_shuffle":
            label += f"({key[2]}×{key[2]})"
        elif key[0] == "texture_shuffle":
            label += f"(macro={key[2]}, micro={key[3]})"
        table_rows.append(
            [
                label,
                str(len(group)),
                percent(mean(accuracies)),
                f"{100 * mean(drops):+.2f} pp",
                f"{100 * pstdev(drops):.2f} pp" if len(drops) > 1 else "—",
            ]
        )
        diagnosis[label] = {
            "repeat_count": len(group),
            "mean_parent_accuracy": mean(accuracies),
            "mean_accuracy_drop": mean(drops),
            "std_accuracy_drop": pstdev(drops) if len(drops) > 1 else 0.0,
        }
    lines.append(markdown_table(["条件", "重复", "原图 Acc 均值", "相对下降", "下降标准差"], table_rows))
    lines.extend(
        [
            "",
            "判读边界：单次掉点不能直接证明机制；建议关注重复后仍稳定下降的条件，并与 crop、融合证据联合判断。",
        ]
    )
    return lines, diagnosis


def main() -> None:
    crop_lines, crop_diagnosis = summarize_crop(read_csv(CROP_SUMMARY))
    perturbation_lines, perturbation_diagnosis = summarize_perturbation(
        read_csv(PERTURBATION_SUMMARY)
    )
    report = [
        "# 红茶五分类任务诊断报告",
        "",
        "> 正式结论优先采用原图聚合级指标；patch 级指标仅用于定位。",
        "",
        *crop_lines,
        "",
        *perturbation_lines,
        "",
        "## 3. 结论使用约束",
        "",
        "- 一个模块涨点不等于一个机制成立。",
        "- 一个假设至少需要 2～4 条互补证据；本报告只自动整理证据，不替代统计复核。",
        "- test 集只用于最终一次确认；开发阶段应优先在 val 集完成方案选择。",
        "- 正式比较建议补 3 个随机种子，并报告均值、标准差和配对原图结果。",
        "",
    ]
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text("\n".join(report), encoding="utf-8")
    diagnosis_json = REPORT_OUTPUT.with_suffix(".json")
    diagnosis_json.write_text(
        json.dumps(
            {"crop": crop_diagnosis, "perturbation": perturbation_diagnosis},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"报告已生成：{REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
