"""审计文件名首位编号与数据划分、模型错误之间的关系。

当前原图命名形如 ``4-1.bmp``。代码只能确认首位编号形成 1～6 六个规则组，
不能仅凭文件名证明它就是茶叶批次、采集轮次或样品来源。因此报告统一称为
``source_group（来源组假设）``，业务语义必须回查实验记录后才能确认。
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
ARCHIVED_RESULTS_ROOT = Path(
    r"E:\docs\服务器跑模型结果备份\data01234\grid裁剪30+basic处理的模型结果"
    r"\problem_diagnosis_experiments\results"
)
# 当前电脑优先复用用户给出的服务器结果备份；换到其他机器时自动回退到本目录结果。
DEFAULT_RESULTS_ROOT = (
    ARCHIVED_RESULTS_ROOT
    if ARCHIVED_RESULTS_ROOT.is_dir()
    else EXPERIMENT_DIR / "results"
)


# =============================================================================
# PyCharm 右键运行配置区
# 先完成尺度实验，然后直接右键运行本文件，不需要命令行参数。
# =============================================================================
@dataclass(frozen=True)
class SourceGroupAuditConfig:
    """来源组审计配置。"""

    manifest_path: Path = (
        PROJECT_ROOT / "datasets_01234_original_split" / "source_split_manifest.csv"
    )
    crop_results_dir: Path = DEFAULT_RESULTS_ROOT / "crop_scale"
    output_dir: Path = EXPERIMENT_DIR / "results" / "source_group_audit"

    # ``4-2`` 中 group_regex 的第一组捕获 ``4``，第二组捕获 ``2``。
    group_regex: str = r"^(\d+)-(\d+)$"
    semantic_status: str = "来源组假设：具体是否为批次/采集轮次，需回查实验记录"


CONFIG = SourceGroupAuditConfig()


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 UTF-8 或 UTF-8-BOM CSV。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """保存 Excel 可直接打开的 UTF-8-BOM CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_source_stem(stem: str, pattern: re.Pattern[str]) -> tuple[str, str]:
    """把 ``4-2`` 解析为来源组 4、组内重复 2。"""

    match = pattern.fullmatch(str(stem))
    if not match or len(match.groups()) < 2:
        raise ValueError(f"文件名不符合 CONFIG.group_regex：{stem!r}")
    return match.group(1), match.group(2)


def parent_stem(parent_id: str) -> str:
    """从 ``20/4-2`` 或 Windows 路径中提取 ``4-2``。"""

    return str(parent_id).replace("\\", "/").rsplit("/", 1)[-1]


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """计算 2×2 Fisher 精确检验双侧 p 值，无需 scipy。

    表格定义为：
    ``[[组内错误, 组内正确], [其他组错误, 其他组正确]]``。
    """

    row_one = a + b
    row_two = c + d
    error_total = a + c
    total = row_one + row_two
    if total <= 0:
        return 1.0

    def probability(x: int) -> float:
        return (
            math.comb(error_total, x)
            * math.comb(total - error_total, row_one - x)
            / math.comb(total, row_one)
        )

    lower = max(0, row_one - (total - error_total))
    upper = min(row_one, error_total)
    observed = probability(a)
    return min(
        1.0,
        sum(
            probability(x)
            for x in range(lower, upper + 1)
            if probability(x) <= observed + 1e-15
        ),
    )


def add_bh_q_values(rows: list[dict[str, Any]]) -> None:
    """原地添加 Benjamini-Hochberg FDR q 值，控制多重比较。"""

    if not rows:
        return
    ranked = sorted(
        enumerate(rows),
        key=lambda item: float(item[1]["fisher_two_sided_p"]),
    )
    count = len(ranked)
    adjusted = [1.0] * count
    running_minimum = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index, row = ranked[rank_index]
        rank = rank_index + 1
        candidate = min(1.0, float(row["fisher_two_sided_p"]) * count / rank)
        running_minimum = min(running_minimum, candidate)
        adjusted[original_index] = running_minimum
    for row, q_value in zip(rows, adjusted):
        row["bh_fdr_q"] = q_value


def condition_name(prediction_path: Path, crop_results_dir: Path) -> str:
    """把预测文件相对路径转换成与尺度汇总一致的条件名。"""

    relative = prediction_path.relative_to(crop_results_dir)
    parts = relative.parts
    experiment = parts[0]
    if experiment.startswith("fusion_"):
        return experiment
    if experiment == "crop_global":
        return "global"
    if experiment.startswith("crop_") and len(parts) >= 3:
        scale = experiment.removeprefix("crop_")
        evaluation = parts[1]
        if evaluation == "test_single":
            return f"crop_{scale}_single"
        if evaluation.startswith("test_multi"):
            view_count = evaluation.removeprefix("test_multi")
            return f"crop_{scale}_multi{view_count}"
    return relative.as_posix().removesuffix("/parent_predictions.csv")


def audit_manifest(config: SourceGroupAuditConfig, pattern: re.Pattern[str]):
    """审计每个来源组在类别和 train/val/test 中的分布。"""

    rows = read_csv(config.manifest_path)
    enriched = []
    for row in rows:
        group, replicate = parse_source_stem(row["source_stem"], pattern)
        enriched.append({**row, "source_group": group, "group_replicate": replicate})

    groups = sorted({row["source_group"] for row in enriched})
    splits = ("train", "val", "test")
    classes = sorted({row["time_code"] for row in enriched})
    output_rows: list[dict[str, Any]] = []
    for group in groups:
        subset = [row for row in enriched if row["source_group"] == group]
        record: dict[str, Any] = {
            "source_group": group,
            "total_images": len(subset),
            "appears_in_train": int(any(row["split"] == "train" for row in subset)),
            "appears_in_val": int(any(row["split"] == "val" for row in subset)),
            "appears_in_test": int(any(row["split"] == "test" for row in subset)),
        }
        for split in splits:
            split_subset = [row for row in subset if row["split"] == split]
            record[f"{split}_count"] = len(split_subset)
            for class_name in classes:
                record[f"{split}_{class_name}_count"] = sum(
                    row["time_code"] == class_name for row in split_subset
                )
        for class_name in classes:
            record[f"all_{class_name}_count"] = sum(
                row["time_code"] == class_name for row in subset
            )
        output_rows.append(record)
    return enriched, output_rows, groups, classes


def audit_predictions(
    config: SourceGroupAuditConfig,
    pattern: re.Pattern[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """逐条件统计各来源组错误率及相对其他组的 Fisher 检验。"""

    prediction_paths = sorted(
        config.crop_results_dir.glob("**/parent_predictions.csv")
    )
    if not prediction_paths:
        raise FileNotFoundError(
            f"没有找到 parent_predictions.csv：{config.crop_results_dir}"
        )

    metric_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for path in prediction_paths:
        condition = condition_name(path, config.crop_results_dir)
        predictions = read_csv(path)
        by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in predictions:
            group, replicate = parse_source_stem(parent_stem(row["parent_id"]), pattern)
            enriched = {
                **row,
                "condition": condition,
                "source_group": group,
                "group_replicate": replicate,
            }
            by_group[group].append(enriched)
            if int(row["label"]) != int(row["prediction"]):
                error_rows.append(enriched)

        for group in sorted(by_group):
            group_rows = by_group[group]
            other_rows = [
                row
                for other_group, rows in by_group.items()
                if other_group != group
                for row in rows
            ]
            group_errors = sum(
                int(row["label"]) != int(row["prediction"]) for row in group_rows
            )
            other_errors = sum(
                int(row["label"]) != int(row["prediction"]) for row in other_rows
            )
            group_correct = len(group_rows) - group_errors
            other_correct = len(other_rows) - other_errors
            metric_rows.append(
                {
                    "condition": condition,
                    "source_group": group,
                    "sample_count": len(group_rows),
                    "correct_count": group_correct,
                    "error_count": group_errors,
                    "accuracy": group_correct / len(group_rows),
                    "error_rate": group_errors / len(group_rows),
                    "other_sample_count": len(other_rows),
                    "other_error_rate": other_errors / len(other_rows),
                    "error_rate_difference": (
                        group_errors / len(group_rows)
                        - other_errors / len(other_rows)
                    ),
                    "fisher_two_sided_p": fisher_two_sided(
                        group_errors,
                        group_correct,
                        other_errors,
                        other_correct,
                    ),
                }
            )
    # 把所有“条件 × 来源组”视为同一个探索性检验家族，统一做 FDR 校正。
    add_bh_q_values(metric_rows)
    return metric_rows, error_rows


def build_report(
    config: SourceGroupAuditConfig,
    manifest_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    groups: list[str],
    classes: list[str],
) -> str:
    """生成保守、可审计的中文结论。"""

    all_balanced = all(
        len({row[f"all_{name}_count"] for name in classes}) == 1
        for row in manifest_rows
    )
    leaked_groups = [
        row["source_group"]
        for row in manifest_rows
        if row["appears_in_train"] and row["appears_in_val"] and row["appears_in_test"]
    ]
    significant = [
        row
        for row in metric_rows
        if row["bh_fdr_q"] < 0.05 and row["error_rate_difference"] > 0
    ]
    significant.sort(key=lambda row: (row["bh_fdr_q"], -row["error_rate_difference"]))

    lines = [
        "# 文件名来源组审计",
        "",
        f"> 语义状态：{config.semantic_status}",
        "",
        "## 1. 可直接确认的事实",
        "",
        f"- 文件名可以稳定解析为 {len(groups)} 个首位组：{', '.join(groups)}。",
        f"- 每个时间点包含来源组结构；类别为：{', '.join(classes)}。",
        f"- 各来源组在所有类别中的数量是否平衡：{'是' if all_balanced else '否'}。",
        f"- 同时跨越 train/val/test 的来源组：{', '.join(leaked_groups) or '无'}。",
        "",
        "如果首位组经实验记录确认是批次、采集轮次或样品来源，那么当前随机划分存在组身份跨划分泄漏；此时必须增加 group-held-out 实验。",
        "",
        "## 2. 预测错误与来源组",
        "",
    ]
    if significant:
        lines.append("以下是对全部‘条件 × 来源组’统一做 BH-FDR 校正后 q<0.05 的错误富集信号：")
        lines.append("")
        lines.append("| 条件 | 来源组 | 样本数 | 错误率 | 其他组错误率 | 差值 | p | BH q |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in significant[:20]:
            lines.append(
                "| {condition} | {source_group} | {sample_count} | {error_rate:.1%} | "
                "{other_error_rate:.1%} | {error_rate_difference:+.1%} | "
                "{fisher_two_sided_p:.4f} | {bh_fdr_q:.4f} |".format(**row)
            )
    else:
        lines.append("未发现 BH-FDR 校正后 q<0.05 的来源组错误富集。")
    lines.extend(
        [
            "",
            "## 3. 使用边界",
            "",
            "- 文件名规则只能形成待核实的来源组，不能自动赋予‘批次’含义。",
            "- 测试集只有 25 张；来源组内样本更少，p 值和准确率均非常离散。",
            "- 已统一做 BH-FDR 校正；但组内样本仍很少，显著项也只能作为复现实验线索。",
            "- 只有确认组语义后，才应运行 leave-one-group-out 或 group-held-out 训练。",
            "",
            f"共审计 {len(metric_rows)} 个‘条件×来源组’单元，记录 {len(error_rows)} 条错误。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    config = CONFIG
    if not config.manifest_path.is_file():
        raise FileNotFoundError(f"找不到 manifest：{config.manifest_path}")
    if not config.crop_results_dir.is_dir():
        raise FileNotFoundError(f"找不到尺度结果目录：{config.crop_results_dir}")
    pattern = re.compile(config.group_regex)
    enriched, manifest_rows, groups, classes = audit_manifest(config, pattern)
    metric_rows, error_rows = audit_predictions(config, pattern)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(config.output_dir / "manifest_with_source_groups.csv", enriched)
    write_csv(config.output_dir / "group_split_counts.csv", manifest_rows)
    write_csv(config.output_dir / "condition_group_metrics.csv", metric_rows)
    write_csv(config.output_dir / "condition_group_errors.csv", error_rows)
    report = build_report(
        config,
        manifest_rows,
        metric_rows,
        error_rows,
        groups,
        classes,
    )
    (config.output_dir / "SOURCE_GROUP_AUDIT.md").write_text(report, encoding="utf-8")
    (config.output_dir / "audit_config.json").write_text(
        json.dumps(
            {
                **vars(config),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"来源组审计完成：{config.output_dir / 'SOURCE_GROUP_AUDIT.md'}")


if __name__ == "__main__":
    main()
