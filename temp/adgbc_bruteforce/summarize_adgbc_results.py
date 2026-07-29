"""Summarize AD-GBC brute-force runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = THIS_DIR / "runs_BaSic"
# 通过 run_name 反解析 AD-GBC 的 K、tau、几何损失权重和训练策略。
RUN_RE = re.compile(
    r"^adgbc_(?P<model>.+)_k(?P<k>\d+)_tau(?P<tau>[0-9p]+)"
    r"_lw(?P<lambda_w>[0-9p]+)_bs(?P<beta_scale>[0-9p]+)"
    r"_(?P<strategy>finetune_all|warmup5_then_finetune|adgbc_head_only|head_only)"
    r"(?:_|$)"
)
BASELINE_RE = re.compile(r"^baseline_(?P<model>.+)_seed(?P<seed>\d+)(?:_|$)")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def parse_scalar(value: str) -> Any:
    text = value.strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        return text


def read_run_config(path: Path) -> dict[str, Any]:
    # 只读取顶层 run_name/random_seed，避免为汇总脚本额外引入 YAML 依赖。
    config: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line[0].isspace() or ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        if key in {"run_name", "random_seed"}:
            config[key] = parse_scalar(value)
    return config


def run_name_without_timestamp(run_name: str) -> str:
    match = re.match(r"^(.+)_20\d{6}_\d{6}(?:_\d{2})?$", run_name)
    return match.group(1) if match else run_name


def parse_slug_float(value: str) -> float:
    return float(value.replace("p", "."))


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_validation_value(metrics: dict[str, Any], key: str) -> Any:
    values = metrics.get("best_validation_metrics")
    return values.get(key) if isinstance(values, dict) else None


def identify(run_name: str, seed: int) -> dict[str, Any] | None:
    # baseline 和 AD-GBC 使用不同命名规则，先统一转成同一张结果表的字段。
    name = run_name_without_timestamp(run_name)
    baseline = BASELINE_RE.match(name)
    if baseline:
        return {
            "family": "baseline",
            "model_name": baseline.group("model"),
            "seed": int(baseline.group("seed")),
            "k": None,
            "tau": None,
            "lambda_w_div": None,
            "beta_scale_con": None,
            "training_strategy": "baseline",
        }

    match = RUN_RE.match(name)
    if not match:
        return None
    return {
        "family": "adgbc",
        "model_name": match.group("model"),
        "seed": seed,
        "k": int(match.group("k")),
        "tau": parse_slug_float(match.group("tau")),
        "lambda_w_div": parse_slug_float(match.group("lambda_w")),
        "beta_scale_con": parse_slug_float(match.group("beta_scale")),
        "training_strategy": match.group("strategy"),
    }


def collect_rows(runs_root: Path, include_duplicates: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("*/test_metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            continue
        metrics = read_json(metrics_path)
        config = read_run_config(config_path)
        run_name = str(config.get("run_name") or run_dir.name)
        seed = int(config.get("random_seed") or 2026)
        identity = identify(run_name, seed)
        if identity is None:
            continue

        row = {
            "run_name": run_name,
            **identity,
            "run_directory": str(run_dir),
            "run_mtime": run_dir.stat().st_mtime,
            "best_epoch": metrics.get("best_epoch"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "mae": metrics.get("mae"),
            "qwk": metrics.get("qwk"),
            "plus_minus_one_accuracy": metrics.get("plus_minus_one_accuracy"),
            "best_val_acc": best_validation_value(metrics, "val_acc"),
            "best_val_loss": best_validation_value(metrics, "val_loss"),
            "best_val_mae": best_validation_value(metrics, "val_mae"),
            "best_val_qwk": best_validation_value(metrics, "val_qwk"),
            "parameters_total": metrics.get("parameters_total"),
            "parameters_trainable": metrics.get("parameters_trainable"),
            "flops_g": metrics.get("flops_g"),
            "training_time_seconds": metrics.get("training_time_seconds"),
            "inference_ms_per_sample": metrics.get("inference_ms_per_sample"),
            "adgbc_loss_w_div": metrics.get("adgbc_loss_w_div"),
            "adgbc_loss_scale_con": metrics.get("adgbc_loss_scale_con"),
            "adgbc_assignment_entropy": metrics.get("adgbc_assignment_entropy"),
            "adgbc_center_norm_mean": metrics.get("adgbc_center_norm_mean"),
            "adgbc_scale_mean": metrics.get("adgbc_scale_mean"),
            "adgbc_scale_std": metrics.get("adgbc_scale_std"),
        }
        rows.append(row)

    if include_duplicates:
        return rows

    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("family"),
            row.get("model_name"),
            row.get("seed"),
            row.get("k"),
            row.get("tau"),
            row.get("lambda_w_div"),
            row.get("beta_scale_con"),
            row.get("training_strategy"),
        )
        current = latest.get(key)
        if current is None or float(row["run_mtime"]) > float(current["run_mtime"]):
            latest[key] = row
    return list(latest.values())


def with_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [
        {key: value for key, value in row.items() if key != "run_mtime"}
        for row in rows
    ]
    ranked.sort(
        key=lambda row: (
            row.get("family") != "adgbc",
            to_float(row.get("best_val_acc")) is None,
            -(to_float(row.get("best_val_acc")) or -1.0),
            -(to_float(row.get("best_val_qwk")) or -1.0),
            -(to_float(row.get("macro_f1")) or -1.0),
            to_float(row.get("flops_g")) or 999999.0,
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank_by_best_val_acc"] = rank
    return ranked


def group_summary(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    adgbc_rows = [row for row in rows if row.get("family") == "adgbc"]
    values = sorted({row.get(group_key) for row in adgbc_rows}, key=lambda item: str(item))
    summary: list[dict[str, Any]] = []
    for value in values:
        group = [row for row in adgbc_rows if row.get(group_key) == value]
        # 每个超参分组只取验证集表现最好的实验，方便快速筛冠军组合。
        best = max(
            group,
            key=lambda row: (
                to_float(row.get("best_val_acc")) or -1.0,
                to_float(row.get("best_val_qwk")) or -1.0,
                to_float(row.get("macro_f1")) or -1.0,
            ),
        )
        summary.append(
            {
                group_key: value,
                "run_count": len(group),
                "best_run_name": best.get("run_name"),
                "best_val_acc": best.get("best_val_acc"),
                "best_val_qwk": best.get("best_val_qwk"),
                "test_accuracy": best.get("accuracy"),
                "test_macro_f1": best.get("macro_f1"),
                "k": best.get("k"),
                "tau": best.get("tau"),
                "lambda_w_div": best.get("lambda_w_div"),
                "beta_scale_con": best.get("beta_scale_con"),
                "training_strategy": best.get("training_strategy"),
            }
        )
    summary.sort(
        key=lambda row: (
            to_float(row.get("best_val_acc")) is None,
            -(to_float(row.get("best_val_acc")) or -1.0),
            -(to_float(row.get("best_val_qwk")) or -1.0),
        )
    )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def summarize_results(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    include_duplicates: bool = False,
) -> list[Path]:
    runs_root = Path(runs_root)
    rows = collect_rows(runs_root, include_duplicates=include_duplicates)
    if not rows:
        print(f"No AD-GBC test_metrics.json files found under {runs_root}")
        return []

    ranked = with_ranks(rows)
    paths = [
        write_csv(runs_root / "adgbc_all_results.csv", ranked),
        write_csv(runs_root / "adgbc_summary_by_k.csv", group_summary(rows, "k")),
        write_csv(runs_root / "adgbc_summary_by_tau.csv", group_summary(rows, "tau")),
        write_csv(runs_root / "adgbc_summary_by_lambda_w.csv", group_summary(rows, "lambda_w_div")),
        write_csv(runs_root / "adgbc_summary_by_beta_scale.csv", group_summary(rows, "beta_scale_con")),
        write_csv(runs_root / "adgbc_summary_by_training_strategy.csv", group_summary(rows, "training_strategy")),
    ]
    written = [path for path in paths if path is not None]
    for path in written:
        print(f"Wrote {path}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize AD-GBC brute-force runs.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
    )
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="Keep duplicate logical runs instead of using the latest run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_results(args.runs_root, include_duplicates=args.include_duplicates)


if __name__ == "__main__":
    main()
