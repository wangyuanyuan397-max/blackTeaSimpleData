"""Summarize SupCon + Margin brute-force runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = THIS_DIR / "runs_BaSic"
# 从 run_name 中反解析本轮暴力搜索的所有关键超参。
RUN_RE = re.compile(
    r"^supm_(?P<model>.+)_m(?P<margin>[0-9p]+)_s(?P<scale>[0-9p]+)"
    r"_t(?P<temperature>[0-9p]+)_ls(?P<lambda_supcon>[0-9p]+)"
    r"_lr(?P<lr>[0-9p]+)_p(?P<projector_out>\d+)_(?P<classifier_feature>projected|raw)"
    r"(?:_|$)"
)
BASELINE_RE = re.compile(r"^baseline_(?P<model>.+)_ce_seed(?P<seed>\d+)(?:_|$)")


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
    # 只读取顶层 run_name/random_seed，避免给汇总脚本增加 YAML 依赖。
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
    # baseline 和 SupCon+Margin 使用不同命名规则，统一映射到一张表。
    name = run_name_without_timestamp(run_name)
    baseline = BASELINE_RE.match(name)
    if baseline:
        return {
            "family": "baseline",
            "model_name": baseline.group("model"),
            "seed": int(baseline.group("seed")),
            "margin": None,
            "scale": None,
            "temperature": None,
            "lambda_supcon": None,
            "lr": None,
            "projector_out": None,
            "classifier_feature": "baseline",
        }

    match = RUN_RE.match(name)
    if not match:
        return None
    return {
        "family": "supcon_margin",
        "model_name": match.group("model"),
        "seed": int(seed),
        "margin": parse_slug_float(match.group("margin")),
        "scale": parse_slug_float(match.group("scale")),
        "temperature": parse_slug_float(match.group("temperature")),
        "lambda_supcon": parse_slug_float(match.group("lambda_supcon")),
        "lr": parse_slug_float(match.group("lr")),
        "projector_out": int(match.group("projector_out")),
        "classifier_feature": match.group("classifier_feature"),
    }


def collect_rows(runs_root: Path, include_duplicates: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("*/test_metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            continue

        config = read_run_config(config_path)
        run_name = str(config.get("run_name") or run_dir.name)
        seed = int(config.get("random_seed") or 0)
        identity = identify(run_name, seed)
        if identity is None:
            continue

        metrics = read_json(metrics_path)
        row = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            **identity,
            "best_val_acc": best_validation_value(metrics, "accuracy"),
            "best_val_loss": best_validation_value(metrics, "loss"),
            "best_val_qwk": best_validation_value(metrics, "qwk"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "mae": metrics.get("mae"),
            "qwk": metrics.get("qwk"),
            "params_m": metrics.get("params_m"),
            "flops_g": metrics.get("flops_g"),
            "total_time_sec": metrics.get("total_time_sec"),
            "keep_pth_files": metrics.get("keep_pth_files"),
            "removed_pth_files": json.dumps(metrics.get("removed_pth_files", []), ensure_ascii=False),
        }
        rows.append(row)

    if include_duplicates:
        return rows

    # 同一逻辑 run 如果重复跑，只保留按目录名排序最后的一次。
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        logical_name = run_name_without_timestamp(str(row["run_name"]))
        current = latest.get(logical_name)
        if current is None or str(row["run_dir"]) > str(current["run_dir"]):
            latest[logical_name] = row
    return list(latest.values())


def with_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            to_float(row.get("best_val_acc")) is None,
            -(to_float(row.get("best_val_acc")) or -1.0),
            -(to_float(row.get("best_val_qwk")) or -1.0),
            -(to_float(row.get("macro_f1")) or -1.0),
            to_float(row.get("flops_g")) or 999999.0,
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank_by_best_val_acc"] = rank
    return ranked


def group_summary(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    supcon_rows = [row for row in rows if row.get("family") == "supcon_margin"]
    values = sorted({row.get(group_key) for row in supcon_rows}, key=lambda item: str(item))
    summary: list[dict[str, Any]] = []
    for value in values:
        group = [row for row in supcon_rows if row.get(group_key) == value]
        # 每个超参取验证集表现最好的实验，方便快速筛冠军组合。
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
                "margin": best.get("margin"),
                "scale": best.get("scale"),
                "temperature": best.get("temperature"),
                "lambda_supcon": best.get("lambda_supcon"),
                "lr": best.get("lr"),
                "projector_out": best.get("projector_out"),
                "classifier_feature": best.get("classifier_feature"),
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
        print(f"No SupCon+Margin test_metrics.json files found under {runs_root}")
        return []

    ranked = with_ranks(rows)
    paths = [
        write_csv(runs_root / "supcon_margin_all_results.csv", ranked),
        write_csv(runs_root / "supcon_margin_summary_by_margin.csv", group_summary(rows, "margin")),
        write_csv(runs_root / "supcon_margin_summary_by_scale.csv", group_summary(rows, "scale")),
        write_csv(runs_root / "supcon_margin_summary_by_temperature.csv", group_summary(rows, "temperature")),
        write_csv(runs_root / "supcon_margin_summary_by_lambda_supcon.csv", group_summary(rows, "lambda_supcon")),
        write_csv(runs_root / "supcon_margin_summary_by_lr.csv", group_summary(rows, "lr")),
        write_csv(runs_root / "supcon_margin_summary_by_projector_out.csv", group_summary(rows, "projector_out")),
        write_csv(runs_root / "supcon_margin_summary_by_classifier_feature.csv", group_summary(rows, "classifier_feature")),
    ]
    written = [path for path in paths if path is not None]
    for path in written:
        print(f"Wrote {path}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SupCon+Margin brute-force results.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
    )
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="Keep repeated timestamped runs instead of deduplicating logical names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_results(args.runs_root, include_duplicates=args.include_duplicates)


if __name__ == "__main__":
    main()
