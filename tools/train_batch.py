"""使用现成 train/val/test 目录顺序训练多个模型并生成本地 HTML 报告。"""

import argparse
import copy
import gc
import html
import json
import re
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_CONFIG = PROJECT_ROOT / "configs" / "fixed_split_batch.yaml"
CONFIG_LIST = (
    Path('configs/full6fold_server_b16a2/baselines/safnet_strict_ce_scratch.yaml'),
    Path('configs/full6fold_server_b16a2/baselines/safnet_imagenet_strict_ce.yaml'),
    Path('configs/full6fold_server_b16a2/baselines/mambaout_tiny_strict_ce.yaml'),
    Path('configs/full6fold_server_b16a2/baselines/resnet50_strict_ce.yaml'),
    Path('configs/full6fold_server_b16a2/baselines/mobilenet_v3_large_strict_ce.yaml'),
    Path('configs/full6fold_server_b16a2/baselines/efficientnetv2_s_strict_ce.yaml'),
    Path('configs/full6fold_server_b16a2/baselines/efficientnetv2_s_softlabel_eps007.yaml'),
    Path('configs/full6fold_server_b16a2/baselines/efficientnetv2_s_strict_ce_with_sfi_regression_lambda10.yaml'),
    Path('configs/full6fold_server_b16a2/baselines/efficientnetv2_s_softlabel_eps007_with_sfi_regression_lambda10.yaml'),
    Path('configs/full6fold_server_b16a2/baselines/convnext_tiny_strict_ce.yaml'),
)
DATASET_ROOT = Path('datasets_split_patches')
RUNS_ROOT = Path('runs')
FIXED_CLASS_TO_IDX = {'pre': 0, 'slight': 1, 'moderate': 2, 'over': 3}

# 当前任务只比较不同模型结构，不继承旧 YAML 中的评价体系和损失变体。
MODEL_LIST = (
    {
        'name': 'safnet_scratch',
        'backbone': {'type': 'safnet', 'pretrained': False, 'num_classes': 4, 'reduction': 16},
        'head': {'type': 'identity', 'drop_rate': 0.0},
    },
    {
        'name': 'safnet_imagenet',
        'backbone': {'type': 'safnet', 'pretrained': True, 'num_classes': 4, 'reduction': 16},
        'head': {'type': 'identity', 'drop_rate': 0.0},
    },
    {
        'name': 'mambaout_tiny',
        'backbone': {
            'type': 'mambaout_tiny_ce',
            'model_name': 'mambaout_tiny.in1k',
            'pretrained': True,
            'num_classes': 4,
        },
        'head': {'type': 'identity', 'drop_rate': 0.0},
    },
    {
        'name': 'resnet50',
        'backbone': {'type': 'torchvision', 'model_name': 'resnet50', 'pretrained': True},
        'head': {'type': 'linear', 'drop_rate': 0.0},
    },
    {
        'name': 'mobilenet_v3_large',
        'backbone': {'type': 'torchvision', 'model_name': 'mobilenet_v3_large', 'pretrained': True},
        'head': {'type': 'linear', 'drop_rate': 0.0},
    },
    {
        'name': 'efficientnet_v2_s',
        'backbone': {'type': 'torchvision', 'model_name': 'efficientnet_v2_s', 'pretrained': True},
        'head': {'type': 'linear', 'drop_rate': 0.0},
    },
    {
        'name': 'convnext_tiny',
        'backbone': {'type': 'torchvision', 'model_name': 'convnext_tiny', 'pretrained': True},
        'head': {'type': 'linear', 'drop_rate': 0.0},
    },
)

# 所有模型共享同一套当前任务设置，数据不会重新划分，也不执行旧式滑窗投票或额外分析。
CURRENT_TASK_SETTINGS = {
    'random_seed': 42,
    'class_to_idx': FIXED_CLASS_TO_IDX,
    'data': {
        'type': 'image_folder',
        'train_transform': {'type': 'patch_train_224', 'image_size': 224},
        'eval_transform': {'type': 'patch_eval_224', 'image_size': 224},
        'test_transform': {'type': 'patch_eval_224', 'image_size': 224},
    },
    'train': {
        'epochs': 30,
        'batch_size': 32,
        'val_batch_size': 64,
        'test_batch_size': 64,
        'accumulation_steps': 1,
        'num_workers': 4,
        'weighted_sampler': False,
        'patience': 8,
        'selection_metric': 'val_acc',
        'selection_mode': 'max',
        'selection_min_delta': 0.0,
        'enable_error_analysis': False,
        'tta_mode': 'mean',
        'tta_topk': 0,
        'tta_compare': False,
    },
    'optimizer': {'type': 'adamw', 'lr': 0.0001, 'weight_decay': 0.0005},
    'scheduler': {'type': 'cosine', 'warmup_epochs': 2, 'min_lr': 0.000001},
    'loss': {'type': 'cross_entropy', 'label_smoothing': 0.0},
}


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models  # noqa: E402,F401 - 导入时完成模型、骨干网络和损失函数注册。
from src.engine import ComponentBuilder, Trainer  # noqa: E402
from src.schemas import TrainingConfig  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """解析批量训练命令行参数。"""
    parser = argparse.ArgumentParser(
        description="直接使用 datasets_split_patches 中固定的 train/val/test 顺序训练多个模型。"
    )
    parser.add_argument(
        "--batch-config",
        type=Path,
        default=DEFAULT_BATCH_CONFIG,
        help=f"批量训练 YAML（默认：{DEFAULT_BATCH_CONFIG}）。",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="只运行列表中的模型，例如：--models resnet50 mobilenet_v3_large。",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="训练设备（默认：auto，优先使用 CUDA）。",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="只显示 YAML 中可运行的模型列表。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查配置、固定数据目录和一个样本，不创建 runs 或训练模型。",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="任一模型失败后立即停止；默认记录失败报告并继续下一个模型。",
    )
    return parser.parse_args()


def load_batch_config(config_path: Path) -> Dict[str, Any]:
    """读取并执行批量配置的基础结构检查。"""
    resolved_path = config_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"批量配置不存在：{resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("批量配置顶层必须是 YAML 字典。")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("批量配置必须包含非空 models 列表。")
    for index, model_spec in enumerate(models, start=1):
        if not isinstance(model_spec, dict) or not model_spec.get("name"):
            raise ValueError(f"models 中第 {index} 项必须包含 name。")
        if not isinstance(model_spec.get("backbone"), dict):
            raise ValueError(f"模型 {model_spec['name']} 缺少 backbone 字典。")
    return config


def resolve_project_path(value: str | Path) -> Path:
    """把 YAML 中的相对路径统一解释为相对于项目根目录。"""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def select_models(
    model_specs: Iterable[Dict[str, Any]], requested_names: Optional[List[str]]
) -> List[Dict[str, Any]]:
    """按 YAML 顺序或命令行指定顺序选择需要训练的模型。"""
    enabled_specs = [spec for spec in model_specs if spec.get("enabled", True)]
    by_name = {str(spec["name"]): spec for spec in enabled_specs}
    if len(by_name) != len(enabled_specs):
        raise ValueError("models 列表中存在重复的模型 name。")
    if not requested_names:
        return enabled_specs
    unknown_names = [name for name in requested_names if name not in by_name]
    if unknown_names:
        raise ValueError(f"--models 包含未知或已禁用模型：{unknown_names}")
    if len(set(requested_names)) != len(requested_names):
        raise ValueError("--models 中不能重复指定同一个模型。")
    return [by_name[name] for name in requested_names]


def resolve_device(device_name: str) -> torch.device:
    """根据命令行选项确定实际训练设备。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已指定 --device cuda，但当前环境没有可用 CUDA。")
    return torch.device(device_name)


def safe_run_name(name: str) -> str:
    """把模型名转换成适合 Windows 文件夹的安全名称。"""
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("._-")
    return cleaned or "model"


def create_unique_run_directory(runs_root: Path, model_name: str) -> Path:
    """创建“模型名_时间戳”目录，并在极少见的同秒重名时追加序号。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{safe_run_name(model_name)}_{timestamp}"
    candidate = runs_root / base_name
    suffix = 2
    while candidate.exists():
        candidate = runs_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def build_training_config(
    batch_config: Dict[str, Any],
    model_spec: Dict[str, Any],
    dataset_root: Path,
    run_directory: Path,
    device: torch.device,
) -> TrainingConfig:
    """把共享训练参数和单个模型定义合成为 Trainer 可直接使用的配置。"""
    data_config = copy.deepcopy(batch_config.get("data") or {})
    data_config["root"] = str(dataset_root)
    data_config.setdefault("type", "image_folder")
    data_config["class_to_idx"] = copy.deepcopy(
        batch_config.get("class_to_idx")
        or {"pre": 0, "slight": 1, "moderate": 2, "over": 3}
    )

    train_config = copy.deepcopy(batch_config.get("train") or {})
    train_config["device"] = device.type
    train_config.setdefault("enable_error_analysis", False)

    model_config = {
        "type": model_spec.get("type", "classifier"),
        "strategy": model_spec.get("strategy", "classification"),
        "backbone": copy.deepcopy(model_spec["backbone"]),
        "head": copy.deepcopy(
            model_spec.get("head") or {"type": "linear", "drop_rate": 0.0}
        ),
    }
    for optional_key in ("return_embeddings", "neck", "aux_head"):
        if optional_key in model_spec:
            model_config[optional_key] = copy.deepcopy(model_spec[optional_key])

    runtime_config = {
        "run_name": str(model_spec["name"]),
        "description": "固定 train/val/test patch 数据集批量训练",
        "output_dir": str(run_directory),
        "random_seed": int(batch_config.get("random_seed", 42)),
        "use_wandb": False,
        "enable_google_drive_upload": False,
        "data": data_config,
        "model": model_config,
        "train": train_config,
        "optimizer": copy.deepcopy(batch_config.get("optimizer") or {}),
        "scheduler": copy.deepcopy(batch_config.get("scheduler")),
        "loss": copy.deepcopy(batch_config.get("loss") or {"type": "cross_entropy"}),
    }
    return TrainingConfig(**runtime_config)


def dataset_counts(dataset) -> Dict[str, int]:
    """按数据集公开的 classes 和 targets 统计每类样本数。"""
    counts_by_index = Counter(int(target) for target in dataset.targets)
    return {
        class_name: counts_by_index.get(class_index, 0)
        for class_index, class_name in enumerate(dataset.classes)
    }


def validate_fixed_dataset(config: TrainingConfig, device: torch.device) -> Dict[str, Any]:
    """构建固定的三个 DataLoader，并实际读取每个集合的第一张图片。"""
    builder = ComponentBuilder(config, device, logger=None)
    train_loader, val_loader, test_loader = builder.build_dataloaders()
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    summary: Dict[str, Any] = {}
    expected_classes = None
    for split_name, loader in loaders.items():
        dataset = loader.dataset
        classes = list(dataset.classes)
        if expected_classes is None:
            expected_classes = classes
        elif classes != expected_classes:
            raise ValueError(
                f"{split_name} 类别顺序 {classes} 与训练集 {expected_classes} 不一致。"
            )
        sample_image, sample_label, sample_path = dataset[0]
        summary[split_name] = {
            "total": len(dataset),
            "classes": dataset_counts(dataset),
            "sample_shape": list(sample_image.shape),
            "sample_label": int(sample_label),
            "sample_path": str(sample_path),
        }
    return summary


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """兼容不同 PyTorch 版本读取最佳模型检查点。"""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model_state_dict": checkpoint}


def to_builtin(value: Any) -> Any:
    """把 NumPy、Tensor 和 Path 等对象递归转换成 JSON 可序列化类型。"""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def evaluate_best_checkpoint(trainer: Trainer) -> Dict[str, Any]:
    """加载验证集选出的最佳权重，并只在固定测试集上执行最终评估。"""
    checkpoint_path = trainer.output_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"训练结束后未找到最佳模型：{checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, trainer.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    trainer.model.load_state_dict(state_dict)
    trainer.model.eval()

    test_metrics = trainer.evaluator.evaluate(
        trainer.test_loader,
        trainer.loss_fn,
        desc="Testing best checkpoint",
    )
    class_names = list(trainer.test_loader.dataset.classes)
    confusion_matrix = trainer.evaluator.compute_confusion_matrix(
        trainer.test_loader,
        num_classes=len(class_names),
    )
    per_class_accuracy: Dict[str, float] = {}
    for class_index, class_name in enumerate(class_names):
        class_total = int(confusion_matrix[class_index, :].sum())
        class_correct = int(confusion_matrix[class_index, class_index])
        per_class_accuracy[class_name] = class_correct / class_total if class_total else 0.0

    result = to_builtin(test_metrics)
    result.update(
        {
            "class_names": class_names,
            "per_class_accuracy": per_class_accuracy,
            "confusion_matrix": to_builtin(confusion_matrix),
            "num_samples": int(confusion_matrix.sum()),
            "best_epoch": int(checkpoint.get("epoch", -1)) + 1,
            "best_validation_metrics": to_builtin(checkpoint.get("metrics", {})),
        }
    )
    return result


def save_json(path: Path, data: Any) -> None:
    """使用 UTF-8 和便于阅读的缩进保存 JSON。"""
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(data), file, ensure_ascii=False, indent=2)


def format_metric(value: Any) -> str:
    """把报告中的数字统一格式化为便于阅读的字符串。"""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.6f}" if isinstance(value, float) else str(value)
    return str(value)


def render_metric_cards(metrics: Dict[str, Any]) -> str:
    """为 HTML 报告生成关键测试指标卡片。"""
    preferred_keys = ("accuracy", "loss", "mae", "qwk", "plus_minus_one_accuracy")
    cards = []
    for key in preferred_keys:
        if key in metrics:
            cards.append(
                '<div class="metric"><span>'
                + html.escape(key)
                + "</span><strong>"
                + html.escape(format_metric(metrics[key]))
                + "</strong></div>"
            )
    return "".join(cards) or '<p class="muted">没有可显示的测试指标。</p>'


def render_confusion_matrix(metrics: Dict[str, Any]) -> str:
    """把混淆矩阵转换成带类别名称的 HTML 表格。"""
    class_names = metrics.get("class_names") or []
    matrix = metrics.get("confusion_matrix") or []
    if not class_names or not matrix:
        return '<p class="muted">没有混淆矩阵。</p>'
    header = "".join(f"<th>预测 {html.escape(str(name))}</th>" for name in class_names)
    rows = []
    for class_name, row in zip(class_names, matrix):
        cells = "".join(f"<td>{int(value)}</td>" for value in row)
        rows.append(f"<tr><th>真实 {html.escape(str(class_name))}</th>{cells}</tr>")
    return f"<table><thead><tr><th></th>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_history_table(history: Dict[str, List[Any]]) -> str:
    """把逐 epoch 训练历史转换成 HTML 表格。"""
    epochs = max((len(values) for values in history.values()), default=0)
    if epochs == 0:
        return '<p class="muted">没有训练历史。</p>'
    keys = ["train_loss", "train_acc", "val_loss", "val_acc", "val_mae", "val_qwk"]
    header = "".join(f"<th>{html.escape(key)}</th>" for key in keys)
    rows = []
    for epoch_index in range(epochs):
        cells = []
        for key in keys:
            values = history.get(key, [])
            value = values[epoch_index] if epoch_index < len(values) else ""
            cells.append(f"<td>{html.escape(format_metric(value))}</td>")
        rows.append(f"<tr><td>{epoch_index + 1}</td>{''.join(cells)}</tr>")
    return f"<div class='scroll'><table><thead><tr><th>epoch</th>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def write_run_report(
    run_directory: Path,
    model_name: str,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    config: Optional[TrainingConfig] = None,
    metrics: Optional[Dict[str, Any]] = None,
    history: Optional[Dict[str, List[Any]]] = None,
    error_text: Optional[str] = None,
) -> Path:
    """生成不依赖第三方 HTML 库的单模型离线报告。"""
    metrics = metrics or {}
    history = history or {}
    duration_seconds = (finished_at - started_at).total_seconds()
    status_class = "ok" if status == "success" else "failed"
    config_json = "{}" if config is None else json.dumps(
        to_builtin(config.model_dump()), ensure_ascii=False, indent=2
    )
    per_class_rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{float(value):.4%}</td></tr>"
        for name, value in (metrics.get("per_class_accuracy") or {}).items()
    )
    error_section = ""
    if error_text:
        error_section = (
            "<section><h2>失败信息</h2><pre class='error'>"
            + html.escape(error_text)
            + "</pre></section>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(model_name)} 训练报告</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#65708a; --line:#dfe5ef; --accent:#315efb; }}
    body {{ margin:0; background:#f5f7fb; color:var(--ink); font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif; }}
    main {{ max-width:1100px; margin:32px auto; padding:0 20px 60px; }}
    header,section {{ background:white; border:1px solid var(--line); border-radius:14px; padding:22px; margin-bottom:18px; box-shadow:0 6px 20px #1d2a4410; }}
    h1,h2 {{ margin-top:0; }} .muted {{ color:var(--muted); }}
    .status {{ display:inline-block; padding:3px 10px; border-radius:999px; font-weight:700; }}
    .status.ok {{ color:#087443; background:#dff7eb; }} .status.failed {{ color:#a32727; background:#ffe5e5; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }}
    .metric {{ border:1px solid var(--line); border-radius:10px; padding:14px; }}
    .metric span {{ display:block; color:var(--muted); }} .metric strong {{ font-size:22px; }}
    table {{ border-collapse:collapse; width:100%; }} th,td {{ border:1px solid var(--line); padding:8px 10px; text-align:center; }} th {{ background:#f0f4fa; }}
    pre {{ overflow:auto; background:#101828; color:#e8eefc; padding:16px; border-radius:10px; }} pre.error {{ color:#ffd5d5; }}
    .scroll {{ overflow:auto; max-height:520px; }} a {{ color:var(--accent); }}
  </style>
</head>
<body><main>
  <header>
    <h1>{html.escape(model_name)} 训练报告</h1>
    <p><span class="status {status_class}">{html.escape(status)}</span></p>
    <p class="muted">开始：{started_at.strftime('%Y-%m-%d %H:%M:%S')}　结束：{finished_at.strftime('%Y-%m-%d %H:%M:%S')}　耗时：{duration_seconds:.1f} 秒</p>
    <p><a href="config.yaml">实际配置</a> · <a href="best_model.pth">最佳模型</a> · <a href="test_metrics.json">测试指标 JSON</a> · <a href="history.json">训练历史 JSON</a> · <a href="train.log">训练日志</a></p>
  </header>
  {error_section}
  <section><h2>测试集关键指标</h2><div class="metrics">{render_metric_cards(metrics)}</div></section>
  <section><h2>每类准确率</h2><table><thead><tr><th>类别</th><th>准确率</th></tr></thead><tbody>{per_class_rows}</tbody></table></section>
  <section><h2>测试集混淆矩阵</h2>{render_confusion_matrix(metrics)}</section>
  <section><h2>训练历史</h2>{render_history_table(history)}</section>
  <section><h2>完整运行配置</h2><pre>{html.escape(config_json)}</pre></section>
</main></body></html>"""
    report_path = run_directory / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def write_batch_summary(
    runs_root: Path,
    batch_timestamp: str,
    results: List[Dict[str, Any]],
) -> Path:
    """生成链接到每个模型报告的整批 HTML 总览。"""
    rows = []
    for result in results:
        report_path = Path(result["report_path"])
        relative_report = report_path.relative_to(runs_root).as_posix()
        accuracy = result.get("accuracy")
        accuracy_text = "-" if accuracy is None else f"{float(accuracy):.4%}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(result['model_name'])}</td>"
            f"<td>{html.escape(result['status'])}</td>"
            f"<td>{accuracy_text}</td>"
            f"<td><a href='{html.escape(relative_report)}'>打开报告</a></td>"
            "</tr>"
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>批量训练总览 {batch_timestamp}</title>
<style>body{{max-width:900px;margin:36px auto;padding:0 18px;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#172033}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #dfe5ef;padding:10px;text-align:center}}th{{background:#f0f4fa}}a{{color:#315efb}}</style>
</head><body><h1>批量训练总览</h1><p>批次时间：{html.escape(batch_timestamp)}</p>
<table><thead><tr><th>模型</th><th>状态</th><th>测试准确率</th><th>报告</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    summary_path = runs_root / f"batch_{batch_timestamp}_summary.html"
    summary_path.write_text(document, encoding="utf-8")
    return summary_path


def select_config_files(requested_names: Optional[List[str]]) -> List[Path]:
    '''从代码内 CONFIG_LIST 选择本次需要运行的相对配置路径。'''
    relative_paths = list(CONFIG_LIST)
    absolute_entries = [path for path in relative_paths if path.is_absolute()]
    if absolute_entries:
        raise ValueError(f'CONFIG_LIST 只允许项目相对路径：{absolute_entries}')
    names = [path.stem for path in relative_paths]
    if len(set(names)) != len(names):
        raise ValueError('CONFIG_LIST 中存在同名配置，无法作为唯一模型名称。')
    missing_paths = [path for path in relative_paths if not (PROJECT_ROOT / path).is_file()]
    if missing_paths:
        raise FileNotFoundError(f'CONFIG_LIST 中的配置不存在：{missing_paths}')
    if not requested_names:
        return relative_paths
    by_name = {path.stem: path for path in relative_paths}
    unknown_names = [name for name in requested_names if name not in by_name]
    if unknown_names:
        raise ValueError(f'--models 包含不在 CONFIG_LIST 中的名称：{unknown_names}')
    if len(set(requested_names)) != len(requested_names):
        raise ValueError('--models 中不能重复指定同一个配置。')
    return [by_name[name] for name in requested_names]


def build_training_config_from_file(
    relative_config_path: Path,
    dataset_root: Path,
    run_directory: Path,
    device: torch.device,
) -> TrainingConfig:
    '''读取原实验 YAML，并仅覆盖固定数据目录与本地运行相关设置。'''
    absolute_config_path = (PROJECT_ROOT / relative_config_path).resolve()
    with absolute_config_path.open('r', encoding='utf-8') as file:
        raw_config = yaml.safe_load(file)
    if not isinstance(raw_config, dict):
        raise ValueError(f'训练配置顶层必须是字典：{relative_config_path}')

    raw_config = copy.deepcopy(raw_config)
    raw_config['run_name'] = relative_config_path.stem
    raw_config['description'] = f'固定 patch 数据集运行；模板={relative_config_path.as_posix()}'
    raw_config['output_dir'] = str(run_directory)
    raw_config['use_wandb'] = False
    raw_config['enable_google_drive_upload'] = False
    raw_config.pop('logocv', None)

    raw_config['data'] = {
        'root': str(dataset_root),
        'type': 'image_folder',
        'class_to_idx': copy.deepcopy(FIXED_CLASS_TO_IDX),
        'train_transform': {'type': 'patch_train_224', 'image_size': 224},
        'eval_transform': {'type': 'patch_eval_224', 'image_size': 224},
        'test_transform': {'type': 'patch_eval_224', 'image_size': 224},
    }
    train_config = raw_config.setdefault('train', {})
    train_config['device'] = device.type
    train_config['enable_error_analysis'] = False
    train_config['tta_mode'] = 'mean'
    train_config['tta_topk'] = 0
    train_config['tta_compare'] = False
    train_config['tta_compare_modes'] = []
    return TrainingConfig(**raw_config)


def run_config_file(
    relative_config_path: Path,
    dataset_root: Path,
    runs_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    '''训练一个列表内 YAML、执行最终测试并保存完整归档。'''
    model_name = relative_config_path.stem
    run_directory = create_unique_run_directory(runs_root, model_name)
    config = build_training_config_from_file(
        relative_config_path, dataset_root, run_directory, device
    )
    started_at = datetime.now()
    trainer: Optional[Trainer] = None
    metrics: Dict[str, Any] = {}
    history: Dict[str, List[Any]] = {}
    status = 'success'
    error_text = None

    try:
        separator = '=' * 80
        print(f'\n{separator}\n开始训练：{model_name}\n配置模板：{relative_config_path}\n输出目录：{run_directory}\n{separator}')
        trainer = Trainer(config=config, device=device)
        trainer.train()
        history = to_builtin(trainer.history)
        metrics = evaluate_best_checkpoint(trainer)
        save_json(run_directory / 'history.json', history)
        save_json(run_directory / 'test_metrics.json', metrics)
    except Exception:
        status = 'failed'
        error_text = traceback.format_exc()
        failure_path = run_directory / 'failure.txt'
        failure_path.write_text(error_text, encoding='utf-8')
        if trainer is not None:
            history = to_builtin(getattr(trainer, 'history', {}))
            save_json(run_directory / 'history.json', history)
        print(f'配置 {model_name} 运行失败，详情见 {failure_path}')

    finished_at = datetime.now()
    report_path = write_run_report(
        run_directory=run_directory,
        model_name=model_name,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        config=config,
        metrics=metrics,
        history=history,
        error_text=error_text,
    )
    result = {
        'model_name': model_name,
        'config_path': relative_config_path.as_posix(),
        'status': status,
        'accuracy': metrics.get('accuracy'),
        'run_directory': str(run_directory),
        'report_path': str(report_path),
        'error': error_text,
    }
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_one_model(
    batch_config: Dict[str, Any],
    model_spec: Dict[str, Any],
    dataset_root: Path,
    runs_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """训练单个模型、执行最终测试并保存完整归档。"""
    model_name = str(model_spec["name"])
    run_directory = create_unique_run_directory(runs_root, model_name)
    config = build_training_config(batch_config, model_spec, dataset_root, run_directory, device)
    started_at = datetime.now()
    trainer: Optional[Trainer] = None
    metrics: Dict[str, Any] = {}
    history: Dict[str, List[Any]] = {}
    status = "success"
    error_text = None

    try:
        print(f"\n{'=' * 80}\n开始训练：{model_name}\n输出目录：{run_directory}\n{'=' * 80}")
        trainer = Trainer(config=config, device=device)
        trainer.train()
        history = to_builtin(trainer.history)
        metrics = evaluate_best_checkpoint(trainer)
        save_json(run_directory / "history.json", history)
        save_json(run_directory / "test_metrics.json", metrics)
    except Exception:
        status = "failed"
        error_text = traceback.format_exc()
        (run_directory / "failure.txt").write_text(error_text, encoding="utf-8")
        if trainer is not None:
            history = to_builtin(getattr(trainer, "history", {}))
            save_json(run_directory / "history.json", history)
        print(f"模型 {model_name} 运行失败，详情见 {run_directory / 'failure.txt'}")

    finished_at = datetime.now()
    report_path = write_run_report(
        run_directory=run_directory,
        model_name=model_name,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        config=config,
        metrics=metrics,
        history=history,
        error_text=error_text,
    )
    result = {
        "model_name": model_name,
        "status": status,
        "accuracy": metrics.get("accuracy"),
        "run_directory": str(run_directory),
        "report_path": str(report_path),
        "error": error_text,
    }
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def print_dataset_summary(summary: Dict[str, Any]) -> None:
    """以紧凑文本形式显示固定数据集检查结果。"""
    print("\n固定数据集检查通过：")
    for split_name in ("train", "val", "test"):
        item = summary[split_name]
        print(
            f"  {split_name}: total={item['total']}, classes={item['classes']}, "
            f"sample_shape={item['sample_shape']}"
        )


def parse_code_list_arguments() -> argparse.Namespace:
    '''解析代码内配置列表批量运行所需的命令行参数。'''
    parser = argparse.ArgumentParser(
        description='按照 train_batch.py 内 CONFIG_LIST 顺序运行固定 train/val/test 数据集。'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='只运行指定配置名；名称为 YAML 文件名去掉 .yaml。',
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cuda', 'cpu'),
        default='auto',
        help='训练设备（默认：auto，优先使用 CUDA）。',
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='显示代码内 CONFIG_LIST 的全部相对配置路径。',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='检查十个配置和固定数据集，不创建 runs 或训练模型。',
    )
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='任一配置失败后立即停止；默认记录失败并继续下一项。',
    )
    return parser.parse_args()


def parse_model_list_arguments() -> argparse.Namespace:
    '''解析当前任务模型列表的运行参数。'''
    parser = argparse.ArgumentParser(
        description='直接用固定 train/val/test patch 数据运行代码内 MODEL_LIST。'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='只运行指定模型，例如：--models resnet50 convnext_tiny。',
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cuda', 'cpu'),
        default='auto',
        help='训练设备（默认：auto，优先使用 CUDA）。',
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='显示代码内 MODEL_LIST 的全部模型。',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='检查模型配置和固定数据集，不创建 runs 或训练模型。',
    )
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='任一模型失败后立即停止；默认记录失败并继续下一模型。',
    )
    return parser.parse_args()


def main() -> None:
    '''用统一的当前任务设置顺序运行代码内七个唯一模型。'''
    args = parse_model_list_arguments()
    selected_models = select_models(MODEL_LIST, args.models)

    if args.list_models:
        print('代码内 MODEL_LIST：')
        for model_spec in selected_models:
            print('  - ' + str(model_spec.get('name')))
        return

    device = resolve_device(args.device)
    dataset_root = resolve_project_path(DATASET_ROOT)
    runs_root = resolve_project_path(RUNS_ROOT)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'固定数据集目录不存在：{dataset_root}')

    preview_directory = runs_root / '__dry_run__'
    preview_configs = [
        build_training_config(
            CURRENT_TASK_SETTINGS, model_spec, dataset_root, preview_directory, device
        )
        for model_spec in selected_models
    ]
    dataset_summary = validate_fixed_dataset(preview_configs[0], device)
    print_dataset_summary(dataset_summary)
    print('本次模型列表：' + ', '.join(str(spec.get('name')) for spec in selected_models))
    print(f'固定数据集相对路径：{DATASET_ROOT.as_posix()}')
    print('训练方式：统一交叉熵；不使用旧 YAML、SFI、soft-label、LOGOCV 或滑窗投票。')
    print(f'训练设备：{device}')

    if args.dry_run:
        print('\ndry-run 完成：模型和数据均已通过检查，未创建 runs 或训练模型。')
        return

    runs_root.mkdir(parents=True, exist_ok=True)
    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results: List[Dict[str, Any]] = []
    for model_spec in selected_models:
        result = run_one_model(
            CURRENT_TASK_SETTINGS, model_spec, dataset_root, runs_root, device
        )
        results.append(result)
        if result['status'] == 'failed' and args.fail_fast:
            break

    summary_path = write_batch_summary(runs_root, batch_timestamp, results)
    print(f'\n批量运行结束，总览报告：{summary_path}')
    failed_models = [result['model_name'] for result in results if result['status'] == 'failed']
    if failed_models:
        print(f'运行失败的模型：{failed_models}')
        raise SystemExit(1)


def _yaml_reference_main() -> None:
    '''使用代码内相对路径列表执行固定数据集批量训练。'''
    args = parse_code_list_arguments()
    selected_config_paths = select_config_files(args.models)

    if args.list_models:
        print('代码内 CONFIG_LIST：')
        for relative_path in selected_config_paths:
            print(f'  - {relative_path.stem}: {relative_path.as_posix()}')
        return

    device = resolve_device(args.device)
    dataset_root = resolve_project_path(DATASET_ROOT)
    runs_root = resolve_project_path(RUNS_ROOT)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'固定数据集目录不存在：{dataset_root}')

    preview_directory = runs_root / '__dry_run__'
    parsed_configs = [
        build_training_config_from_file(
            relative_path, dataset_root, preview_directory, device
        )
        for relative_path in selected_config_paths
    ]
    dataset_summary = validate_fixed_dataset(parsed_configs[0], device)
    print_dataset_summary(dataset_summary)
    print('本次相对配置列表：')
    for relative_path in selected_config_paths:
        print(f'  - {relative_path.as_posix()}')
    print(f'固定数据集相对路径：{DATASET_ROOT.as_posix()}')
    print(f'训练设备：{device}')

    auxiliary_configs = [
        path.stem for path, config in zip(selected_config_paths, parsed_configs)
        if 'aux_regression' in str(config.loss.get('type', ''))
    ]
    if auxiliary_configs:
        print(
            '提示：当前图片目录没有独立 SFI 连续值；以下配置仍可运行，'
            '其损失实现会在缺少 SFI 目标时只计算分类损失：'
            f'{auxiliary_configs}'
        )

    if args.dry_run:
        print('\ndry-run 完成：配置和数据均已通过检查，未创建 runs 或训练模型。')
        return

    runs_root.mkdir(parents=True, exist_ok=True)
    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results: List[Dict[str, Any]] = []
    for relative_path in selected_config_paths:
        result = run_config_file(relative_path, dataset_root, runs_root, device)
        results.append(result)
        if result['status'] == 'failed' and args.fail_fast:
            break

    summary_path = write_batch_summary(runs_root, batch_timestamp, results)
    print(f'\n批量运行结束，总览报告：{summary_path}')
    failed_models = [result['model_name'] for result in results if result['status'] == 'failed']
    if failed_models:
        print(f'运行失败的配置：{failed_models}')
        raise SystemExit(1)


def _legacy_batch_main() -> None:
    """执行批量训练入口。"""
    args = parse_arguments()
    batch_config = load_batch_config(args.batch_config)
    selected_models = select_models(batch_config["models"], args.models)

    if args.list_models:
        print("批量配置中的可运行模型：")
        for model_spec in selected_models:
            print(f"  - {model_spec['name']}")
        return

    device = resolve_device(args.device)
    dataset_root = resolve_project_path(batch_config.get("dataset_root", "datasets_split_patches"))
    runs_root = resolve_project_path(batch_config.get("runs_root", "runs"))
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"固定数据集目录不存在：{dataset_root}")

    preview_directory = runs_root / "__dry_run__"
    preview_config = build_training_config(
        batch_config,
        selected_models[0],
        dataset_root,
        preview_directory,
        device,
    )
    dataset_summary = validate_fixed_dataset(preview_config, device)
    print_dataset_summary(dataset_summary)
    print("本次模型列表：" + ", ".join(str(spec["name"]) for spec in selected_models))
    print(f"训练设备：{device}")

    if args.dry_run:
        print("\ndry-run 完成：未创建 runs 目录，也未训练模型。")
        return

    runs_root.mkdir(parents=True, exist_ok=True)
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: List[Dict[str, Any]] = []
    for model_spec in selected_models:
        result = run_one_model(batch_config, model_spec, dataset_root, runs_root, device)
        results.append(result)
        if result["status"] == "failed" and args.fail_fast:
            break

    summary_path = write_batch_summary(runs_root, batch_timestamp, results)
    print(f"\n批量运行结束，总览报告：{summary_path}")
    failed_models = [result["model_name"] for result in results if result["status"] == "failed"]
    if failed_models:
        print(f"运行失败的模型：{failed_models}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
