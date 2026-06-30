"""使用现成 train/val/test 目录顺序训练多个模型并生成本地 HTML 报告。"""

import argparse
import copy
import gc
import html
import json
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_CONFIG = Path('configs/fixed_split_patches_train.yaml')
CONFIG_LIST = (
    Path('configs/tryPractice/Attention/baseline.yaml'),
    Path('configs/tryPractice/Attention/eca_p1.yaml'),
    Path('configs/tryPractice/Attention/eca_p2.yaml'),
    Path('configs/tryPractice/Attention/eca_p3.yaml'),
    Path('configs/tryPractice/Attention/eca_p4.yaml'),
    Path('configs/tryPractice/Attention/eca_p5.yaml'),
    Path('configs/tryPractice/Attention/eca_p6.yaml'),
    Path('configs/tryPractice/Attention/eca_p46.yaml'),
    Path('configs/tryPractice/Attention/eca_p56.yaml'),
    Path('configs/tryPractice/Attention/eca_p456.yaml'),
    Path('configs/tryPractice/Attention/eca_pall.yaml'),
    Path('configs/tryPractice/Attention/cbam_p1.yaml'),
    Path('configs/tryPractice/Attention/cbam_p2.yaml'),
    Path('configs/tryPractice/Attention/cbam_p3.yaml'),
    Path('configs/tryPractice/Attention/cbam_p4.yaml'),
    Path('configs/tryPractice/Attention/cbam_p5.yaml'),
    Path('configs/tryPractice/Attention/cbam_p6.yaml'),
    Path('configs/tryPractice/Attention/cbam_p46.yaml'),
    Path('configs/tryPractice/Attention/cbam_p56.yaml'),
    Path('configs/tryPractice/Attention/cbam_p456.yaml'),
    Path('configs/tryPractice/Attention/cbam_pall.yaml'),
    Path('configs/tryPractice/Attention/ca_p1.yaml'),
    Path('configs/tryPractice/Attention/ca_p2.yaml'),
    Path('configs/tryPractice/Attention/ca_p3.yaml'),
    Path('configs/tryPractice/Attention/ca_p4.yaml'),
    Path('configs/tryPractice/Attention/ca_p5.yaml'),
    Path('configs/tryPractice/Attention/ca_p6.yaml'),
    Path('configs/tryPractice/Attention/ca_p46.yaml'),
    Path('configs/tryPractice/Attention/ca_p56.yaml'),
    Path('configs/tryPractice/Attention/ca_p456.yaml'),
    Path('configs/tryPractice/Attention/ca_pall.yaml'),
)

# 在 PyCharm 中右键运行前，只需要编辑上面的 YAML 路径列表。
# 模型结构和参数放在 YAML，具体实现通过 BACKBONES 注册表按 type 创建。

# PyCharm 右键运行时使用的设备；auto 表示有 CUDA 就用 CUDA，否则自动退回 CPU。
PYCHARM_DEVICE = 'auto'

# PyCharm 右键运行时是否只检查配置和数据，不真正训练；正式实验保持 False。
PYCHARM_DRY_RUN = False

# PyCharm 右键运行时是否遇到第一个失败模型就停止；False 会记录失败并继续后面的模型。
PYCHARM_FAIL_FAST = False

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models  # noqa: E402,F401 - 导入时完成模型、骨干网络和损失函数注册。
from src.engine import ComponentBuilder, Trainer  # noqa: E402
from src.schemas import TrainingConfig  # noqa: E402


def resolve_project_path(value: str | Path) -> Path:
    """把 YAML 中的相对路径统一解释为相对于项目根目录。"""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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


def compute_classification_details(
    confusion_matrix,
    class_names: List[str],
) -> Dict[str, Any]:
    '''由混淆矩阵统一计算 Macro-F1、逐类指标和行归一化矩阵。'''
    class_wise_metrics: Dict[str, Dict[str, Any]] = {}
    normalized_confusion_matrix: List[List[float]] = []
    f1_values: List[float] = []
    for class_index, class_name in enumerate(class_names):
        true_positive = int(confusion_matrix[class_index, class_index])
        actual_total = int(confusion_matrix[class_index, :].sum())
        predicted_total = int(confusion_matrix[:, class_index].sum())
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        class_wise_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': actual_total,
        }
        f1_values.append(f1)
        normalized_confusion_matrix.append(
            [
                float(value) / actual_total if actual_total else 0.0
                for value in confusion_matrix[class_index, :]
            ]
        )

    adjacent_confusions: Dict[str, Dict[str, Any]] = {}
    for left_index in range(len(class_names) - 1):
        right_index = left_index + 1
        left_name = class_names[left_index]
        right_name = class_names[right_index]
        left_total = int(confusion_matrix[left_index, :].sum())
        right_total = int(confusion_matrix[right_index, :].sum())
        left_to_right = int(confusion_matrix[left_index, right_index])
        right_to_left = int(confusion_matrix[right_index, left_index])
        adjacent_confusions[f'{left_name}<->{right_name}'] = {
            f'{left_name}_to_{right_name}_count': left_to_right,
            f'{left_name}_to_{right_name}_rate': (
                left_to_right / left_total if left_total else 0.0
            ),
            f'{right_name}_to_{left_name}_count': right_to_left,
            f'{right_name}_to_{left_name}_rate': (
                right_to_left / right_total if right_total else 0.0
            ),
        }
    return {
        'macro_f1': sum(f1_values) / len(f1_values) if f1_values else 0.0,
        'class_wise_metrics': class_wise_metrics,
        'normalized_confusion_matrix': normalized_confusion_matrix,
        'adjacent_confusions': adjacent_confusions,
    }


def profile_model_complexity(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 224,
) -> Dict[str, Any]:
    '''统计参数量，并用一次前向 hook 估算 Conv/Linear 的 FLOPs。'''
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    flops = 0
    hooks = []

    def convolution_hook(module, inputs, output):
        nonlocal flops
        kernel_height, kernel_width = module.kernel_size
        operations_per_output = (
            kernel_height * kernel_width * module.in_channels // module.groups
        )
        # 一个乘法和一个加法按 2 FLOPs 计算。
        flops += int(output.numel() * operations_per_output * 2)

    def convolution_1d_hook(module, inputs, output):
        nonlocal flops
        kernel_size = module.kernel_size[0]
        operations_per_output = kernel_size * module.in_channels // module.groups
        flops += int(output.numel() * operations_per_output * 2)

    def linear_hook(module, inputs, output):
        nonlocal flops
        flops += int(output.numel() * module.in_features * 2)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            hooks.append(module.register_forward_hook(convolution_1d_hook))
        elif isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    try:
        model.eval()
        dummy_input = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            model(dummy_input)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    return {
        'parameters_total': int(total_parameters),
        'parameters_trainable': int(trainable_parameters),
        'flops': int(flops),
        'flops_g': float(flops / 1_000_000_000),
        'flops_method': 'single 224x224 forward; Conv1d/Conv2d/Linear; multiply-add=2 FLOPs',
    }


def measure_inference_time(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    '''逐 batch 同步计时模型前向，不把数据读取和主机到设备传输计入。'''
    was_training = model.training
    model.eval()
    elapsed_seconds = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started_at = time.perf_counter()
            model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed_seconds += time.perf_counter() - started_at
            sample_count += int(images.shape[0])
    model.train(was_training)
    return {
        'inference_time_seconds': elapsed_seconds,
        'inference_ms_per_sample': (
            elapsed_seconds * 1000.0 / sample_count if sample_count else 0.0
        ),
        'inference_samples_per_second': (
            sample_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
    }


def evaluate_best_checkpoint(
    trainer: Trainer,
    training_time_seconds: float,
) -> Dict[str, Any]:
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
    result['training_time_seconds'] = training_time_seconds
    result.update(compute_classification_details(confusion_matrix, class_names))
    image_size = int(getattr(trainer.config.data, 'image_size', 224) or 224)
    result.update(
        profile_model_complexity(trainer.model, trainer.device, image_size=image_size)
    )
    result.update(
        measure_inference_time(trainer.model, trainer.test_loader, trainer.device)
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
    preferred_keys = preferred_keys + (
        'macro_f1',
        'parameters_total',
        'flops_g',
        'training_time_seconds',
        'inference_ms_per_sample',
    )
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


def render_normalized_confusion_matrix(metrics: Dict[str, Any]) -> str:
    '''把按真实类别行归一化的混淆矩阵渲染成百分比表格。'''
    class_names = metrics.get('class_names') or []
    matrix = metrics.get('normalized_confusion_matrix') or []
    if not class_names or not matrix:
        return '<p class="muted">没有归一化混淆矩阵。</p>'
    header = ''.join(f'<th>预测 {html.escape(str(name))}</th>' for name in class_names)
    rows = []
    for class_name, row in zip(class_names, matrix):
        cells = ''.join(f'<td>{float(value):.2%}</td>' for value in row)
        rows.append(f'<tr><th>真实 {html.escape(str(class_name))}</th>{cells}</tr>')
    return (
        '<table><thead><tr><th></th>'
        + header
        + '</tr></thead><tbody>'
        + ''.join(rows)
        + '</tbody></table>'
    )


def render_class_wise_metrics(metrics: Dict[str, Any]) -> str:
    '''渲染每个类别的 Precision、Recall、F1 和样本数。'''
    rows = []
    for class_name, values in (metrics.get('class_wise_metrics') or {}).items():
        precision = float(values['precision'])
        recall = float(values['recall'])
        f1 = float(values['f1'])
        support = int(values['support'])
        rows.append(
            '<tr>'
            f'<td>{html.escape(str(class_name))}</td>'
            f'<td>{precision:.4f}</td>'
            f'<td>{recall:.4f}</td>'
            f'<td>{f1:.4f}</td>'
            f'<td>{support}</td>'
            '</tr>'
        )
    if not rows:
        return '<p class="muted">没有逐类指标。</p>'
    return (
        '<table><thead><tr><th>类别</th><th>Precision</th><th>Recall</th>'
        '<th>F1</th><th>Support</th></tr></thead><tbody>'
        + ''.join(rows)
        + '</tbody></table>'
    )


def render_adjacent_confusions(metrics: Dict[str, Any]) -> str:
    '''渲染三个相邻严重程度类别之间的双向误分类。'''
    rows = []
    for pair_name, values in (metrics.get('adjacent_confusions') or {}).items():
        items = list(values.items())
        if len(items) != 4:
            continue
        rows.append(
            '<tr>'
            f'<td>{html.escape(str(pair_name))}</td>'
            f'<td>{html.escape(items[0][0])}</td>'
            f'<td>{int(items[0][1])}</td>'
            f'<td>{float(items[1][1]):.2%}</td>'
            f'<td>{html.escape(items[2][0])}</td>'
            f'<td>{int(items[2][1])}</td>'
            f'<td>{float(items[3][1]):.2%}</td>'
            '</tr>'
        )
    if not rows:
        return '<p class="muted">没有相邻类别混淆统计。</p>'
    return (
        '<table><thead><tr><th>类别对</th><th>方向一</th><th>数量</th><th>比例</th>'
        '<th>方向二</th><th>数量</th><th>比例</th></tr></thead><tbody>'
        + ''.join(rows)
        + '</tbody></table>'
    )


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
  <section><h2>逐类 Precision / Recall / F1</h2>{render_class_wise_metrics(metrics)}</section>
  <section><h2>测试集混淆矩阵</h2>{render_confusion_matrix(metrics)}</section>
  <section><h2>测试集归一化混淆矩阵</h2>{render_normalized_confusion_matrix(metrics)}</section>
  <section><h2>相邻严重程度混淆</h2>{render_adjacent_confusions(metrics)}</section>
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
        macro_f1 = result.get("macro_f1")
        macro_f1_text = "-" if macro_f1 is None else f"{float(macro_f1):.4f}"
        mae = result.get("mae")
        mae_text = "-" if mae is None else f"{float(mae):.4f}"
        qwk = result.get("qwk")
        qwk_text = "-" if qwk is None else f"{float(qwk):.4f}"
        params = result.get("parameters_total")
        params_text = "-" if params is None else f"{int(params):,}"
        flops_g = result.get("flops_g")
        flops_text = "-" if flops_g is None else f"{float(flops_g):.3f} G"
        rows.append(
            "<tr>"
            f"<td>{html.escape(result['model_name'])}</td>"
            f"<td>{html.escape(result['status'])}</td>"
            f"<td>{accuracy_text}</td>"
            f"<td>{macro_f1_text}</td>"
            f"<td>{mae_text}</td>"
            f"<td>{qwk_text}</td>"
            f"<td>{params_text}</td>"
            f"<td>{flops_text}</td>"
            f"<td><a href='{html.escape(relative_report)}'>打开报告</a></td>"
            "</tr>"
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>批量训练总览 {batch_timestamp}</title>
<style>body{{max-width:900px;margin:36px auto;padding:0 18px;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#172033}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #dfe5ef;padding:10px;text-align:center}}th{{background:#f0f4fa}}a{{color:#315efb}}</style>
</head><body><h1>批量训练总览</h1><p>批次时间：{html.escape(batch_timestamp)}</p>
<table><thead><tr><th>模型</th><th>状态</th><th>Accuracy</th><th>Macro-F1</th><th>MAE</th><th>QWK</th><th>Params</th><th>FLOPs</th><th>报告</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
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


def load_yaml_mapping(relative_path: Path, config_kind: str) -> Dict[str, Any]:
    '''读取项目内 YAML，并检查其顶层是否为字典。'''
    absolute_path = resolve_project_path(relative_path)
    if not absolute_path.is_file():
        raise FileNotFoundError(f'{config_kind}不存在：{absolute_path}')
    with absolute_path.open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f'{config_kind}顶层必须是 YAML 字典：{relative_path}')
    return config


def load_common_config() -> Dict[str, Any]:
    '''读取不包含模型结构的公共训练配置。'''
    config = load_yaml_mapping(COMMON_CONFIG, '公共训练配置')
    required_sections = ('data', 'train', 'optimizer', 'loss')
    missing_sections = [
        section for section in required_sections
        if not isinstance(config.get(section), dict)
    ]
    if missing_sections:
        raise ValueError(f'公共训练配置缺少字典节点：{missing_sections}')
    if 'model' in config or 'models' in config:
        raise ValueError(
            '公共训练配置不能包含 model/models；模型结构请放到独立模型 YAML。'
        )
    return config


def load_model_config(relative_path: Path) -> Dict[str, Any]:
    '''读取单个模型 YAML，并检查名称和模型结构。'''
    config = load_yaml_mapping(relative_path, '模型配置')
    model_name = config.get('name')
    model_config = config.get('model')
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(f'模型配置必须包含非空 name：{relative_path}')
    if model_name != relative_path.stem:
        raise ValueError(
            f'模型 name 必须与 YAML 文件名一致：name={model_name!r}, '
            f'文件名={relative_path.stem!r}'
        )
    if not isinstance(model_config, dict):
        raise ValueError(f'模型配置缺少 model 字典：{relative_path}')
    if not isinstance(model_config.get('backbone'), dict):
        raise ValueError(f'模型配置缺少 model.backbone 字典：{relative_path}')
    if not isinstance(model_config.get('head'), dict):
        raise ValueError(f'模型配置缺少 model.head 字典：{relative_path}')
    return config


def build_training_config_from_file(
    common_config: Dict[str, Any],
    model_config: Dict[str, Any],
    relative_config_path: Path,
    dataset_root: Path,
    run_directory: Path,
    device: torch.device,
) -> TrainingConfig:
    '''合并公共训练 YAML 与单个模型 YAML，生成 Trainer 的最终配置。'''
    runtime_config = copy.deepcopy(common_config)
    model_name = str(model_config['name'])
    runtime_config['run_name'] = model_name
    runtime_config['description'] = (
        f'固定 patch 数据集训练；模型配置={relative_config_path.as_posix()}'
    )
    runtime_config['output_dir'] = str(run_directory)
    runtime_config['use_wandb'] = False
    runtime_config['enable_google_drive_upload'] = False
    runtime_config['model'] = copy.deepcopy(model_config['model'])

    data_config = runtime_config['data']
    data_config['root'] = str(dataset_root)
    data_config['class_to_idx'] = copy.deepcopy(
        runtime_config.pop(
            'class_to_idx',
            {'pre': 0, 'slight': 1, 'moderate': 2, 'over': 3},
        )
    )

    train_config = runtime_config['train']
    train_config['device'] = device.type

    # 这些字段只负责批量脚本寻址，不属于 Trainer 的运行语义。
    runtime_config.pop('experiment_name', None)
    runtime_config.pop('dataset_root', None)
    runtime_config.pop('runs_root', None)
    return TrainingConfig(**runtime_config)


def run_config_file(
    common_config: Dict[str, Any],
    relative_config_path: Path,
    model_config: Dict[str, Any],
    dataset_root: Path,
    runs_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    '''训练一个模型 YAML，执行最终测试并保存完整归档。'''
    model_name = str(model_config['name'])
    run_directory = create_unique_run_directory(runs_root, model_name)
    config = build_training_config_from_file(
        common_config,
        model_config,
        relative_config_path,
        dataset_root,
        run_directory,
        device,
    )
    started_at = datetime.now()
    trainer: Optional[Trainer] = None
    metrics: Dict[str, Any] = {}
    history: Dict[str, List[Any]] = {}
    status = 'success'
    error_text = None

    try:
        separator = '=' * 80
        print(
            f'\n{separator}\n开始训练：{model_name}'
            f'\n模型配置：{relative_config_path}'
            f'\n输出目录：{run_directory}\n{separator}'
        )
        trainer = Trainer(config=config, device=device)
        training_started_at = time.perf_counter()
        trainer.train()
        training_time_seconds = time.perf_counter() - training_started_at
        history = to_builtin(trainer.history)
        metrics = evaluate_best_checkpoint(trainer, training_time_seconds)
        save_json(run_directory / 'history.json', history)
        save_json(run_directory / 'test_metrics.json', metrics)
    except Exception:
        status = 'failed'
        error_text = traceback.format_exc()
        print(error_text)
        failure_path = run_directory / 'failure.txt'
        failure_path.write_text(error_text, encoding='utf-8')
        if trainer is not None:
            history = to_builtin(getattr(trainer, 'history', {}))
            save_json(run_directory / 'history.json', history)
        print(f'模型 {model_name} 运行失败，详情见 {failure_path}')

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
        'macro_f1': metrics.get('macro_f1'),
        'mae': metrics.get('mae'),
        'qwk': metrics.get('qwk'),
        'parameters_total': metrics.get('parameters_total'),
        'flops_g': metrics.get('flops_g'),
        'run_directory': str(run_directory),
        'report_path': str(report_path),
        'error': error_text,
    }
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def print_dataset_summary(summary: Dict[str, Any]) -> None:
    '''以紧凑文本显示固定数据集检查结果。'''
    print('\n固定数据集检查通过：')
    for split_name in ('train', 'val', 'test'):
        item = summary[split_name]
        print(
            f'  {split_name}: total={item["total"]}, classes={item["classes"]}, '
            f'sample_shape={item["sample_shape"]}'
        )


def parse_arguments() -> argparse.Namespace:
    '''解析 YAML 列表批量训练参数。'''
    parser = argparse.ArgumentParser(
        description='按照 CONFIG_LIST 顺序运行固定 train/val/test patch 数据集。'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='只运行指定配置名；名称为 YAML 文件名去掉 .yaml。',
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cuda', 'cpu'),
        default=PYCHARM_DEVICE,
        help=f'训练设备（默认：{PYCHARM_DEVICE}）。',
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='显示 CONFIG_LIST 中本次会运行的模型 YAML。',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=PYCHARM_DRY_RUN,
        help='只检查 YAML 和固定数据集，不创建 runs 或训练模型。',
    )
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        default=PYCHARM_FAIL_FAST,
        help='任一模型失败后立即停止；默认记录失败并继续下一项。',
    )
    return parser.parse_args()


def main() -> None:
    '''加载公共训练 YAML，并依次运行 CONFIG_LIST 中的模型 YAML。'''
    args = parse_arguments()
    selected_paths = select_config_files(args.models)

    if args.list_models:
        print('CONFIG_LIST 中的模型配置：')
        for relative_path in selected_paths:
            print(f'  - {relative_path.stem}: {relative_path.as_posix()}')
        return

    common_config = load_common_config()
    model_entries = [
        (relative_path, load_model_config(relative_path))
        for relative_path in selected_paths
    ]

    device = resolve_device(args.device)
    dataset_root = resolve_project_path(
        common_config.get('dataset_root', 'datasets_split_patches')
    )
    runs_root = resolve_project_path(common_config.get('runs_root', 'runs'))
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'固定数据集目录不存在：{dataset_root}')

    preview_directory = runs_root / '__dry_run__'
    preview_configs = [
        build_training_config_from_file(
            common_config,
            model_config,
            relative_path,
            dataset_root,
            preview_directory,
            device,
        )
        for relative_path, model_config in model_entries
    ]
    dataset_summary = validate_fixed_dataset(preview_configs[0], device)
    print_dataset_summary(dataset_summary)
    print(f'公共训练配置：{COMMON_CONFIG.as_posix()}')
    print('本次模型配置：')
    for relative_path, _ in model_entries:
        print(f'  - {relative_path.as_posix()}')
    print(f'固定数据集：{dataset_root}')
    print(f'随机种子：{common_config.get("random_seed", 42)}')
    print(f'训练轮数：{common_config["train"]["epochs"]}')
    print(f'训练设备：{device}')

    if args.dry_run:
        print('\ndry-run 完成：全部 YAML 和数据均已通过检查，未创建 runs 或训练模型。')
        return

    runs_root.mkdir(parents=True, exist_ok=True)
    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results: List[Dict[str, Any]] = []
    for relative_path, model_config in model_entries:
        result = run_config_file(
            common_config,
            relative_path,
            model_config,
            dataset_root,
            runs_root,
            device,
        )
        results.append(result)
        if result['status'] == 'failed' and args.fail_fast:
            break

    summary_path = write_batch_summary(runs_root, batch_timestamp, results)
    print(f'\n批量运行结束，总览报告：{summary_path}')
    failed_models = [
        result['model_name']
        for result in results
        if result['status'] == 'failed'
    ]
    if failed_models:
        print(f'运行失败的模型：{failed_models}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
