"""使用现成 train/val/test 目录顺序训练多个模型并生成本地 HTML 报告。"""

import argparse
import copy
import csv
import gc
import html
import json
import re
import statistics
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

    # 可直接通过 timm 获取的 backbone：统一 224 输入、ImageNet-1K 预训练、hard-label CE。
    Path('configs/tryPractice/TimmBackbone224/convnextv2_tiny_ce.yaml'),
    Path('configs/tryPractice/TimmBackbone224/fasternet_t2_ce.yaml'),
    Path('configs/tryPractice/TimmBackbone224/inceptionnext_tiny_ce.yaml'),
    Path('configs/tryPractice/TimmBackbone224/repvit_m2_3_ce.yaml'),
    Path('configs/tryPractice/TimmBackbone224/mambaout_tiny_timm_ce.yaml'),
    # Transformer / hybrid backbone：同样只保留可直接通过 timm 获取的模型。
    Path('configs/tryPractice/TransformerBackbone224/fastvit_sa24_256pre_224ft_ce.yaml'),
    Path('configs/tryPractice/TransformerBackbone224/efficientformerv2_s2_ce.yaml'),
    Path('configs/tryPractice/TransformerBackbone224/shvit_s3_ce.yaml'),
)

# 在 PyCharm 中右键运行前，只需要编辑上面的 YAML 路径列表。
# 模型结构和参数放在 YAML，具体实现通过 BACKBONES 注册表按 type 创建。

# PyCharm 右键运行时使用的设备；auto 表示有 CUDA 就用 CUDA，否则自动退回 CPU。
PYCHARM_DEVICE = 'auto'

# PyCharm 右键运行时是否只检查配置和数据，不真正训练；正式实验保持 False。
PYCHARM_DRY_RUN = False

# PyCharm 右键运行时是否遇到第一个失败模型就停止；False 会记录失败并继续后面的模型。
PYCHARM_FAIL_FAST = False

# PyCharm 右键运行时是否保留 .pth：
# False = 完成最佳权重测试和报告后删除 .pth，适合大批量实验节省空间。
# True  = 保留 best_model.pth，适合需要后续加载权重的正式模型。
PYCHARM_KEEP_PTH_FILES = False

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


def identify_hard_adjacent_samples(
    trainer: Trainer,
    probability_margin: float,
    hard_weight: float,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """用当前 Stage5 checkpoint 在确定性训练集上固定识别困难相邻样本。"""
    from torch.utils.data import DataLoader

    from src.data.loader import ImageFolderWithPaths

    if hard_weight <= 0:
        raise ValueError('hard_weight 必须大于 0。')
    train_dataset = trainer.train_loader.dataset
    if not isinstance(train_dataset, ImageFolderWithPaths):
        raise TypeError('hard-adjacent 实验目前要求 image_folder 固定划分数据集。')

    # 困难样本检测不使用随机翻转，避免同一 checkpoint 因增强随机性得到不同清单。
    eval_transform = trainer.val_loader.dataset.transform
    detection_dataset = ImageFolderWithPaths(
        root=train_dataset.root,
        transform=eval_transform,
        class_to_idx=train_dataset.class_to_idx,
    )
    local_generator = torch.Generator()
    local_generator.manual_seed(0)
    detection_loader = DataLoader(
        detection_dataset,
        batch_size=int(getattr(trainer.config.train, 'val_batch_size', 64) or 64),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=local_generator,
    )

    model = trainer.model
    was_training = model.training
    model.eval()
    hard_sample_weights: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in detection_loader:
            images, labels, paths = batch
            images = images.to(trainer.device)
            labels_device = labels.to(trainer.device)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            if logits.ndim != 2 or logits.shape[1] != 4:
                raise ValueError(
                    'hard-adjacent 检测要求原始四分类 logits，'
                    f'实际形状为 {tuple(logits.shape)}。'
                )
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            for index, path in enumerate(paths):
                true_label = int(labels_device[index].item())
                pred_label = int(predictions[index].item())
                adjacent_indices = [
                    candidate
                    for candidate in (true_label - 1, true_label + 1)
                    if 0 <= candidate < 4
                ]
                true_probability = float(probabilities[index, true_label].item())
                max_adjacent_probability = float(
                    probabilities[index, adjacent_indices].max().item()
                )
                margin = true_probability - max_adjacent_probability
                adjacent_misclassification = (
                    pred_label != true_label
                    and abs(pred_label - true_label) == 1
                )
                low_adjacent_margin = margin < probability_margin
                if not (adjacent_misclassification or low_adjacent_margin):
                    continue
                path_text = str(path)
                hard_sample_weights[path_text] = float(hard_weight)
                reasons = []
                if adjacent_misclassification:
                    reasons.append('adjacent_misclassification')
                if low_adjacent_margin:
                    reasons.append('true_vs_adjacent_margin')
                rows.append(
                    {
                        'image_path': path_text,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'true_probability': true_probability,
                        'max_adjacent_probability': max_adjacent_probability,
                        'probability_margin': margin,
                        'adjacent_misclassification': int(adjacent_misclassification),
                        'low_adjacent_margin': int(low_adjacent_margin),
                        'reason': '+'.join(reasons),
                        'sample_weight': float(hard_weight),
                    }
                )
    model.train(was_training)

    csv_path = trainer.output_dir / 'hard_adjacent_samples.csv'
    fieldnames = [
        'image_path',
        'true_label',
        'pred_label',
        'true_probability',
        'max_adjacent_probability',
        'probability_margin',
        'adjacent_misclassification',
        'low_adjacent_margin',
        'reason',
        'sample_weight',
    ]
    with csv_path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return hard_sample_weights, {
        'hard_adjacent_samples_csv': str(csv_path),
        'hard_adjacent_sample_count': len(rows),
        'hard_adjacent_sample_rate': (
            len(rows) / len(detection_dataset) if len(detection_dataset) else 0.0
        ),
        'hard_adjacent_probability_margin': float(probability_margin),
        'hard_adjacent_sample_weight': float(hard_weight),
    }


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
    boundary_accuracies: Dict[str, Dict[str, Any]] = {}
    adjacent_error_counts: Dict[str, int] = {}
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
        # 只按真实标签筛选相邻两类，但预测仍使用原始 K 类结果。
        # 因此预测成这两个类别之外的类别同样会计为错误，绝不重新二选一。
        boundary_total = left_total + right_total
        boundary_correct = (
            int(confusion_matrix[left_index, left_index])
            + int(confusion_matrix[right_index, right_index])
        )
        boundary_key = f'acc_{left_index}_{right_index}'
        boundary_accuracies[boundary_key] = {
            'accuracy': (
                boundary_correct / boundary_total if boundary_total else 0.0
            ),
            'correct': boundary_correct,
            'total': boundary_total,
            'true_classes': [left_index, right_index],
        }
        adjacent_error_counts[f'error_{left_index}_to_{right_index}'] = left_to_right
        adjacent_error_counts[f'error_{right_index}_to_{left_index}'] = right_to_left

    adjacent_error_count = 0
    distant_error_count = 0
    for true_index in range(len(class_names)):
        for predicted_index in range(len(class_names)):
            if true_index == predicted_index:
                continue
            error_count = int(confusion_matrix[true_index, predicted_index])
            if abs(true_index - predicted_index) == 1:
                adjacent_error_count += error_count
            else:
                distant_error_count += error_count
    total_error_count = adjacent_error_count + distant_error_count
    details = {
        'macro_f1': sum(f1_values) / len(f1_values) if f1_values else 0.0,
        'class_wise_metrics': class_wise_metrics,
        'normalized_confusion_matrix': normalized_confusion_matrix,
        'adjacent_confusions': adjacent_confusions,
        'boundary_accuracies': boundary_accuracies,
        'adjacent_error_counts': adjacent_error_counts,
        'total_error_count': total_error_count,
        'adjacent_error_count': adjacent_error_count,
        'distant_error_count': distant_error_count,
        'far_error_count': distant_error_count,
        'adjacent_error_rate': (
            adjacent_error_count / total_error_count if total_error_count else 0.0
        ),
        'distant_error_rate': (
            distant_error_count / total_error_count if total_error_count else 0.0
        ),
    }
    for boundary_key, values in boundary_accuracies.items():
        details[boundary_key] = values['accuracy']
    details.update(adjacent_error_counts)
    return details


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


ORDINAL_REPRESENTATION_RUNS = {
    'baseline_ce',
    'ce_proto',
    'ce_opcl',
    'ce_rank_opcl',
    'softlabel_rank_opcl',
}


def save_ordinal_representation_artifacts(trainer: Trainer) -> Dict[str, Any]:
    '''为指定 OPCL 版本保存表征、t-SNE、rank 预测和 prototype 距离。'''
    run_name = str(getattr(trainer.config, 'run_name', '') or '')
    backbone = getattr(trainer.model, 'backbone', None)
    if (
        run_name not in ORDINAL_REPRESENTATION_RUNS
        or not bool(getattr(backbone, 'is_ordinal_opcl', False))
    ):
        return {}

    model = trainer.model
    was_training = model.training
    model.eval()
    representations = []
    labels_all = []
    predictions_all = []
    paths_all: List[str] = []
    rank_predictions = []
    normalized_prototypes = None

    with torch.no_grad():
        for batch in trainer.test_loader:
            images = batch[0].to(trainer.device)
            labels = batch[1]
            paths = batch[2] if len(batch) >= 3 else [''] * int(labels.shape[0])
            outputs = model(images)
            if not isinstance(outputs, tuple) or len(outputs) < 2:
                raise RuntimeError('Ordinal OPCL 模型必须返回 (logits, auxiliary)。')
            logits, auxiliary = outputs[0], outputs[1]
            representation = auxiliary.get('embedding')
            if representation is None:
                representation = auxiliary.get('feature')
            if representation is None:
                raise RuntimeError('模型没有返回 embedding 或 feature。')
            representations.append(representation.detach().cpu())
            labels_all.append(labels.detach().cpu())
            predictions_all.append(logits.argmax(dim=1).detach().cpu())
            paths_all.extend(str(path) for path in paths)
            if auxiliary.get('rank_pred') is not None:
                rank_predictions.append(auxiliary['rank_pred'].detach().cpu().view(-1))
            if auxiliary.get('prototypes') is not None:
                normalized_prototypes = auxiliary['prototypes'].detach().cpu()
    model.train(was_training)

    representation_tensor = torch.cat(representations, dim=0)
    label_tensor = torch.cat(labels_all, dim=0)
    prediction_tensor = torch.cat(predictions_all, dim=0)
    artifact_path = trainer.output_dir / 'representation_features.pt'
    torch.save(
        {
            'run_name': run_name,
            'representations': representation_tensor,
            'labels': label_tensor,
            'predictions': prediction_tensor,
            'paths': paths_all,
            'representation_type': (
                'embedding'
                if representation_tensor.shape[1] != 1280
                else 'backbone_feature'
            ),
        },
        artifact_path,
    )

    if rank_predictions:
        rank_tensor = torch.cat(rank_predictions, dim=0)
        with (trainer.output_dir / 'ordinal_rank_predictions.csv').open(
            'w',
            encoding='utf-8-sig',
            newline='',
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                ['path', 'true_class', 'rank_target', 'rank_prediction']
            )
            for path, label, rank_prediction in zip(
                paths_all,
                label_tensor.tolist(),
                rank_tensor.tolist(),
            ):
                writer.writerow(
                    [path, label, float(label) / 3.0, float(rank_prediction)]
                )

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        sample_count = int(representation_tensor.shape[0])
        perplexity = min(30.0, max(2.0, float(sample_count - 1) / 3.0))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init='pca',
            learning_rate='auto',
            random_state=2026,
        )
        coordinates = tsne.fit_transform(representation_tensor.numpy())
        class_names = list(trainer.test_loader.dataset.classes)
        with (trainer.output_dir / 'representation_tsne.csv').open(
            'w',
            encoding='utf-8-sig',
            newline='',
        ) as file:
            writer = csv.writer(file)
            writer.writerow(['path', 'true_class', 'predicted_class', 'tsne_x', 'tsne_y'])
            for path, label, prediction, coordinate in zip(
                paths_all,
                label_tensor.tolist(),
                prediction_tensor.tolist(),
                coordinates,
            ):
                writer.writerow(
                    [path, label, prediction, float(coordinate[0]), float(coordinate[1])]
                )
        figure, axis = plt.subplots(figsize=(8, 7))
        for class_index, class_name in enumerate(class_names):
            mask = label_tensor.numpy() == class_index
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=12,
                alpha=0.65,
                label=class_name,
            )
        axis.set_title(f'{run_name} representation t-SNE')
        axis.legend()
        figure.tight_layout()
        figure.savefig(trainer.output_dir / 'representation_tsne.png', dpi=180)
        plt.close(figure)

        if normalized_prototypes is not None:
            prototype_distances = (
                1.0 - normalized_prototypes @ normalized_prototypes.transpose(0, 1)
            )
            with (trainer.output_dir / 'prototype_distance_matrix.csv').open(
                'w',
                encoding='utf-8-sig',
                newline='',
            ) as file:
                writer = csv.writer(file)
                writer.writerow(['prototype', *class_names])
                for class_name, row in zip(class_names, prototype_distances.tolist()):
                    writer.writerow([class_name, *row])
            figure, axis = plt.subplots(figsize=(6, 5))
            image = axis.imshow(
                prototype_distances.numpy(),
                cmap='viridis',
                vmin=0.0,
            )
            axis.set_xticks(range(len(class_names)), class_names, rotation=30)
            axis.set_yticks(range(len(class_names)), class_names)
            axis.set_title(f'{run_name} prototype cosine distance')
            for row_index in range(len(class_names)):
                for column_index in range(len(class_names)):
                    axis.text(
                        column_index,
                        row_index,
                        f'{prototype_distances[row_index, column_index]:.3f}',
                        ha='center',
                        va='center',
                        color='white',
                    )
            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            figure.savefig(
                trainer.output_dir / 'prototype_distance_matrix.png',
                dpi=180,
            )
            plt.close(figure)
    except Exception:
        (trainer.output_dir / 'representation_analysis_failure.txt').write_text(
            traceback.format_exc(),
            encoding='utf-8',
        )

    return {
        'representation_artifact': str(artifact_path),
        'representation_samples': int(representation_tensor.shape[0]),
        'representation_dimensions': int(representation_tensor.shape[1]),
    }


def save_probabilistic_prediction_csv(trainer: Trainer) -> Dict[str, Any]:
    '''保存概率有序头的逐样本分布参数、区间概率与不确定性。'''
    backbone = getattr(trainer.model, 'backbone', None)
    if not bool(getattr(backbone, 'is_probabilistic_ordinal', False)):
        return {}

    model = trainer.model
    was_training = model.training
    model.eval()
    rows = []
    variance_correct = []
    variance_wrong = []
    max_probability_correct = []
    max_probability_wrong = []
    probabilistic_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in trainer.test_loader:
            images = batch[0].to(trainer.device)
            labels = batch[1].detach().cpu()
            paths = batch[2] if len(batch) >= 3 else [''] * int(labels.shape[0])
            outputs = model(images)
            primary_logits, auxiliary = outputs[0], outputs[1]
            predictions = primary_logits.argmax(dim=1).detach().cpu()
            stage_probs = auxiliary['stage_probs'].detach().cpu()
            probabilistic_predictions = stage_probs.argmax(dim=1)
            probabilistic_correct += int(
                probabilistic_predictions.eq(labels).sum().item()
            )
            total_samples += int(labels.shape[0])
            distribution = str(auxiliary.get('distribution', ''))

            tensor_values = {}
            for key in (
                'alpha',
                'beta',
                'mean',
                'var',
                'a',
                'b',
                'mu',
                'sigma',
                'mean_proxy',
            ):
                value = auxiliary.get(key)
                if torch.is_tensor(value):
                    tensor_values[key] = value.detach().cpu().view(-1)

            for index, (path, true_label, predicted_label) in enumerate(
                zip(paths, labels.tolist(), predictions.tolist())
            ):
                probabilities = stage_probs[index]
                correct = int(true_label == predicted_label)
                max_probability = float(probabilities.max().item())
                variance = (
                    float(tensor_values['var'][index].item())
                    if 'var' in tensor_values
                    else None
                )
                if correct:
                    max_probability_correct.append(max_probability)
                    if variance is not None:
                        variance_correct.append(variance)
                else:
                    max_probability_wrong.append(max_probability)
                    if variance is not None:
                        variance_wrong.append(variance)
                row = {
                    'image_path': str(path),
                    'true_label': int(true_label),
                    'pred_label': int(predicted_label),
                    'correct': correct,
                    'abs_error': abs(int(true_label) - int(predicted_label)),
                    'distribution': distribution,
                    'P_Pre': float(probabilities[0].item()),
                    'P_Slight': float(probabilities[1].item()),
                    'P_Moderate': float(probabilities[2].item()),
                    'P_Over': float(probabilities[3].item()),
                    'max_prob': max_probability,
                }
                for key, values in tensor_values.items():
                    row[key] = float(values[index].item())
                rows.append(row)
    model.train(was_training)

    fieldnames = [
        'image_path',
        'true_label',
        'pred_label',
        'correct',
        'abs_error',
        'distribution',
        'P_Pre',
        'P_Slight',
        'P_Moderate',
        'P_Over',
        'alpha',
        'beta',
        'mean',
        'var',
        'a',
        'b',
        'mu',
        'sigma',
        'mean_proxy',
        'max_prob',
    ]
    csv_path = trainer.output_dir / 'test_predictions_probabilistic.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    def mean_or_none(values):
        return statistics.mean(values) if values else None

    return {
        'probabilistic_predictions_csv': str(csv_path),
        'probabilistic_head_accuracy': (
            probabilistic_correct / total_samples if total_samples else 0.0
        ),
        'correct_mean_variance': mean_or_none(variance_correct),
        'wrong_mean_variance': mean_or_none(variance_wrong),
        'correct_mean_max_prob': mean_or_none(max_probability_correct),
        'wrong_mean_max_prob': mean_or_none(max_probability_wrong),
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
    result.update(save_probabilistic_prediction_csv(trainer))
    result.update(save_ordinal_representation_artifacts(trainer))
    return result


def save_json(path: Path, data: Any) -> None:
    """使用 UTF-8 和便于阅读的缩进保存 JSON。"""
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(data), file, ensure_ascii=False, indent=2)


def save_confusion_matrix_csvs(
    run_directory: Path,
    metrics: Dict[str, Any],
) -> None:
    '''为单次实验保存原始和行归一化混淆矩阵 CSV。'''
    class_names = list(metrics.get('class_names') or [])
    matrix_specs = (
        ('confusion_matrix', 'confusion_matrix.csv'),
        ('normalized_confusion_matrix', 'normalized_confusion_matrix.csv'),
    )
    for metric_key, filename in matrix_specs:
        matrix = metrics.get(metric_key) or []
        if not class_names or not matrix:
            continue
        with (run_directory / filename).open(
            'w',
            encoding='utf-8-sig',
            newline='',
        ) as file:
            writer = csv.writer(file)
            writer.writerow(['true/pred', *class_names])
            for class_name, row in zip(class_names, matrix):
                writer.writerow([class_name, *row])


def _result_to_csv_row(
    result: Dict[str, Any],
    batch_timestamp: str,
    fold_name: str | None = None,
) -> Dict[str, Any]:
    '''把单模型结果压平成适合消融汇总 CSV 的一行。'''
    metrics = result.get('metrics') or {}
    row = {
        'batch_timestamp': batch_timestamp,
        'model_name': result.get('model_name'),
        'status': result.get('status'),
        'accuracy': metrics.get('accuracy'),
        'macro_f1': metrics.get('macro_f1'),
        'mae': metrics.get('mae'),
        'qwk': metrics.get('qwk'),
        'parameters_total': metrics.get('parameters_total'),
        'parameters_trainable': metrics.get('parameters_trainable'),
        'flops': metrics.get('flops'),
        'flops_g': metrics.get('flops_g'),
        'training_time_seconds': metrics.get('training_time_seconds'),
        'inference_time_seconds': metrics.get('inference_time_seconds'),
        'inference_ms_per_sample': metrics.get('inference_ms_per_sample'),
        'adjacent_error_rate': metrics.get('adjacent_error_rate'),
        'distant_error_rate': metrics.get('distant_error_rate'),
        'acc_0_1': metrics.get('acc_0_1'),
        'acc_1_2': metrics.get('acc_1_2'),
        'acc_2_3': metrics.get('acc_2_3'),
        'error_0_to_1': metrics.get('error_0_to_1'),
        'error_1_to_0': metrics.get('error_1_to_0'),
        'error_1_to_2': metrics.get('error_1_to_2'),
        'error_2_to_1': metrics.get('error_2_to_1'),
        'error_2_to_3': metrics.get('error_2_to_3'),
        'error_3_to_2': metrics.get('error_3_to_2'),
        'far_error_count': metrics.get('far_error_count'),
        'probabilistic_head_accuracy': metrics.get('probabilistic_head_accuracy'),
        'correct_mean_variance': metrics.get('correct_mean_variance'),
        'wrong_mean_variance': metrics.get('wrong_mean_variance'),
        'correct_mean_max_prob': metrics.get('correct_mean_max_prob'),
        'wrong_mean_max_prob': metrics.get('wrong_mean_max_prob'),
        'class_wise_metrics': json.dumps(
            metrics.get('class_wise_metrics') or {},
            ensure_ascii=False,
        ),
        'adjacent_confusions': json.dumps(
            metrics.get('adjacent_confusions') or {},
            ensure_ascii=False,
        ),
        'boundary_accuracies': json.dumps(
            metrics.get('boundary_accuracies') or {},
            ensure_ascii=False,
        ),
        'config_path': result.get('config_path'),
        'run_directory': result.get('run_directory'),
    }
    if fold_name is not None:
        row['fold'] = fold_name
    return row


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    '''以 UTF-8 BOM 保存表格，便于 Excel 直接打开中文字段。'''
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_ablation_csv_files(
    runs_root: Path,
    batch_timestamp: str,
    results: List[Dict[str, Any]],
) -> List[Path]:
    '''按 multi-scale、neck、combined 分组生成附件要求的汇总 CSV。'''
    baseline_results = [
        result for result in results if result.get('model_name') == 'baseline'
    ]
    multiscale_results = baseline_results + [
        result
        for result in results
        if str(result.get('model_name', '')).startswith(('msk_', 'msd_', 'msp_', 'msh_'))
    ]
    neck_results = baseline_results + [
        result
        for result in results
        if str(result.get('model_name', '')).startswith('neck_')
    ]
    final_results = [
        result
        for result in results
        if str(result.get('model_name', '')).startswith('combined_')
    ]
    ordinal_opcl_results = [
        result
        for result in results
        if 'OrdinalOPCL/' in str(result.get('config_path', ''))
    ]
    probing_capacity_results = [
        result
        for result in results
        if 'ProbingCapacityStageRepro/' in str(result.get('config_path', ''))
    ]
    probabilistic_ordinal_results = [
        result
        for result in results
        if 'ProbabilisticOrdinalHeads/' in str(result.get('config_path', ''))
    ]
    signal_stability_results = [
        result
        for result in results
        if 'SignalStability3Seeds/' in str(result.get('config_path', ''))
    ]
    controlled_single_variable_results = [
        result
        for result in results
        if 'ControlledSingleVariable/' in str(result.get('config_path', ''))
    ]
    final_feature_refinement_results = [
        result
        for result in results
        if 'FinalFeatureRefinement/' in str(result.get('config_path', ''))
    ]
    written_paths: List[Path] = []
    for filename, group_results, fold_name in (
        ('multiscale_ablation_summary.csv', multiscale_results, None),
        ('multiscale_ablation_folds.csv', multiscale_results, 'fixed_split'),
        ('neck_ablation_summary.csv', neck_results, None),
        ('neck_ablation_folds.csv', neck_results, 'fixed_split'),
        ('final_combination_summary.csv', final_results, None),
        ('ordinal_opcl_summary.csv', ordinal_opcl_results, None),
        ('ordinal_opcl_folds.csv', ordinal_opcl_results, 'fixed_split'),
        (
            'probing_capacity_stage_repro_summary.csv',
            probing_capacity_results,
            None,
        ),
        (
            'probing_capacity_stage_repro_folds.csv',
            probing_capacity_results,
            'fixed_split',
        ),
        (
            'probabilistic_ordinal_summary.csv',
            probabilistic_ordinal_results,
            None,
        ),
        (
            'probabilistic_ordinal_folds.csv',
            probabilistic_ordinal_results,
            'fixed_split',
        ),
        (
            'signal_stability_3seeds_summary.csv',
            signal_stability_results,
            None,
        ),
        (
            'signal_stability_3seeds_folds.csv',
            signal_stability_results,
            'fixed_split',
        ),
        (
            'controlled_single_variable_summary.csv',
            controlled_single_variable_results,
            None,
        ),
        (
            'controlled_single_variable_folds.csv',
            controlled_single_variable_results,
            'fixed_split',
        ),
        (
            'final_feature_refinement_summary.csv',
            final_feature_refinement_results,
            None,
        ),
        (
            'final_feature_refinement_folds.csv',
            final_feature_refinement_results,
            'fixed_split',
        ),
    ):
        rows = [
            _result_to_csv_row(result, batch_timestamp, fold_name)
            for result in group_results
        ]
        if rows:
            path = runs_root / filename
            _write_csv_rows(path, rows)
            written_paths.append(path)

    seed_groups: Dict[str, List[Dict[str, Any]]] = {}
    for result in probing_capacity_results + signal_stability_results:
        model_name = str(result.get('model_name', ''))
        match = re.fullmatch(r'(.+)_seed([123])', model_name)
        if match and result.get('status') == 'success':
            seed_groups.setdefault(match.group(1), []).append(result)
    seed_rows = []
    metric_names = ('accuracy', 'macro_f1', 'mae', 'qwk')
    for model_family, family_results in sorted(seed_groups.items()):
        row: Dict[str, Any] = {
            'batch_timestamp': batch_timestamp,
            'model_family': model_family,
            'seed_count': len(family_results),
        }
        for metric_name in metric_names:
            values = [
                float(result['metrics'][metric_name])
                for result in family_results
                if (result.get('metrics') or {}).get(metric_name) is not None
            ]
            row[f'{metric_name}_mean'] = (
                statistics.mean(values) if values else None
            )
            row[f'{metric_name}_std'] = (
                statistics.stdev(values) if len(values) >= 2 else 0.0
            )
        seed_rows.append(row)
    if seed_rows:
        seed_summary_path = runs_root / 'seed_reproduction_summary.csv'
        _write_csv_rows(seed_summary_path, seed_rows)
        written_paths.append(seed_summary_path)
    return written_paths


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
        'parameters_trainable',
        'flops_g',
        'training_time_seconds',
        'inference_ms_per_sample',
        'adjacent_error_rate',
        'distant_error_rate',
        'acc_0_1',
        'acc_1_2',
        'acc_2_3',
        'adjacent_error_count',
        'far_error_count',
        'hard_adjacent_sample_count',
        'hard_adjacent_sample_rate',
        'probabilistic_head_accuracy',
        'correct_mean_variance',
        'wrong_mean_variance',
        'correct_mean_max_prob',
        'wrong_mean_max_prob',
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


def render_boundary_accuracies(metrics: Dict[str, Any]) -> str:
    '''渲染按真实标签筛选、但保留原始四分类预测的相邻边界准确率。'''
    rows = []
    for metric_name, values in (metrics.get('boundary_accuracies') or {}).items():
        rows.append(
            '<tr>'
            f'<td>{html.escape(str(metric_name))}</td>'
            f'<td>{float(values["accuracy"]):.4%}</td>'
            f'<td>{int(values["correct"])}</td>'
            f'<td>{int(values["total"])}</td>'
            '</tr>'
        )
    if not rows:
        return '<p class="muted">没有相邻边界准确率。</p>'
    return (
        '<p class="muted">仅按真实标签筛选相邻两类；预测仍为原始四分类，'
        '预测到其他类别同样计错。</p>'
        '<table><thead><tr><th>指标</th><th>准确率</th><th>正确数</th>'
        '<th>样本数</th></tr></thead><tbody>'
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


def should_keep_pth_files(config: TrainingConfig) -> bool:
    '''读取公共训练开关；未配置时保持向后兼容，默认保留权重。'''
    return bool(getattr(config.train, 'keep_pth_files', True))


def cleanup_pth_files(run_directory: Path, keep_pth_files: bool) -> List[str]:
    '''最终评估完成后按开关删除该实验目录中的所有 .pth 文件。'''
    if keep_pth_files:
        return []
    removed_files = []
    for checkpoint_path in run_directory.glob('*.pth'):
        checkpoint_path.unlink(missing_ok=True)
        removed_files.append(checkpoint_path.name)
    return removed_files


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
    keep_pth_files = True if config is None else should_keep_pth_files(config)
    checkpoint_entry = (
        '<a href="best_model.pth">最佳模型</a>'
        if keep_pth_files
        else '<span class="muted">最佳模型未保留（keep_pth_files=false）</span>'
    )
    training_curve_path = run_directory / "training_curves.png"
    if training_curve_path.exists():
        training_curve_entry = '<a href="training_curves.png">训练曲线 PNG</a>'
        training_curve_section = (
            '<section><h2>训练曲线</h2>'
            '<p class="muted">蓝色为训练集，红色为验证集；MAE 越低越好，QWK 越高越好。</p>'
            '<a href="training_curves.png">'
            '<img class="training-curve" src="training_curves.png" alt="训练曲线">'
            '</a></section>'
        )
    else:
        training_curve_entry = '<span class="muted">暂无训练曲线</span>'
        training_curve_section = (
            '<section><h2>训练曲线</h2>'
            '<p class="muted">尚未完成任何 epoch，或绘图过程失败，因此没有可用曲线。</p>'
            '</section>'
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
    .training-curve {{ display:block; width:100%; height:auto; border:1px solid var(--line); border-radius:10px; }}
  </style>
</head>
<body><main>
  <header>
    <h1>{html.escape(model_name)} 训练报告</h1>
    <p><span class="status {status_class}">{html.escape(status)}</span></p>
    <p class="muted">开始：{started_at.strftime('%Y-%m-%d %H:%M:%S')}　结束：{finished_at.strftime('%Y-%m-%d %H:%M:%S')}　耗时：{duration_seconds:.1f} 秒</p>
    <p><a href="config.yaml">实际配置</a> · {checkpoint_entry} · {training_curve_entry} · <a href="test_metrics.json">测试指标 JSON</a> · <a href="history.json">训练历史 JSON</a> · <a href="train.log">训练日志</a></p>
  </header>
  {error_section}
  <section><h2>测试集关键指标</h2><div class="metrics">{render_metric_cards(metrics)}</div></section>
  <section><h2>每类准确率</h2><table><thead><tr><th>类别</th><th>准确率</th></tr></thead><tbody>{per_class_rows}</tbody></table></section>
  <section><h2>逐类 Precision / Recall / F1</h2>{render_class_wise_metrics(metrics)}</section>
  <section><h2>测试集混淆矩阵</h2>{render_confusion_matrix(metrics)}</section>
  <section><h2>测试集归一化混淆矩阵</h2>{render_normalized_confusion_matrix(metrics)}</section>
  <section><h2>相邻边界准确率</h2>{render_boundary_accuracies(metrics)}</section>
  <section><h2>相邻严重程度混淆</h2>{render_adjacent_confusions(metrics)}</section>
  {training_curve_section}
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
    initial_checkpoint_from = config.get('initial_checkpoint_from')
    if initial_checkpoint_from is not None and not isinstance(initial_checkpoint_from, str):
        raise ValueError(
            f'initial_checkpoint_from 必须是配置名字符串：{relative_path}'
        )
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
    if isinstance(model_config.get('loss'), dict):
        runtime_config['loss'] = copy.deepcopy(model_config['loss'])
    if model_config.get('random_seed') is not None:
        runtime_config['random_seed'] = int(model_config['random_seed'])
    # 单模型 YAML 可以只覆盖二阶段训练所需字段，而不复制整份公共配置。
    for section_name in ('train', 'optimizer', 'scheduler'):
        section_override = model_config.get(section_name)
        if not isinstance(section_override, dict):
            continue
        if not isinstance(runtime_config.get(section_name), dict):
            runtime_config[section_name] = {}
        runtime_config[section_name].update(copy.deepcopy(section_override))
    for orchestration_key in ('initial_checkpoint_from', 'hard_adjacent_mining'):
        if model_config.get(orchestration_key) is not None:
            runtime_config[orchestration_key] = copy.deepcopy(
                model_config[orchestration_key]
            )

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
    initial_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    capture_best_state: bool = False,
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
    captured_best_state: Optional[Dict[str, torch.Tensor]] = None
    hard_mining_metrics: Dict[str, Any] = {}

    try:
        separator = '=' * 80
        print(
            f'\n{separator}\n开始训练：{model_name}'
            f'\n模型配置：{relative_config_path}'
            f'\n输出目录：{run_directory}\n{separator}'
        )
        trainer = Trainer(config=config, device=device)
        if initial_state_dict is not None:
            trainer.model.load_state_dict(initial_state_dict, strict=True)
            trainer.logger.info(
                'initial_checkpoint_loaded',
                source=model_config.get('initial_checkpoint_from'),
            )
        hard_mining_config = model_config.get('hard_adjacent_mining')
        if isinstance(hard_mining_config, dict) and bool(
            hard_mining_config.get('enabled', False)
        ):
            sample_weights, hard_mining_metrics = identify_hard_adjacent_samples(
                trainer,
                probability_margin=float(
                    hard_mining_config.get('probability_margin', 0.15)
                ),
                hard_weight=float(hard_mining_config.get('hard_weight', 2.0)),
            )
            trainer.set_sample_weights_by_path(sample_weights)
        training_started_at = time.perf_counter()
        trainer.train()
        training_time_seconds = time.perf_counter() - training_started_at
        history = to_builtin(trainer.history)
        metrics = evaluate_best_checkpoint(trainer, training_time_seconds)
        metrics.update(hard_mining_metrics)
        if capture_best_state:
            captured_best_state = {
                key: value.detach().cpu().clone()
                for key, value in trainer.model.state_dict().items()
            }
        save_json(run_directory / 'history.json', history)
        save_json(run_directory / 'test_metrics.json', metrics)
        save_confusion_matrix_csvs(run_directory, metrics)
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
    keep_pth_files = should_keep_pth_files(config)
    removed_pth_files = cleanup_pth_files(run_directory, keep_pth_files)
    metrics['keep_pth_files'] = keep_pth_files
    metrics['removed_pth_files'] = removed_pth_files
    if metrics:
        save_json(run_directory / 'test_metrics.json', metrics)
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
        'parameters_trainable': metrics.get('parameters_trainable'),
        'flops_g': metrics.get('flops_g'),
        'run_directory': str(run_directory),
        'report_path': str(report_path),
        'metrics': metrics,
        'error': error_text,
    }
    if captured_best_state is not None:
        # 仅供当前批次的依赖实验使用，不写入 JSON/CSV，也不要求保留 PTH。
        result['_best_state_dict'] = captured_best_state
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
    pth_group = parser.add_mutually_exclusive_group()
    pth_group.add_argument(
        '--keep-pth',
        '--keep-pth-files',
        dest='keep_pth_files',
        action='store_true',
        help='本次运行保留每个实验的 best_model.pth。',
    )
    pth_group.add_argument(
        '--discard-pth',
        '--discard-pth-files',
        dest='keep_pth_files',
        action='store_false',
        help='本次运行完成测试和报告后删除每个实验的所有 .pth。',
    )
    parser.set_defaults(keep_pth_files=None)
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
    keep_pth_files = (
        PYCHARM_KEEP_PTH_FILES
        if args.keep_pth_files is None
        else bool(args.keep_pth_files)
    )
    common_config['train']['keep_pth_files'] = keep_pth_files
    model_entries = [
        (relative_path, load_model_config(relative_path))
        for relative_path in selected_paths
    ]
    selected_model_names = [str(config['name']) for _, config in model_entries]
    selected_model_indices = {
        model_name: index
        for index, model_name in enumerate(selected_model_names)
    }
    required_checkpoint_sources = set()
    for dependent_index, (_, model_config) in enumerate(model_entries):
        source_name = model_config.get('initial_checkpoint_from')
        if source_name is None:
            continue
        source_index = selected_model_indices.get(str(source_name))
        if source_index is None:
            raise ValueError(
                f'{model_config["name"]} 依赖 {source_name}，'
                '请在 --models 中同时选择该来源配置。'
            )
        if source_index >= dependent_index:
            raise ValueError(
                f'{model_config["name"]} 的来源 {source_name} '
                '必须排在 CONFIG_LIST 更前面。'
            )
        required_checkpoint_sources.add(str(source_name))

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
    seed_by_model = [
        (relative_path.stem, int(getattr(config, 'random_seed', 42)))
        for (relative_path, _), config in zip(model_entries, preview_configs)
    ]
    unique_seeds = sorted({seed for _, seed in seed_by_model})
    if len(unique_seeds) == 1:
        print(f'随机种子：{unique_seeds[0]}')
    else:
        print('本次包含多个随机种子：')
        for model_name, seed in seed_by_model:
            print(f'  - {model_name}: {seed}')
    epoch_by_model = [
        (relative_path.stem, int(config.train.epochs))
        for (relative_path, _), config in zip(model_entries, preview_configs)
    ]
    unique_epochs = sorted({epochs for _, epochs in epoch_by_model})
    if len(unique_epochs) == 1:
        print(f'训练轮数：{unique_epochs[0]}')
    else:
        print('本次包含不同训练轮数：')
        for model_name, epochs in epoch_by_model:
            print(f'  - {model_name}: {epochs}')
    print(
        '保留 PTH：'
        + str(bool(common_config['train'].get('keep_pth_files', True)))
    )
    print(f'训练设备：{device}')

    if args.dry_run:
        print('\ndry-run 完成：全部 YAML 和数据均已通过检查，未创建 runs 或训练模型。')
        return

    runs_root.mkdir(parents=True, exist_ok=True)
    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results: List[Dict[str, Any]] = []
    checkpoint_state_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    for relative_path, model_config in model_entries:
        model_name = str(model_config['name'])
        source_name = model_config.get('initial_checkpoint_from')
        initial_state_dict = None
        if source_name is not None:
            initial_state_dict = checkpoint_state_cache.get(str(source_name))
            if initial_state_dict is None:
                raise RuntimeError(
                    f'{model_name} 无法开始：来源实验 {source_name} '
                    '没有成功生成最佳权重。'
                )
        result = run_config_file(
            common_config,
            relative_path,
            model_config,
            dataset_root,
            runs_root,
            device,
            initial_state_dict=initial_state_dict,
            capture_best_state=model_name in required_checkpoint_sources,
        )
        captured_state = result.pop('_best_state_dict', None)
        if captured_state is not None:
            checkpoint_state_cache[model_name] = captured_state
        results.append(result)
        if result['status'] == 'failed' and args.fail_fast:
            break

    summary_path = write_batch_summary(runs_root, batch_timestamp, results)
    csv_paths = write_ablation_csv_files(runs_root, batch_timestamp, results)
    for csv_path in csv_paths:
        print(f'消融结果 CSV：{csv_path}')
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
