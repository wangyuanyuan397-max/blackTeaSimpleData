"""
??: src/utils/metrics.py
??: ???????
????: ????????????????????
????: ??????????????????
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

def compute_metrics(all_labels: np.ndarray,
                    all_predictions: np.ndarray,
                    classes: List[str]) -> Dict[str, Any]:
    """
    计算图像分类任务的评估指标。
    
    参数:
        all_labels (np.ndarray): 真实标签数组。
                                 形状: (N,)，内容为 0 到 num_classes-1 的整数索引。
        all_predictions (np.ndarray): 预测标签数组。
                                      形状: (N,)，内容同上。
        classes (List[str]): 类别名称列表。
                             例如: ["Unfermented", "Light", "Moderate", "Heavy"]。
                             
    返回:
        metrics (dict): 包含各项指标的字典。
            - class_metrics: 每个类别的 Precision, Recall, F1。
            - confusion_matrix: 混淆矩阵 (List[List[int]])。
            - report: sklearn 生成的详细分类报告。
            - accuracy: 全局准确率。
    """
    metrics = {}
    
    # --- 1. 计算每个类别的详细指标 (Per-class Metrics) ---
    class_metrics = {}
    for class_idx, class_name in enumerate(classes):
        # TP (True Positive): 预测为该类，且实际也为该类
        tp = np.sum((all_labels == class_idx) & (all_predictions == class_idx))
        
        # FP (False Positive): 预测为该类，但实际不是该类 (误报)
        fp = np.sum((all_labels != class_idx) & (all_predictions == class_idx))
        
        # FN (False Negative): 预测不是该类，但实际是该类 (漏报)
        fn = np.sum((all_labels == class_idx) & (all_predictions != class_idx))
        
        # Precision (查准率) = TP / (TP + FP)
        # 含义：在模型预测为“红茶”的样本中，有多少真的是红茶？
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall (查全率) = TP / (TP + FN)
        # 含义：在所有真实的“红茶”样本中，模型找出了多少？
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-Score: Precision 和 Recall 的调和平均数
        # 综合反映了模型的稳健性。当 Precision 和 Recall 都不错时，F1 才高。
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'count': int(np.sum(all_labels == class_idx)) # 该类别的真实样本数
        }
    
    metrics['class_metrics'] = class_metrics
    
    # --- 2. 计算混淆矩阵 (Confusion Matrix) ---
    # 行表示真实类别 (True Label)，列表示预测类别 (Predicted Label)。
    # 对角线上的数字表示分类正确的数量。
    cm = confusion_matrix(all_labels, all_predictions)
    metrics['confusion_matrix'] = cm.tolist() # 转为 List 以便序列化为 JSON/YAML
    
    # --- 3. 生成 sklearn 标准分类报告 ---
    # 包含 macro avg (宏平均), weighted avg (加权平均) 等汇总指标
    try:
        report = classification_report(all_labels, all_predictions, target_names=classes, output_dict=True)
        metrics['report'] = report
    except:
        metrics['report'] = None
        
    # --- 4. 计算全局准确率 (Overall Accuracy) ---
    # 预测正确的总数 / 样本总数
    correct = np.sum(all_labels == all_predictions)
    total = len(all_labels)
    metrics['accuracy'] = float(correct) / total if total > 0 else 0.0
    # 对有序分类补充 ±1 accuracy：预测成相邻类别也算正确。
    metrics['plus_minus_one_accuracy'] = float(np.mean(np.abs(all_labels - all_predictions) <= 1)) if total > 0 else 0.0
    
    return metrics
