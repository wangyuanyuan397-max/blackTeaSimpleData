"""
File: src/utils/analysis.py
Purpose: evaluation and error-analysis helpers.
Provides: metrics aggregation, confusion matrix assets, and report utilities.
Notes: designed to work safely inside training/evaluation loops.
"""

import os
import torch
import shutil
import csv
import numpy as np
from tqdm import tqdm
from .metrics import compute_metrics
from .visualization import plot_confusion_matrix, plot_training_curves
import base64
import re
try:
    import markdown
except ImportError:
    markdown = None

def _rank_probas_to_logits(rank_probas, eps=1e-4):
    rank_probas = rank_probas.clamp(min=eps, max=1.0 - eps)
    return torch.log(rank_probas) - torch.log1p(-rank_probas)


def _rank_probas_to_class_probs(rank_probas):
    first = 1.0 - rank_probas[..., :1]
    last = rank_probas[..., -1:]
    middle = rank_probas[..., :-1] - rank_probas[..., 1:]
    class_probs = torch.cat([first, middle, last], dim=-1)
    class_probs = class_probs.clamp_min(0.0)
    return class_probs / class_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _class_probs_to_rank_probas(class_probs):
    tail_sums = torch.cumsum(torch.flip(class_probs, dims=[-1]), dim=-1)
    return torch.flip(tail_sums, dims=[-1])[..., 1:]


def _aggregate_tta_logits(logits, batch_size, num_crops, mode="mean", topk=0, ordinal=False, ordinal_get_label=None):
    """
    Aggregate multi-crop logits with the same policy as Evaluator.
    Supported modes: mean, vote, topk.
    """
    mode = (mode or "mean").lower()

    if ordinal:
        rank_probs = torch.sigmoid(logits).view(batch_size, num_crops, -1)
        class_probs = _rank_probas_to_class_probs(rank_probs)
        num_classes = class_probs.size(-1)

        if mode == "vote":
            if ordinal_get_label is None:
                raise ValueError("ordinal_get_label is required when ordinal=True and mode='vote'")
            preds = ordinal_get_label(logits).view(batch_size, num_crops)
            counts = torch.zeros(batch_size, num_classes, device=logits.device, dtype=rank_probs.dtype)
            ones = torch.ones_like(preds, dtype=rank_probs.dtype)
            counts.scatter_add_(1, preds, ones)
            agg_rank_probs = _class_probs_to_rank_probas(counts / float(num_crops))
        elif mode == "topk":
            if topk <= 0:
                agg_rank_probs = rank_probs.mean(dim=1)
            else:
                k = min(int(topk), num_crops)
                conf = class_probs.max(dim=2).values
                topk_idx = conf.topk(k, dim=1).indices
                gathered = rank_probs.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, rank_probs.size(-1)))
                agg_rank_probs = gathered.mean(dim=1)
        else:
            agg_rank_probs = rank_probs.mean(dim=1)

        return _rank_probas_to_logits(agg_rank_probs)

    probs = torch.softmax(logits, dim=1).view(batch_size, num_crops, -1)
    num_classes = probs.size(-1)

    if mode == "vote":
        preds = probs.argmax(dim=2)
        counts = torch.zeros(batch_size, num_classes, device=probs.device, dtype=probs.dtype)
        ones = torch.ones_like(preds, dtype=probs.dtype)
        counts.scatter_add_(1, preds, ones)
        agg = counts / float(num_crops)
    elif mode == "topk":
        if topk <= 0:
            agg = probs.mean(dim=1)
        else:
            k = min(int(topk), num_crops)
            conf = probs.max(dim=2).values
            topk_idx = conf.topk(k, dim=1).indices
            gathered = probs.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, num_classes))
            agg = gathered.mean(dim=1)
    else:
        agg = probs.mean(dim=1)

    return torch.log(agg.clamp_min(1e-12))


def analyze_and_save_errors(
    model,
    dataloader,
    device,
    output_dir,
    classes,
    history=None,
    logger=None,
    split_name="validation",
    tta_mode="mean",
    tta_topk=0,
    save_error_images=True,
):
    """
    Runs inference, identifies errors, saves images and generates a comprehensive report.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader returning (inputs, labels, paths)
        device: torch.device
        output_dir: Directory to save results
        classes: List of class names
        history: (Optional) Dict containing training history for plotting curves
        logger: (Optional) Logger instance to use for output. If None, uses print.
        save_error_images: Whether to copy misclassified samples into error_analysis/images
    """
    def log_msg(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    model.eval()
    ordinal_head = hasattr(model, "head") and hasattr(model.head, "get_label")
    
    # Containers for full evaluation
    all_preds = []
    all_labels = []
    errors = []
    
    analysis_dir = os.path.join(output_dir, "error_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    images_dir = None
    if save_error_images:
        images_dir = os.path.join(analysis_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

    split_name = str(split_name or "validation")
    log_msg(
        f"[Analysis] Starting comprehensive analysis on split={split_name} "
        f"(tta_mode={tta_mode}, tta_topk={tta_topk}, save_error_images={save_error_images})..."
    )
    
    # 显式清理显存，确保有足够空间
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="[Analysis]"):
            # Handle both (inputs, labels) and (inputs, labels, paths)
            if len(batch_data) == 4:
                inputs, labels, paths, _ = batch_data
            elif len(batch_data) == 3:
                inputs, labels, paths = batch_data
            elif len(batch_data) == 2:
                inputs, labels = batch_data
                # Generate placeholder paths if not provided
                paths = [f"sample_{i}" for i in range(len(labels))]
            else:
                raise ValueError(f"Unexpected batch format: got {len(batch_data)} elements")
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # [Optimization] Mini-batch processing to avoid OOM
            # Error analysis often uses larger batches or multi-crop inputs which can explode memory
            # 使用极小的 mini-batch 确保安全 (ViT/Swin 显存占用大)
            mini_batch_size = 4
            outputs_list = []
            
            try:
                # Check if inputs are 5D (Batch, Crops, C, H, W) or 4D (Batch, C, H, W)
                if inputs.dim() == 5:
                    bsz, ncrops, c, h, w = inputs.shape
                    # Flatten: (Batch * Crops, C, H, W)
                    flat_inputs = inputs.reshape(-1, c, h, w)
                    
                    # Process flat inputs in mini-batches
                    num_samples = flat_inputs.size(0)
                    mini_outputs = []
                    
                    for i in range(0, num_samples, mini_batch_size):
                        batch_slice = flat_inputs[i:i + mini_batch_size]
                        out = model(batch_slice)
                        # Handle tuple output
                        if isinstance(out, tuple):
                            out = out[0]
                        mini_outputs.append(out)
                        
                        # 及时清理
                        del batch_slice, out
                        torch.cuda.empty_cache()
                    
                    # Concatenate all mini-batch outputs
                    raw_outputs = torch.cat(mini_outputs, dim=0)
                    
                    outputs = _aggregate_tta_logits(
                        raw_outputs,
                        batch_size=bsz,
                        num_crops=ncrops,
                        mode=tta_mode,
                        topk=tta_topk,
                        ordinal=ordinal_head,
                        ordinal_get_label=model.head.get_label if ordinal_head else None,
                    )
                
                else:
                    # Standard 4D inputs (Batch, C, H, W)
                    num_samples = inputs.size(0)
                    for i in range(0, num_samples, mini_batch_size):
                        batch_slice = inputs[i:i + mini_batch_size]
                        out = model(batch_slice)
                        if isinstance(out, tuple):
                            out = out[0]
                        outputs_list.append(out)
                        
                        # 及时清理
                        del batch_slice, out
                        torch.cuda.empty_cache()
                        
                    outputs = torch.cat(outputs_list, dim=0)
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log_msg(f"⚠️ Error analysis skipped for a batch due to OOM: {e}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            # [Fix] 处理 Ordinal Head 等自定义分类头
            if ordinal_head:
                predictions = model.head.get_label(outputs)
                # 对于 Ordinal，置信度可以简单取 Sigmoid 后的平均值或最大值
                # 这里为了简单，我们计算 Sigmoid 后的"确信度"
                rank_probs = torch.sigmoid(outputs)
                class_probs = _rank_probas_to_class_probs(rank_probs)
                # 这里的置信度定义比较模糊，暂且取 max(probs) 近似
                # 注意：Ordinal 的 outputs 并不是多分类的 logits，所以 softmax 也不适用
                # 为了保持代码运行，我们暂时用全 1.0 代替，或者根据具体业务逻辑修改
                confidences = torch.max(class_probs, dim=1)[0]
            else:
                # 标准分类流程
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                confidences, predictions = probs.max(1)
            
            # Store for metrics
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Find errors
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predictions[i].item()
                confidence = confidences[i].item()
                path = paths[i]
                
                if true_label != pred_label:
                    # It's an error
                    file_name = os.path.basename(path) if os.path.exists(str(path)) else path
                    # classes may be int labels in csv_dataset mode.
                    true_class = str(classes[true_label])
                    pred_class = str(classes[pred_label])
                    
                    # Record error
                    errors.append({
                        "filename": file_name,
                        "true_label": true_class,
                        "pred_label": pred_class,
                        "confidence": confidence,
                        "original_path": path,
                        "saved_name": ""
                    })
                    
                    # Copy image to analysis folder (optional)
                    if save_error_images and images_dir and os.path.exists(str(path)):
                        # Naming convention: {conf:.3f}_pred_{pred}_true_{true}_{filename}
                        safe_pred = pred_class.replace(" ", "_")
                        safe_true = true_class.replace(" ", "_")
                        
                        # Ensure filename doesn't contain invalid characters
                        safe_file_name = re.sub(r'[\\/*?:"<>|]', "", file_name)
                        
                        new_name = f"{confidence:.3f}_pred_{safe_pred}_true_{safe_true}_{safe_file_name}"
                        errors[-1]["saved_name"] = new_name
                        
                        dest_path = os.path.join(images_dir, new_name)
                        try:
                            shutil.copy2(path, dest_path)
                        except Exception as e:
                            log_msg(f"[Warning] Failed to copy {path}: {e}")

    # --- 1. Compute & Save Metrics ---
    # 定义期望的类别顺序（从常量导入，可在配置中覆盖）
    try:
        from .constants import get_class_order
        desired_order = get_class_order()
    except ImportError:
        desired_order = classes

    # 检查所有期望的类别是否都在当前类别列表中
    if all(c in classes for c in desired_order):
        old_indices = [classes.index(c) for c in desired_order]
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(old_indices)}
        def map_labels(arr):
            return np.array([mapping.get(x, x) for x in arr])
            
        all_labels_mapped = map_labels(np.array(all_labels))
        all_preds_mapped = map_labels(np.array(all_preds))
        
        metrics = compute_metrics(all_labels_mapped, all_preds_mapped, desired_order)
        plot_classes = desired_order
    else:
        log_msg(f"[Info] Using dataset's natural class order: {classes}")
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), classes)
        plot_classes = classes
    
    # Plot Confusion Matrix
    cm_path = os.path.join(analysis_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics['confusion_matrix'], plot_classes, cm_path)
    
    # Plot Training Curves (if history provided)
    if history:
        curves_path = os.path.join(analysis_dir, "training_curves.png")
        plot_training_curves(history, curves_path)

    # --- 2. Generate Report ---
    
    # Save Error CSV (if any)
    if errors:
        # Sort by confidence (descending)
        errors.sort(key=lambda x: x["confidence"], reverse=True)
        
        csv_path = os.path.join(analysis_dir, "error_report.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "true_label", "pred_label", "confidence", "original_path", "saved_name"])
            writer.writeheader()
            writer.writerows(errors)
    
    # Generate Markdown Report
    md_path = os.path.join(analysis_dir, "evaluation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Evaluation Report\n\n")
        
        # Section 1: Overall Metrics
        f.write("## 1. Overall Metrics\n\n")
        f.write(f"- **Split**: {split_name}\n")
        f.write(f"- **Num Samples**: {len(all_labels)}\n")
        f.write(f"- **TTA Mode**: {tta_mode}\n")
        f.write(f"- **Save Error Images**: {save_error_images}\n")
        f.write(f"- **Accuracy**: {metrics['accuracy']:.2%}\n")
        f.write(f"- **±1 Accuracy**: {metrics.get('plus_minus_one_accuracy', 0.0):.2%}\n")
        
        # Weighted avg from sklearn report
        if metrics['report']:
            f.write(f"- **Weighted F1-Score**: {metrics['report']['weighted avg']['f1-score']:.4f}\n")
            f.write(f"- **Weighted Recall**: {metrics['report']['weighted avg']['recall']:.4f}\n")
            f.write(f"- **Weighted Precision**: {metrics['report']['weighted avg']['precision']:.4f}\n")
        
        f.write("\n### Detailed Classification Report\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        if metrics['report']:
            for cls_name in plot_classes:
                if cls_name in metrics['report']:
                    row = metrics['report'][cls_name]
                    f.write(f"| {cls_name} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {row['support']} |\n")
        
        # Section 2: Visualizations
        f.write("\n## 2. Visualizations\n\n")
        f.write("### Confusion Matrix\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n\n")
        
        if history:
            f.write("### Training Curves (Loss & Accuracy)\n")
            f.write("![Training Curves](training_curves.png)\n\n")
        
        # Section 3: Error Analysis
        f.write(f"## 3. Error Analysis (Total Errors: {len(errors)})\n\n")
        if not errors:
            f.write(f"No errors found on the {split_name} set!\n")
        else:
            f.write("The following table shows the top 50 misclassified images sorted by **confidence** (High Confidence Errors).\n\n")
            if save_error_images:
                f.write("| Confidence | True Label | Pred Label | Image |\n")
                f.write("| --- | --- | --- | --- |\n")
                for row in errors[:50]:
                    rel_img_path = f"images/{row['saved_name']}"
                    rel_img_path_enc = rel_img_path.replace(" ", "%20")
                    f.write(f"| **{row['confidence']:.4f}** | {row['true_label']} | {row['pred_label']} | ![{row['filename']}]({rel_img_path_enc}) |\n")
            else:
                f.write("| Confidence | True Label | Pred Label | Filename |\n")
                f.write("| --- | --- | --- | --- |\n")
                for row in errors[:50]:
                    f.write(f"| **{row['confidence']:.4f}** | {row['true_label']} | {row['pred_label']} | {row['filename']} |\n")
            
    log_msg(f"[Analysis] Markdown report saved to {analysis_dir}")

    # --- 3. Convert to Self-Contained HTML ---
    try:
        html_path = os.path.join(analysis_dir, "evaluation_report_full.html")
        stats = embed_images_to_html(md_path, html_path)
        
        log_msg(f"[Analysis] ✓ HTML Report generated successfully!")
        log_msg(f"           Path: {html_path}")
        log_msg(f"           Images embedded: {stats['images_embedded']}/{stats['images_found']}")
        log_msg(f"           File size: {stats['output_size_kb']:.2f} KB")
        
        if stats['images_failed'] > 0:
            log_msg(f"           [Warning] {stats['images_failed']} images failed to embed")
            
    except ImportError as e:
        log_msg(f"[Warning] Cannot generate HTML report: {e}")
        log_msg(f"          Please install: pip install markdown")
    except Exception as e:
        log_msg(f"[Warning] Failed to generate HTML report: {e}")
        import traceback
        log_msg(f"          {traceback.format_exc()}")

def embed_images_to_html(md_path, output_path, dataset_urls_path=None):
    """
    将 Markdown 文件转换为自包含的 HTML 文件（支持URL或Base64嵌入）
    
    功能特性：
        - 优先使用Google Drive URL（如果dataset_urls.json存在）
        - Fallback到Base64嵌入（向后兼容）
        - 支持多种图片格式（PNG, JPG, BMP, GIF, WEBP）
        - 生成美观的响应式 HTML 报告
        
    参数：
        md_path (str): Markdown 文件路径
        output_path (str): 输出 HTML 文件路径
        dataset_urls_path (str): dataset_urls.json文件路径（可选）
        
    抛出：
        ImportError: 如果 markdown 库未安装
        FileNotFoundError: 如果 Markdown 文件不存在
    """
    # 检查依赖
    if markdown is None:
        raise ImportError(
            "'markdown' library not found. Please install it:\n"
            "  pip install markdown"
        )
    
    # 检查输入文件
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    
    # 读取 Markdown 内容
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
    except Exception as e:
        raise IOError(f"Failed to read Markdown file: {e}")
    
    md_dir = os.path.dirname(md_path)
    
    # 🔧 加载URL映射（如果存在）
    url_mapping = {}
    use_urls = False
    
    if dataset_urls_path is None:
        # 尝试从项目根目录加载
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(md_dir)))
        dataset_urls_path = os.path.join(project_root, "dataset_urls.json")
    
    if os.path.exists(dataset_urls_path):
        try:
            import json
            with open(dataset_urls_path, 'r', encoding='utf-8') as f:
                url_mapping = json.load(f)
            use_urls = True
            print(f"[Info] Loaded {len(url_mapping)} image URLs from {dataset_urls_path}")
        except Exception as e:
            print(f"[Warning] Failed to load URL mapping: {e}. Using base64 fallback.")
    
    # 统计信息
    images_found = 0
    images_embedded = 0
    images_from_url = 0
    images_failed = 0
    
    def replace_image(match):
        """
        图片替换回调函数
        
        将 Markdown 图片语法 ![alt](path) 替换为 URL链接 或 Base64 嵌入的 <img> 标签
        """
        nonlocal images_found, images_embedded, images_from_url, images_failed
        
        images_found += 1
        
        alt_text = match.group(1)
        image_rel_path = match.group(2)
        
        # 处理 URL 编码的路径（例如 %20 代表空格）
        image_rel_path_clean = image_rel_path.replace("%20", " ")
        
        # 🔧 方案1: 尝试使用Google Drive URL
        if use_urls:
            # 从images/xxx.jpg提取相对于数据集的路径
            # 例如: images/0.999_pred_轻微发酵_true_发酵前_001.bmp
            # 需要找到原始数据集中的路径
            
            # 尝试从文件名推断原始路径
            # 假设文件名格式: {conf}_pred_{pred}_true_{true}_{original_filename}
            filename = os.path.basename(image_rel_path_clean)
            
            # 尝试各种可能的数据集路径
            possible_paths = []
            for split in ['train', 'val', 'test']:
                for cls in ['发酵前', '轻微发酵', '适度发酵', '中度发酵']:
                    # 查找包含原始文件名的URL
                    for url_key, url_info in url_mapping.items():
                        if filename in url_key or os.path.basename(url_key) in filename:
                            images_from_url += 1
                            return (
                                f'<img src="{url_info["url"]}" '
                                f'alt="{alt_text}" '
                                f'title="{alt_text} (from Google Drive)" '
                                f'style="max-width:100%; height:auto; border: 1px solid #ddd; '
                                f'border-radius: 4px; margin: 10px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" '
                                f'loading="lazy" '
                                f'crossorigin="anonymous">'
                            )
        
        # 🔧 方案2: Fallback到Base64嵌入
        # 构建绝对路径
        image_abs_path = os.path.join(md_dir, image_rel_path_clean)
        
        # 检查文件是否存在
        if not os.path.exists(image_abs_path):
            images_failed += 1
            return (
                f'<div style="padding: 10px; background-color: #fff3cd; border: 1px solid #ffc107; '
                f'border-radius: 4px; margin: 10px 0;">'
                f'⚠️ <strong>Image not found:</strong> {image_rel_path_clean}'
                f'</div>'
            )
        
        try:
            # 读取图片文件（二进制模式）
            with open(image_abs_path, "rb") as f:
                img_data = f.read()
            
            # 转换为 Base64
            b64_data = base64.b64encode(img_data).decode('utf-8')
            
            # 确定图片格式
            ext = os.path.splitext(image_abs_path)[1].lower().replace('.', '')
            
            # 标准化扩展名
            ext_mapping = {
                'jpg': 'jpeg',
                'jpe': 'jpeg',
                'bmp': 'bmp',
                'png': 'png',
                'gif': 'gif',
                'webp': 'webp',
                'svg': 'svg+xml'
            }
            mime_type = ext_mapping.get(ext, ext)
            
            # 计算文件大小（用于调试）
            file_size_kb = len(img_data) / 1024
            
            images_embedded += 1
            
            # 生成 HTML img 标签
            return (
                f'<img src="data:image/{mime_type};base64,{b64_data}" '
                f'alt="{alt_text}" '
                f'title="{alt_text} ({file_size_kb:.1f} KB)" '
                f'style="max-width:100%; height:auto; border: 1px solid #ddd; '
                f'border-radius: 4px; margin: 10px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" '
                f'loading="lazy">'
            )
            
        except Exception as e:
            images_failed += 1
            return (
                f'<div style="padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; '
                f'border-radius: 4px; margin: 10px 0;">'
                f'❌ <strong>Error processing image:</strong> {image_rel_path}<br>'
                f'<small>{str(e)}</small>'
                f'</div>'
            )
    
    # 正则表达式匹配 Markdown 图片语法: ![alt](path)
    pattern = r'!\[(.*?)\]\((.*?)\)'
    
    # 步骤 1: 替换所有图片为 Base64 嵌入的 <img> 标签
    md_with_images = re.sub(pattern, replace_image, md_content)
    
    # 步骤 2: 将 Markdown 转换为 HTML
    try:
        html_body = markdown.markdown(
            md_with_images, 
            extensions=['tables', 'fenced_code', 'nl2br']
        )
    except Exception as e:
        raise RuntimeError(f"Failed to convert Markdown to HTML: {e}")
    
    # 步骤 3: 生成完整的 HTML 文档（带样式）
    final_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="BlackTeaFermentation Analysis Tool">
    <title>Evaluation Report - Model Performance Analysis</title>
    <style>
        /* ========== 全局样式 ========== */
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, 
                         "Helvetica Neue", Arial, "Noto Sans", sans-serif;
            line-height: 1.6;
            color: #24292e;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        
        /* ========== 标题样式 ========== */
        h1, h2, h3, h4 {{
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }}
        
        h1 {{
            font-size: 2.5em;
            border-bottom: 3px solid #0366d6;
            padding-bottom: 0.3em;
            color: #0366d6;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        
        h2 {{
            font-size: 2em;
            border-bottom: 2px solid #6a737d;
            padding-bottom: 0.3em;
            color: #24292e;
        }}
        
        h3 {{
            font-size: 1.5em;
            color: #586069;
        }}
        
        /* ========== 表格样式 ========== */
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1.5rem 0;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        th, td {{
            border: 1px solid #e1e4e8;
            padding: 12px 16px;
            text-align: left;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }}
        
        tr:nth-child(even) {{
            background-color: #f6f8fa;
        }}
        
        tr:hover {{
            background-color: #e1e4e8;
            transition: background-color 0.2s ease;
        }}
        
        /* ========== 图片样式 ========== */
        img {{
            display: block;
            margin: 20px auto;
            border-radius: 8px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        img:hover {{
            transform: scale(1.02);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            cursor: zoom-in;
        }}
        
        /* ========== 代码样式 ========== */
        code {{
            background-color: rgba(27,31,35,0.05);
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 0.9em;
        }}
        
        pre {{
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
            border: 1px solid #e1e4e8;
        }}
        
        /* ========== 列表样式 ========== */
        ul, ol {{
            padding-left: 2em;
            margin: 1em 0;
        }}
        
        li {{
            margin: 0.5em 0;
        }}
        
        /* ========== 强调样式 ========== */
        strong {{
            color: #0366d6;
            font-weight: 600;
        }}
        
        em {{
            color: #6a737d;
        }}
        
        /* ========== 链接样式 ========== */
        a {{
            color: #0366d6;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        /* ========== 响应式设计 ========== */
        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            h2 {{
                font-size: 1.5em;
            }}
            
            table {{
                font-size: 0.9em;
            }}
            
            th, td {{
                padding: 8px;
            }}
        }}
        
        /* ========== 页脚样式 ========== */
        .footer {{
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 2px solid #e1e4e8;
            text-align: center;
            color: #6a737d;
            font-size: 0.9em;
        }}
        
        /* ========== 统计信息卡片 ========== */
        .stats-card {{
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }}
    </style>
</head>
<body>
    {html_body}
    
    <div class="footer">
        <p>
            <strong>Report Generated:</strong> {os.path.basename(output_path)}<br>
            <strong>Images Embedded:</strong> {images_embedded} / {images_found} 
            {f'({images_failed} failed)' if images_failed > 0 else '✓'}
        </p>
        <p style="font-size: 0.8em; color: #959da5;">
            Generated by BlackTeaFermentation Analysis Tool | 
            Self-contained HTML Report (No external dependencies)
        </p>
    </div>
</body>
</html>
"""
    
    # 步骤 4: 写入 HTML 文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
    except Exception as e:
        raise IOError(f"Failed to write HTML file: {e}")
    
    # 返回统计信息（用于日志）
    return {
        'images_found': images_found,
        'images_embedded': images_embedded,
        'images_from_url': images_from_url,
        'images_failed': images_failed,
        'output_size_kb': len(final_html.encode('utf-8')) / 1024,
        'using_urls': use_urls
    }
