# Soft supervision diagnostics

这个临时实验目录用于在 `datasets_split_patches` 四分类任务上比较软监督策略。当前重点已经从自蒸馏转到 SLS：Structural Label Smoothing，结构化标签平滑。

核心问题仍然是：

```text
moderate 和 over 的局部视觉特征高度重叠，普通 CE 容易学得过度自信。
```

## 当前默认实验

默认实验现在是：

- `ce_baseline`
- `label_smoothing_eps0.1`
- `sls_c16_a0.1_b0.2`
- `sls_c32_a0.1_b0.2`
- `sls_c64_a0.1_b0.2`
- `reverse_sls_c32_a0.1_b0.2`

额外可选实验包括：

- `label_smoothing_eps0.05`
- `label_smoothing_eps0.2`
- `bootstrap_soft_beta0.8_warm30`
- `self_distill_t2_alpha0.5`
- `self_distill_t2_alpha0.7`
- `self_distill_t2_alpha0.9`
- `ce_retrain_same_cost`

默认不保存 `.pth`；只有显式加 `--keep-pth` 才会保存权重。

## SLS 在这里怎么实现

工程版 SLS 流程：

```text
train patch
→ 冻结 ImageNet ResNet-50 提取特征
→ StandardScaler
→ PCA 到 128 维
→ KMeans 聚类
→ 每个簇内用 MST 跨类别边比例估计类别混杂程度
→ 混杂越强，该簇 alpha 越大
→ 用样本级 soft label 训练普通四分类模型
```

只使用训练集建立区域，不使用验证集和测试集，避免数据泄漏。

每个 SLS 实验会额外保存：

```text
sls_summary.json
sls_cluster_summary.csv
sls_sample_assignments.csv
```

用来检查每个簇的类别比例、overlap score 和最终分配到的 alpha。

## 先做 dry-run

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --dry-run --device cpu --num-workers 0
```

## 推荐第一轮：三模型三 seed，多 C 值 + 反向对照

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --num-workers 0 --models mambaout_tiny resnet50 convnext_tiny --experiments ce_baseline label_smoothing_eps0.1 sls_c16_a0.1_b0.2 sls_c32_a0.1_b0.2 sls_c64_a0.1_b0.2 reverse_sls_c32_a0.1_b0.2 --seeds 2026 2027 2028
```

这一轮回答三个问题：

1. SLS 是否稳定优于 CE？
2. SLS 是否稳定优于统一 Label Smoothing？
3. 正向 SLS 是否优于 Reverse-SLS？

如果 `sls_c32_a0.1_b0.2` 和 `reverse_sls_c32_a0.1_b0.2` 差不多，说明“重叠区域更强平滑”这个方向未被验证。

## 可选第二轮：统一 LS 的 alpha 对照

如果想确认统一标签平滑本身是不是已经足够，可以跑：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --num-workers 0 --models mambaout_tiny resnet50 convnext_tiny --experiments ce_baseline label_smoothing_eps0.05 label_smoothing_eps0.1 label_smoothing_eps0.2 sls_c32_a0.1_b0.2 reverse_sls_c32_a0.1_b0.2 --seeds 2026 2027 2028
```

## 结果位置

结果保存到：

```text
temp/soft_supervision_diagnostics/results/
```

批次级汇总：

```text
summary.csv
summary.html
aggregate_summary.csv
aggregate_summary.html
```

其中 `aggregate_summary.*` 是按 `model + experiment` 汇总的均值和标准差。

重点看：

- `accuracy`
- `macro_f1`
- `qwk`
- `acc_2_3`
- `moderate_recall`
- `over_recall`
- `moderate_to_over_count`
- `over_to_moderate_count`

## 常用命令

列出模型：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --list-models
```

列出实验：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --list-experiments
```

只跑一个模型：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --num-workers 0 --models mambaout_tiny
```

只跑 SLS C=32：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --num-workers 0 --models mambaout_tiny --experiments ce_baseline label_smoothing_eps0.1 sls_c32_a0.1_b0.2 reverse_sls_c32_a0.1_b0.2 --seeds 2026 2027 2028
```

Windows 上建议一直显式写 `--num-workers 0`，避免 PyTorch DataLoader 的共享内存映射错误。
