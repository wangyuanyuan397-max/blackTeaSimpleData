# Soft Supervision Diagnostics 交接文档

这份交接只针对：

```text
temp/soft_supervision_diagnostics/
```

写给完全没有上下文的新会话。

## 我们在做什么

我们在 `datasets_split_patches` 四分类 patch 数据集上，验证软监督方法是否能改善红茶发酵程度分类，尤其是 `moderate` 和 `over` 两类边界过硬、互相误判的问题。

当前自蒸馏已经被用户判断为“没效果，不管它了”。本目录现在重点转向：

```text
Structural Label Smoothing，SLS，结构化标签平滑
```

这不是时间点二分类诊断。时间点诊断在另一个目录：

```text
temp/moderate_over_boundary_diagnostics/
```

本目录关注完整四分类：

```text
pre = 0
slight = 1
moderate = 2
over = 3
```

重点指标：

- 总体 `accuracy`
- `macro_f1`
- `qwk`
- `acc_2_3`，即只看 moderate/over 子集的准确率
- `moderate_recall`
- `over_recall`
- `moderate_to_over_count`
- `over_to_moderate_count`

## 已经完成了什么

主脚本：

```text
temp/soft_supervision_diagnostics/run_soft_supervision_experiments.py
```

使用说明：

```text
temp/soft_supervision_diagnostics/README.md
```

本交接：

```text
temp/soft_supervision_diagnostics/HANDOFF.md
```

结果和日志已经被 `.gitignore` 忽略：

```text
temp/soft_supervision_diagnostics/results/
temp/soft_supervision_diagnostics/logs/
```

脚本已支持的模型：

```text
mambaout_tiny
resnet50
convnext_tiny
safnet_imagenet
```

脚本已支持的实验：

```text
ce_baseline
ce_retrain_same_cost
label_smoothing_eps0.05
label_smoothing_eps0.1
label_smoothing_eps0.2
sls_c16_a0.1_b0.2
sls_c32_a0.1_b0.2
sls_c64_a0.1_b0.2
reverse_sls_c32_a0.1_b0.2
bootstrap_soft_beta0.8_warm30
self_distill_t2_alpha0.5
self_distill_t2_alpha0.7
self_distill_t2_alpha0.9
```

默认实验列表已经改为 SLS 方向：

```text
ce_baseline
label_smoothing_eps0.1
sls_c16_a0.1_b0.2
sls_c32_a0.1_b0.2
sls_c64_a0.1_b0.2
reverse_sls_c32_a0.1_b0.2
```

默认训练参数：

```text
epochs = 150
batch_size = 32
val_batch_size = 64
test_batch_size = 64
num_workers = 0
patience = 30
lr = 1e-4
weight_decay = 5e-4
warmup_epochs = 2
min_lr = 1e-6
image_size = 224
seed = 2026
```

默认不保存 `.pth`。只有显式加：

```text
--keep-pth
```

才会保存权重。

## SLS 当前怎么实现

这是工程版 SLS，不是训练自动编码器的论文原始复刻。

流程：

```text
训练集 patch
→ 冻结 ImageNet ResNet-50 提取特征
→ L2 normalize
→ StandardScaler
→ PCA 到 128 维
→ KMeans 聚类
→ 每个簇内计算 MST 跨类别边比例
→ 跨类别边比例越高，说明局部类别越混杂
→ 混杂区域分配更大的 label smoothing alpha
→ 用样本级 soft label 训练普通四分类模型
```

只使用训练集建立区域，不碰验证集和测试集，避免数据泄漏。

每个 SLS run 会额外输出：

```text
sls_summary.json
sls_cluster_summary.csv
sls_sample_assignments.csv
```

其中：

- `sls_summary.json`：整体 SLS 配置、实际 alpha 均值、alpha 范围、overlap score 范围。
- `sls_cluster_summary.csv`：每个簇的样本数、类别数量、类别比例、overlap score、alpha。
- `sls_sample_assignments.csv`：每个训练样本属于哪个簇、使用多少 alpha。

## Reverse-SLS 是什么

`reverse_sls_c32_a0.1_b0.2` 是反向对照。

正常 SLS：

```text
高混杂区域 → alpha 大
低混杂区域 → alpha 小
```

Reverse-SLS：

```text
高混杂区域 → alpha 小
低混杂区域 → alpha 大
```

如果正常 SLS 比 Reverse-SLS 稳定更好，才说明“重叠区域应该更强平滑”这个方向是有信息量的。

## 当前卡在哪

当前不是代码实现卡住，而是需要正式跑实验。

已知历史坑：Windows + PyTorch DataLoader 多 worker 曾经触发：

```text
RuntimeError: Couldn't open shared file mapping: <torch_...>, error code: <1455>
```

因此脚本默认：

```text
num_workers = 0
```

命令里仍建议显式写：

```text
--num-workers 0
```

## 下一步计划

第一步，先做 dry-run：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --dry-run --device cpu --num-workers 0
```

第二步，正式跑三模型三 seed，多 C 值 + 反向对照：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --num-workers 0 --models mambaout_tiny resnet50 convnext_tiny --experiments ce_baseline label_smoothing_eps0.1 sls_c16_a0.1_b0.2 sls_c32_a0.1_b0.2 sls_c64_a0.1_b0.2 reverse_sls_c32_a0.1_b0.2 --seeds 2026 2027 2028
```

第三步，看批次汇总：

```text
temp/soft_supervision_diagnostics/results/batch_YYYYMMDD_HHMMSS/summary.csv
temp/soft_supervision_diagnostics/results/batch_YYYYMMDD_HHMMSS/aggregate_summary.csv
temp/soft_supervision_diagnostics/results/batch_YYYYMMDD_HHMMSS/aggregate_summary.html
```

判断逻辑：

- SLS 是否稳定优于 `ce_baseline`。
- SLS 是否稳定优于 `label_smoothing_eps0.1`。
- 正向 SLS 是否优于 `reverse_sls_c32_a0.1_b0.2`。
- C=16/32/64 是否结果方向一致；如果只有某个 C 有效，要小心偶然性。

第四步，可选跑统一 LS alpha 对照：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --num-workers 0 --models mambaout_tiny resnet50 convnext_tiny --experiments ce_baseline label_smoothing_eps0.05 label_smoothing_eps0.1 label_smoothing_eps0.2 sls_c32_a0.1_b0.2 reverse_sls_c32_a0.1_b0.2 --seeds 2026 2027 2028
```

## 绝对不要再踩的坑

1. 不要在 Windows 上用多 worker 正式跑。

   用：

   ```text
   --num-workers 0
   ```

2. 不要默认保存 `.pth`。

   用户明确不想保留权重。除非用户主动要求，否则不要加：

   ```text
   --keep-pth
   ```

3. 不要把 SLS 的聚类/BER 估计用到 val/test。

   SLS 区域只能用训练集建立，否则会数据泄漏。

4. 不要把当前 SLS 说成论文原始完整复刻。

   当前是工程适配：

   ```text
   冻结 ResNet-50 特征 + PCA + KMeans + MST overlap proxy
   ```

   原论文更接近：

   ```text
   自动编码器特征 + KMeans + Henze-Penrose/Bayes error bound
   ```

5. 不要只看一次 seed。

   用户已经明确说“可以多试几个，以免偶然”。建议至少：

   ```text
   --seeds 2026 2027 2028
   ```

6. 不要只比较 SLS 和 CE。

   必须同时看统一 LS 和 Reverse-SLS，否则不知道提升来自“平滑本身”还是“结构化分配方向”。

7. 不要混淆两个实验目录。

   ```text
   temp/soft_supervision_diagnostics/
   ```

   是四分类软监督。

   ```text
   temp/moderate_over_boundary_diagnostics/
   ```

   是按时间点做 moderate/over 边界诊断。

8. 不要提交 `results/` 或 `logs/`。

   它们已经在 `.gitignore`，如果 git status 里出现大量结果文件，先查 `.gitignore`。

## 文件职责

`run_soft_supervision_experiments.py`

主实验脚本。负责构建数据集、模型、loss、训练、测试、SLS 区域分析、保存指标和报告。

`README.md`

给用户看的简短使用说明和推荐命令。

`HANDOFF.md`

给新会话看的完整交接。

`results/`

正式实验输出目录，git 忽略。

`logs/`

后台训练日志目录，git 忽略。
