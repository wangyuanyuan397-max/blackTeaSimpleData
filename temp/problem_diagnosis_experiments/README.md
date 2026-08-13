# 红茶五分类任务诊断实验

这是一套与主工程隔离的诊断性尝试。代码全部位于本目录，不注册 `src` 组件，不修改现有训练配置。目标不是继续给 MixNet-S 堆模块，而是先用受控实验回答：判别证据依赖什么、尺度是否互补、空间布局与颜色/纹理是否重要。

## 已实现的证据链

1. **Global 与物理视野尺度**：整张原图 resize 作为 Global；另从同一原图随机裁出 `408/306/204/102` 像素视野，全部 resize 到 `224×224`，分别训练同一个 MixNet-S。
2. **尺度互补**：候选单尺度/双尺度只在验证集排序；组合冻结后才评估一次测试集。
3. **单局部与多局部**：每个局部尺度同时报告中心单 crop 和固定多 crop 概率平均，避免把视野尺度与集成增益混为一谈。
4. **局部证据分布**：对同一 checkpoint 做 `10%～50%` 的随机矩形遮挡，默认重复 3 次。
5. **空间关系**：`3×3 patch shuffle` 保留局部像素内容但破坏全局布局。
6. **颜色与纹理**：灰度、Gaussian blur、综合色扰动，以及亮度/对比度/饱和度/色相/白平衡独立扰动和“粗区域内细块打乱”的 texture shuffle。
7. **有序错误**：每个条件自动输出 confusion matrix、MAE、QWK、相邻错误占比和远距离错误占比。
8. **来源组审计**：解析 `4-1/4-2` 这类文件名，检查来源组是否跨 train/val/test，并对错误富集做 Fisher 检验与 BH-FDR 校正。
9. **多随机种子**：只复现 `408/204/102` 三个优先尺度，报告均值、样本标准差与范围。

所有随机破坏都由“图片路径 + 条件 + 重复编号 + seed”生成稳定随机数，重复执行不会悄悄换一套扰动。
所有扰动都在 resize 后的模型输入画布上执行，因此 Global 与 Local 的 blur、shuffle、遮挡强度具有同一像素尺度。

## 为什么默认用原图目录

默认数据为：

```text
datasets_01234_original_split/
├── train/00..40
├── val/00..40
└── test/00..40
```

该目录每类为固定划分的原始 `2448×2048` 图像。同一原图随机产生多个训练 crop，但 train/val/test 间不会共享原图。这样比较不同 crop 尺度时，改变的是“模型看到的物理范围”，不是提前生成的数据集差异。

如果改用 `datasets_01234_408` 等派生 patch 数据，工具仍会按文件名恢复原图 ID，并同时输出：

- `sample_*`：patch 级指标；
- `parent_*`：同一原图所有 patch 概率平均后的原图级指标。

正式结论优先使用 `parent_*`。否则 55 个高度相关 patch 会被误当成 55 个独立样本。

## 0. PyCharm 直接运行

所有入口都不读取命令行参数。每个脚本顶部都有带中文注释的配置区，路径根据脚本位置自动解析，因此 PyCharm 的 Working directory 设在哪里都可以。

最简单的方法是打开 `run_all_experiments.py`，右键 **Run 'run_all_experiments'**。它会自动完成：

```text
尺度训练 → 受控扰动 → 中文报告
```

总入口顶部的三个布尔开关可以跳过已经完成的阶段，`PERTURBATION_CROP_SIZE` 选择要诊断的尺度（默认 `204`，Global 可填 `"global"`）。详细训练参数仍在对应阶段脚本顶部配置。

如果希望分阶段运行，推荐顺序：

1. 打开 `run_crop_experiments.py`，修改顶部 `CropExperimentConfig`（通常默认值即可），右键 **Run 'run_crop_experiments'**；
2. 右键运行 `audit_source_groups.py`，先核查文件名来源组；
3. 右键运行 `select_fusion_on_validation.py`，在验证集选组合并冻结测试；
4. 右键运行 `run_selected_scales_multiseed.py`，复现精选尺度；
5. 右键运行 `run_color_component_experiments.py`，拆分颜色依赖；
6. 如需其他受控破坏，再运行 `run_perturbation_experiments.py`；
7. 最后运行 `summarize_diagnosis.py` 生成原诊断汇总。

请在 PyCharm 中选择已安装项目依赖的解释器。依赖复用项目现有 `requirements.txt`，无需安装额外包。

## 1. 先做冒烟测试

在 `run_crop_experiments.py` 顶部临时修改：

```python
crop_sizes = (408, 204)
epochs = 1
batch_size = 2
eval_batch_size = 2
num_workers = 0
train_repeats = 1
val_views = 1
test_views = 1
use_pretrained = False
max_samples_per_class = 1
dry_run = True
```

右键运行即可。冒烟结束后，应把这些值恢复为正式配置，尤其是：

```python
use_pretrained = True
max_samples_per_class = None
dry_run = False
```

## 2. 正式训练尺度诊断模型

`run_crop_experiments.py` 顶部已预置正式配置：

```python
crop_sizes = (408, 306, 204, 102)
include_global = True
input_size = 224
epochs = 150
train_repeats = 30
val_views = 5
test_views = 9
batch_size = 32
eval_batch_size = 16
device = "auto"
evaluate_fusions = False
```

确认后直接右键运行该文件。

说明：

- `train-repeats=30` 表示每轮每张原图产生 30 个随机 crop，不会在磁盘复制数据；
- `crop_global` 保留整张原图后 resize，作为真正的全局对照；
- 每个局部 checkpoint 会分别生成 `test_single` 与 `test_multi<N>` 结果；
- 各尺度都送入相同的 `224×224` 网络输入，架构与像素张量大小保持一致；
- 各尺度重新设置相同随机种子；
- early stopping 与最佳权重选择使用验证集原图聚合准确率；
- 最优权重保存为 `results/crop_scale/crop_<size>/best.pth`；
- `summary.csv` 默认只含单中心与多局部结果；不会在测试集上枚举融合。
- 各指标附带 Wilson 95% 区间；扰动结果另含同一批原图的配对得失数和精确 McNemar p 值。

旧版把多个融合组合都放到测试集比较，会产生测试集选择偏差。只有为了复现旧结果时，才临时把 `evaluate_fusions=True`；该结果必须标为探索性，不能当作无偏性能。

原图很大、训练 IO 压力高。如显存或磁盘吞吐不足，可先把 `train-repeats` 调到 10、`batch-size` 调小；正式对比时所有尺度必须用相同值。

## 3. 验证集选融合，冻结测试

先确保 `crop_408/crop_306/crop_204/crop_102` 下已有 `best.pth`，然后右键运行 `select_fusion_on_validation.py`。该脚本：

- 默认复用用户给出的 `E:\docs\...\results\crop_scale`；该路径不存在时回退到本目录 `results/crop_scale`；
- 只在 val 上比较单尺度和双尺度平均概率；
- 用 Accuracy 排序，平局依次看 QWK、Macro-F1、NLL，并优先较少组件；
- 胜出组合确定后才创建 test 数据集；
- 同时报告验证集选出的最佳单尺度，并对融合增益做原图级精确 McNemar 检验。

输出位于：

```text
results/frozen_fusion/
├── validation_ranking.csv
├── FROZEN_FUSION_RESULT.md
├── FROZEN_FUSION_RESULT.json
├── winner_validation/
├── frozen_test/
└── frozen_test_best_single/
```

注意：当前 val 只有 20 张原图，而且同一 val 也参与了 checkpoint early stopping；排名仍会有较大离散性，冻结流程只是消除脚本内部直接用测试集反向选组合的问题。现有 test 此前已经被旧实验查看过，因此这次属于回顾性纠偏，不能称为全新盲测。

## 4. 文件名来源组审计

右键运行 `audit_source_groups.py`。输出位于 `results/source_group_audit/`，包括来源组跨划分计数、各条件分组错误率、错误明细及中文报告。

代码只能确认第一段编号构成 6 个稳定的 `source_group`，不能仅凭 `4-1` 猜它是批次、采集轮次或样本来源。必须先查原始实验记录；只有语义确认后，才能把它作为 group-held-out 的分组键。

## 5. 精选尺度多随机种子

右键运行 `run_selected_scales_multiseed.py`。默认配置为：

```python
scales = (408, 204, 102)
seeds = (2026, 3407, 42)
```

每个 seed 独立训练，只报告各尺度的 single/multi-view，不枚举测试集融合。输出 `per_seed_results.csv`、`multiseed_summary.csv` 与 `MULTISEED_REPORT.md`。这一步计算量大；脚本可跳过已经生成完整 `summary.csv` 的 seed，但一个未完成 seed 会从该 seed 的第一个尺度重跑。

## 6. 颜色分量独立扰动

直接右键运行 `run_color_component_experiments.py`。默认复用已有 `crop_408/best.pth`，并先在 `val` 上使用 5 个固定区域聚合。无需命令行参数。

默认条件为：

```text
Original
Brightness factor 0.70, 0.85, 1.15, 1.30
Contrast factor 0.70, 0.85, 1.15, 1.30
Saturation factor 0.50, 0.75, 1.25, 1.50
Hue shift -0.06, -0.03, +0.03, +0.06
White balance/temperature -0.20, -0.10, +0.10, +0.20
```

其中正色温表示偏暖，负色温表示偏冷；变换会近似保持整图平均亮度，减少和 brightness 的混杂。每个条件只启用一个算子，不再把四种颜色变化混在同一个 `color_jitter` 中。

输出位于：

```text
results/color_components/val/
├── COLOR_COMPONENT_REPORT.md
├── summary.csv
├── summary.json
└── 各条件预测明细/
```

报告同时使用 Accuracy、NLL、QWK、原图级得失数和精确 McNemar p，并对20个颜色条件统一报告 BH-FDR q 值。当前 val 只有 20 张，单张就是 5 个百分点，因此还要重点看正负方向及强度增加时是否存在一致剂量趋势。

颜色变换可能产生训练分布外图像，所以明显掉点首先表示“模型对该颜色算子敏感/不鲁棒”，不能只凭这一项就断言自然图像中的因果机制。最好再结合真实设备、照明或白平衡变化的数据验证。

在 val 上固定最终强度和判读规则后，如确实需要最后一次测试，必须同时修改：

```python
split = "test"
frozen_test_conditions = (
    ("brightness", 0.70),
    ("hue", -0.06),
    # 此处仅填写根据 val 预先确定的条件。
)
```

若冻结条件为空，脚本会拒绝访问 test；它也会拒绝未出现在 val 候选配置中的强度，防止在测试集继续枚举调参。不要根据 test 掉点幅度再次修改条件。

## 7. 受控破坏实验

`run_perturbation_experiments.py` 顶部默认诊断 `crop_204/best.pth`：

```python
checkpoint = EXPERIMENT_DIR / "results" / "crop_scale" / "crop_204" / "best.pth"
split = "test"
views = 9
repeats = 3
device = "auto"
```

如需诊断 Global 或其他尺度，修改 `checkpoint` 一行，然后右键运行。脚本会从 checkpoint 自动读取 crop 大小、输入大小和模型信息。

默认运行：

```text
Original
Occlusion 10%, 20%, 30%, 40%, 50%
Grayscale
Gaussian blur radius 1.5, 3.0
Color jitter strength 0.3
Patch shuffle 3×3
Texture shuffle macro 2×2, micro 4×4
```

随机条件按重复编号分别保存，`summary.csv` 给出相对 Original 的原图准确率下降。

建议先把顶部 `split` 改成 `"val"` 确定扰动强度；测试集只做最后一次冻结评估，避免反复窥视测试结果。

## 8. 生成诊断报告

直接右键运行 `summarize_diagnosis.py`。

生成：

```text
results/DIAGNOSIS_REPORT.md
results/DIAGNOSIS_REPORT.json
```

报告会：

- 找出最佳单尺度；若读取到旧融合结果，仅把它标为测试集反向选择的探索线索；
- 对随机扰动重复计算均值和标准差；
- 明确提醒哪些现象只能算“初步信号”，不能直接写成机制结论。

冻结融合的正式结论以 `results/frozen_fusion/FROZEN_FUSION_RESULT.md` 为准，不要用旧报告中的最佳测试集融合。

## 9. 结果怎么解释

建议使用以下证据组合，而不是孤立看一个数字：

| 假设 | 至少需要的互补证据 |
| --- | --- |
| 局部证据重要 | 中等 crop 优于大视野；遮挡稳定掉点；多视图优于单视图 |
| 全局与局部互补 | 单尺度各自有效；融合稳定优于最佳单尺度；不同 seed 可复现 |
| 空间布局重要 | patch shuffle 在重复后稳定大幅下降；但颜色直方图等仍基本保留 |
| 纹理重要 | blur/texture shuffle 稳定下降；灰度结果可帮助区分颜色与纹理贡献 |
| 亮度/色调/饱和度依赖 | 对应独立颜色分量在正负方向或剂量增加时稳定恶化，并由 NLL、配对得失共同支持 |
| 相邻阶段边界模糊 | 相邻错误占全部错误比例高；MAE/QWK 与 confusion matrix 一致支持 |

不要把 `Local branch +2%`、某次遮挡掉点或一次融合涨点单独写成因果证明。正式结论建议补 3 个训练随机种子，并以原图为独立单位做配对统计。

## 当前范围

本目录已经补齐来源组审计、验证集冻结融合和精选尺度多 seed 复现。`random split vs batch-held-out` 仍需要可靠的业务语义：当前 manifest 只有时间点和原图编号，没有明确批次字段，因此审计脚本只输出 `source_group` 假设，不会擅自生成 batch-held-out 结论。
