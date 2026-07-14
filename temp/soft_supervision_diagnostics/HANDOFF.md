# Soft Supervision Diagnostics 交接文档

这份交接只针对：

```text
temp/soft_supervision_diagnostics/
```

写给完全没有上下文的新会话。

## 我们在做什么

我们在 `datasets_split_patches` 四分类 patch 数据集上，验证软监督方法是否能改善红茶发酵程度分类，尤其是 `moderate` 和 `over` 两类边界过硬、互相误判的问题。

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

对应 YAML：

```text
configs/fixed_split_patches_models/mambaout_tiny.yaml
configs/fixed_split_patches_models/resnet50.yaml
configs/fixed_split_patches_models/convnext_tiny.yaml
configs/fixed_split_patches_models/safnet_imagenet.yaml
```

脚本已支持的实验：

```text
ce_baseline
ce_retrain_same_cost
label_smoothing_eps0.1
bootstrap_soft_beta0.8_warm30
self_distill_t2_alpha0.7
self_distill_t2_alpha0.5
self_distill_t2_alpha0.9
```

默认实验列表是：

```text
ce_baseline
ce_retrain_same_cost
label_smoothing_eps0.1
bootstrap_soft_beta0.8_warm30
self_distill_t2_alpha0.7
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

注意：`num_workers` 已改成默认 0，因为 Windows 上之前触发过 PyTorch DataLoader 共享内存映射错误。

默认不保存 `.pth`。只有显式加：

```text
--keep-pth
```

才会保存权重。

## 这次新实现的方案

用户希望按下面思路验证“当前自蒸馏正结果是否可靠”：

1. 对 `MambaOut`、`ResNet-50`、`ConvNeXt` 各跑 3 个随机种子。
2. 看均值和标准差，而不是只看单次结果。
3. 增加 `CE-long / CE-retrain` 等训练量对照。
4. 暂时不要大范围搜索 `T` 和 `alpha`。
5. 如果重复实验稳定，再固定 `T=2`，只做 `alpha ∈ {0.5, 0.7, 0.9}` 的单变量实验。

对应代码已实现：

- 新增 `--seeds` 参数。
- 每个 seed 单独落在：

  ```text
  results/batch_YYYYMMDD_HHMMSS/seed_2026/
  results/batch_YYYYMMDD_HHMMSS/seed_2027/
  results/batch_YYYYMMDD_HHMMSS/seed_2028/
  ```

- 新增 `ce_retrain_same_cost`。
- 新增 `self_distill_t2_alpha0.5` 和 `self_distill_t2_alpha0.9`。
- 批次结束后生成：

  ```text
  summary.csv
  summary.html
  aggregate_summary.csv
  aggregate_summary.html
  ```

其中 `aggregate_summary.*` 是按 `model_name + experiment_name` 计算的均值和标准差。

## ce_retrain_same_cost 是什么

自蒸馏实际训练了两段：

```text
CE teacher -> 重新初始化 student -> CE + KD
```

普通 `ce_baseline` 只训练一段：

```text
CE model
```

所以如果自蒸馏提升，可能并不是蒸馏有效，而是因为“多训练了一次”或者“第二次初始化更好运气”。

`ce_retrain_same_cost` 就是等训练量对照：

```text
CE teacher -> 重新初始化 student -> 仍然只用 CE，不用 KD
```

判断时应该比较：

```text
ce_baseline
ce_retrain_same_cost
self_distill_t2_alpha0.7
```

如果 `self_distill_t2_alpha0.7` 只比 `ce_baseline` 好，但不比 `ce_retrain_same_cost` 好，就不能说是蒸馏带来的提升。

## 当前卡在哪

现在不再卡在代码实现上。脚本已经支持用户提出的方案。

真正需要做的是正式跑实验。之前有一次正式跑失败，原因不是 loss 逻辑错误，而是 Windows + PyTorch DataLoader 多 worker 触发：

```text
RuntimeError: Couldn't open shared file mapping: <torch_...>, error code: <1455>
```

已经通过两层方式规避：

1. 脚本默认 `num_workers = 0`。
2. README 中所有正式命令仍显式写 `--num-workers 0`。

## 下一步计划

第一步，先做 dry-run：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --dry-run --device cpu --num-workers 0
```

第二步，正式跑三模型三 seed 的重复实验：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --num-workers 0 --models mambaout_tiny resnet50 convnext_tiny --experiments ce_baseline ce_retrain_same_cost self_distill_t2_alpha0.7 --seeds 2026 2027 2028
```

第三步，跑完后看：

```text
temp/soft_supervision_diagnostics/results/batch_YYYYMMDD_HHMMSS/aggregate_summary.csv
temp/soft_supervision_diagnostics/results/batch_YYYYMMDD_HHMMSS/aggregate_summary.html
```

判断逻辑：

- 如果自蒸馏平均提升约 1.5 个百分点，但 seed 标准差约 2 个百分点，说明正结果不可靠。
- 如果 3 个 seed 都稳定高于 `ce_baseline` 和 `ce_retrain_same_cost`，才说明自蒸馏确实有效。

第四步，如果重复结果稳定，再做固定 `T=2` 的 alpha 单变量：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --num-workers 0 --models mambaout_tiny resnet50 convnext_tiny --experiments ce_baseline ce_retrain_same_cost self_distill_t2_alpha0.5 self_distill_t2_alpha0.7 self_distill_t2_alpha0.9 --seeds 2026 2027 2028
```

## 绝对不要再踩的坑

1. 不要在 Windows 上用多 worker 正式跑。

   之前已经触发过：

   ```text
   Couldn't open shared file mapping, error code 1455
   ```

   虽然脚本默认已经是 0，但命令里仍建议显式写：

   ```text
   --num-workers 0
   ```

2. 不要把旧失败 batch 当成有效结果。

   旧失败目录类似：

   ```text
   temp/soft_supervision_diagnostics/results/batch_20260713_231638/
   ```

   它没有完整 `summary.csv`，只是失败排查材料。

3. 不要默认保存 `.pth`。

   用户明确不想保留权重。除非用户主动要求，否则不要加：

   ```text
   --keep-pth
   ```

4. 不要把 `self_distill` 写成从 teacher 权重继续训练。

   正确逻辑是：

   ```text
   teacher 固定 eval + no_grad
   student 重新初始化
   student 学 CE + KD
   ```

5. 不要第一轮就二维搜索 `T` 和 `alpha`。

   当前策略是：

   ```text
   先重复 T=2, alpha=0.7
   稳定后固定 T=2，只改 alpha
   ```

6. 不要混淆两个实验目录。

   ```text
   temp/soft_supervision_diagnostics/
   ```

   是四分类软监督。

   ```text
   temp/moderate_over_boundary_diagnostics/
   ```

   是按时间点做 moderate/over 边界诊断。

7. 不要提交 `results/` 或 `logs/`。

   它们已经在 `.gitignore`，如果 git status 里出现大量结果文件，先查 `.gitignore`，不要直接 `git add -A`。

## 文件职责

`run_soft_supervision_experiments.py`

主实验脚本。负责构建数据集、模型、loss、训练、测试、保存指标和报告。

`README.md`

给用户看的简短使用说明和推荐命令。

`HANDOFF.md`

给新会话看的完整交接。

`results/`

正式实验输出目录，git 忽略。

`logs/`

后台训练日志目录，git 忽略。
