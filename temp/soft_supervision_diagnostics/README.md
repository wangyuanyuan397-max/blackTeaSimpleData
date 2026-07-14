# Soft supervision diagnostics

这个临时实验目录用于在 `datasets_split_patches` 四分类任务上比较软监督策略，重点看 `moderate` 和 `over` 的边界是否更稳定。

默认实验包括：

- `ce_baseline`
- `ce_retrain_same_cost`
- `label_smoothing_eps0.1`
- `bootstrap_soft_beta0.8_warm30`
- `self_distill_t2_alpha0.7`

额外可选实验：

- `self_distill_t2_alpha0.5`
- `self_distill_t2_alpha0.9`

结果保存到：

```text
temp/soft_supervision_diagnostics/results/
```

默认不保存 `.pth`；只有显式加 `--keep-pth` 才会保存权重。

## 先做 dry-run

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --dry-run --device cpu --num-workers 0
```

## 推荐第一轮：重复当前自蒸馏结果

先只跑三个模型、三个 seed，并加入等训练量 CE 对照：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --num-workers 0 --models mambaout_tiny resnet50 convnext_tiny --experiments ce_baseline ce_retrain_same_cost self_distill_t2_alpha0.7 --seeds 2026 2027 2028
```

跑完后看：

```text
summary.csv
summary.html
aggregate_summary.csv
aggregate_summary.html
```

其中 `aggregate_summary.*` 是按 `model + experiment` 汇总的均值和标准差。

判断逻辑：

- 如果自蒸馏平均提升约 1.5 个百分点，但 seed 标准差达到 2 个百分点左右，当前正结果不可靠。
- 如果 3 个 seed 都稳定高于 `ce_baseline` 和 `ce_retrain_same_cost`，才更能说明自蒸馏确实有效。
- `ce_retrain_same_cost` 用于排除“只是多训练一次 / 随机种子更好”造成的假提升。

## 第二轮：只做 alpha 单变量

确认重复实验稳定后，再固定 `T=2`，只改 `alpha`：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --num-workers 0 --models mambaout_tiny resnet50 convnext_tiny --experiments ce_baseline ce_retrain_same_cost self_distill_t2_alpha0.5 self_distill_t2_alpha0.7 self_distill_t2_alpha0.9 --seeds 2026 2027 2028
```

先不要二维搜索 `T` 和 `alpha`，否则很容易把随机波动调成“最优结果”。

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

只跑一个实验：

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --num-workers 0 --experiments self_distill_t2_alpha0.7
```

Windows 上建议一直显式写 `--num-workers 0`，避免 PyTorch DataLoader 的共享内存映射错误。
