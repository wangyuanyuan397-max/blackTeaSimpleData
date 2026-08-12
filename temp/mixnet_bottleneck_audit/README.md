# MixNet-S 76% 瓶颈核查

这是一次独立、可证伪的诊断尝试。所有新代码和默认输出都在：

```text
temp/mixnet_bottleneck_audit/
```

不修改主训练框架、正式配置或已有实验结果。目标不是继续给 MixNet-S 加模块，而是判断 76% 到底卡在数据、监督、泛化还是颜色鲁棒性。

## 已固定的实验事实

本审计只使用 `datasets_01234_BaSic` 及其 `grid30_crop_manifest.csv`：

- 5 个阶段：`00/10/20/30/40`。
- 120 张独立原图，每阶段 24 张。
- 固定 split：每阶段 train/val/test 原图数为 `15/4/5`。
- 每张原图严格对应 30 个 patch，总计 3600 patch。
- manifest 中的 `source_image_id` 是唯一 parent 真值；任何 parent 跨 split 都会直接报错。
- MixNet-S、408 输入、AdamW `1e-4`、weight decay `5e-4`、cosine、150 epochs、patience 30。

每个实验同时报告 patch 与 parent 指标。parent 默认使用 30 patch 平均概率，也额外给 majority vote，并以独立原图数计算 Wilson 95% 区间。请始终同时报告 test parent 数量（五分类只有 25 张，单个二分类只有 10 张），不要单独宣传一次 parent accuracy 的百分点变化。

## 五类核查

| 组 | 问题 | 实验 |
|---|---|---|
| B | 相邻阶段到底分不分得开？ | `00/10`、`10/20`、`20/30`、`30/40`，以及远距 `00/40`、`10/40` |
| D | 同一原图内部变化是否逼近相邻类间变化？ | M01 倒数第二层特征；parent 内 cosine spread / 类中心 cosine distance |
| F | 全量微调是否只换来少量收益却产生记忆？ | head-only、last-stage、full fine-tune |
| L | 放松 hard CE 后泛化是否立即改善？ | CE、uniform LS、adjacent soft、CE+ordinal auxiliary |
| C | 模型依赖稳定发酵颜色还是脆弱颜色/光照线索？ | 同一个 M01 checkpoint 上 RGB、灰度、确定性色彩压力、CLAHE |

`M01_five_class_ce` 同时是 full fine-tune、CE 和 RGB 的公共基线，避免重复训练三次。

## 建议运行顺序

先做环境、数据和单 batch 冒烟测试，不下载权重：

```powershell
D:\Softs\Anaconda\Anaconda3_202410\envs\yolov8\python.exe temp\mixnet_bottleneck_audit\run_audit.py --dry-run --jobs M01_five_class_ce --epochs 1 --batch-size 4 --eval-batch-size 8 --image-size 96 --device cuda --run-name smoke
```

第一刀只跑相邻与远距二分类，推荐 3 seeds：

```powershell
D:\Softs\Anaconda\Anaconda3_202410\envs\yolov8\python.exe temp\mixnet_bottleneck_audit\run_audit.py --groups binary --seeds 2026 2027 2028 --device cuda --run-name binary_3seed
```

完整核心审计（12 个 job × 3 seeds，计算量较大）：

```powershell
D:\Softs\Anaconda\Anaconda3_202410\envs\yolov8\python.exe temp\mixnet_bottleneck_audit\run_audit.py --groups core --seeds 2026 2027 2028 --device cuda --evaluate-colors --keep-pth --run-name core_3seed
```

如先跑单 seed，可用：

```powershell
D:\Softs\Anaconda\Anaconda3_202410\envs\yolov8\python.exe temp\mixnet_bottleneck_audit\run_audit.py --groups core --seeds 2026 --device cuda --evaluate-colors --keep-pth --run-name core_seed2026
```

特征空间核查必须使用五分类 M01 的 checkpoint：

```powershell
D:\Softs\Anaconda\Anaconda3_202410\envs\yolov8\python.exe temp\mixnet_bottleneck_audit\analyze_features.py --checkpoint temp\mixnet_bottleneck_audit\results\core_seed2026\M01_five_class_ce\seed_2026\best_model.pth --output-dir temp\mixnet_bottleneck_audit\results\core_seed2026\feature_seed2026 --device cuda
```

最后汇总：

```powershell
D:\Softs\Anaconda\Anaconda3_202410\envs\yolov8\python.exe temp\mixnet_bottleneck_audit\summarize_audit.py temp\mixnet_bottleneck_audit\results\core_seed2026
```

Windows 下默认 `num_workers=0`，用于规避共享内存映射问题。正式服务器环境稳定时可显式提高。

## 关键输出

```text
results/<run>/manifest_audit.json
results/<run>/run_plan.json
results/<run>/summary.csv
results/<run>/<job>/seed_<seed>/metrics.json
results/<run>/<job>/seed_<seed>/history.csv
results/<run>/<job>/seed_<seed>/test_patch_predictions.csv
results/<run>/<job>/seed_<seed>/test_parent_predictions.csv
results/<run>/feature_seed*/within_vs_between.csv
results/<run>/aggregate_summary.csv
results/<run>/evidence_matrix.csv
```

## 怎么判读

### B：相邻二分类

- 远距接近满分、相邻显著低：支持相邻阶段视觉重叠，但单凭它还不能区分“parent 内标签错配”和“总体类分布重叠”。
- 相邻二分类很高、五分类仍混淆：信息存在，更应查多类 decision/loss。

### D：parent 内外特征尺度

`within_to_between_ratio` 是主要量：

- `>= 1`：同图内部平均 spread 已超过对应类中心间距，是很强的局部/全局粒度错配信号。
- `0.5–1`：parent 内异质性已不可忽略。
- `< 0.5`：类间变化仍占主导，不能用该证据支持强错配结论。

它依赖 checkpoint，且 test 只有 25 个 parent，所以要看 parent bootstrap CI，并按 seed 重复。

### F：冻结深度

- head-only/last-stage 接近 full，而 full 的 train-test gap 很大：支持预训练表示足够、全量微调主要增加记忆。
- frozen 显著落后、last-stage/full 逐步提升：说明 task-specific adaptation 确实必要。

### L：监督

不要只看 accuracy；同时看 NLL、ECE、QWK、far errors、`max_abs_logit` 与 top1-top2 logit margin。若温和监督在多 seed 下同时改善泛化与校准，并抑制后期 logit 极端化，才支持 hard CE mismatch。

### C：颜色

- 灰度下降只说明颜色有用；发酵本身有颜色变化，这不等于 shortcut。
- 在保持语义的中等 brightness/contrast/saturation/hue 压力下大幅下降，才是颜色/光照脆弱性信号。
- CLAHE 是分布变换对照，不应仅凭单项下降断言预处理更差。

## 最终决策

`summarize_audit.py` 会产生保守的 `evidence_matrix.csv`，但阈值只负责筛查，不代替人工核查。正式结论至少要求：

1. parent-safe 固定 split；
2. 3 seeds 的方向一致；
3. 同时报告 patch 与 parent 指标；
4. 报告独立 parent 数量；
5. 不把颜色依赖、相关性或一次投票提升误写成因果证明。
