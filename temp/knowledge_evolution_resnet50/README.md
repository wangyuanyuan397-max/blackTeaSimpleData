# ResNet50 Knowledge Evolution V2（最后验证矩阵）

这个目录是一次独立、可下结论的 KE-V2 验证，不修改 `src/`，也不实现瘦身子网络。

V2 修正了旧版最关键的问题：

```text
旧版：上一代 final epoch -> KELS reset -> 下一代
V2：  上一代磁盘 Best-Val -> KELS reset -> 下一代
```

每一代固定训练完整的 10 或 15 epochs，early stopping 关闭；optimizer 和 cosine scheduler 每代重新创建。固定 KELS mask 只在整个 run 开始时创建一次，BN 参数/统计量与全部 bias 继续继承。

## 最终五组矩阵

| ID | KELS reset | sr | Epochs/代 | Generations | 下一代来源 |
| --- | --- | ---: | ---: | ---: | --- |
| CTRL-10 | 否 | — | 10 | 3 | 前一代 Best-Val |
| KE-V2-01 | 是 | 0.8 | 10 | 3 | 前一代 Best-Val |
| KE-V2-02 | 是 | 0.9 | 10 | 3 | 前一代 Best-Val |
| CTRL-15 | 否 | — | 15 | 3 | 前一代 Best-Val |
| KE-V2-03 | 是 | 0.9 | 15 | 3 | 前一代 Best-Val |

CTRL 与对应 KE 唯一的关键差异是是否 reset KELS 区域，因此可以区分 KELS 的贡献与 Best-checkpoint carry-over、optimizer/scheduler restart 的贡献。

五个配置位于 [`experiments/`](experiments/)；公共数据和训练超参数位于 [`common_01234_basic_408_train.yaml`](common_01234_basic_408_train.yaml)。

## 实现口径

- 第 1 代从与 baseline 相同的 ImageNet 预训练 ResNet50 开始。
- 最佳 checkpoint 固定按以下顺序选择：
  1. `val_accuracy` 越高；
  2. 同准确率时 `val_qwk` 越高；
  3. 再相同时 `val_loss` 越低。
- G2/G3 从上一代磁盘 `best_val.pth` 重新加载，不使用内存 final model。
- KE 组构建同结构 `pretrained=false` 的完整 fresh model，从中复制 RESET 权重。
- FIT 权重、所有 bias、所有 BN 参数和 running statistics 保留。
- CTRL 组执行完全相同的 Best-Val 传代和 optimizer/scheduler restart，只是不 reset。
- 每代训练前先做 Start Validation，记录 reset shock。
- `recovery_epoch` 是首次严格超过母代 Best-Val 的训练 epoch；若起始点已经超过则记为 0，始终未超过则为 null。
- BN recalibration 已留开关，但五组配置全部为 `enabled: false`。

## 先验证

列出矩阵：

```powershell
conda run --no-capture-output -n yolov8 python temp\knowledge_evolution_resnet50\run_knowledge_evolution.py --list-experiments
```

检查五个 YAML、固定数据集、ResNet50 和 `sr=0.8/0.9` 的实际 mask 比例，不下载预训练权重、不创建 runs：

```powershell
conda run --no-capture-output -n yolov8 python temp\knowledge_evolution_resnet50\run_knowledge_evolution.py --dry-run --device cpu
```

检查 KELS、fresh-model reset、BN/bias 继承和 checkpoint tie-break：

```powershell
conda run --no-capture-output -n yolov8 python temp\knowledge_evolution_resnet50\smoke_test_ke_core.py
```

检查真实项目 ResNet50 的 Best-Val 落盘/回载、fresh-model reset 和 optimizer/scheduler 重建（不训练 epoch）：

```powershell
conda run --no-capture-output -n yolov8 python temp\knowledge_evolution_resnet50\smoke_test_ke_v2_orchestration.py
```

## 运行

一次运行全部五组：

```powershell
conda run --no-capture-output -n yolov8 python temp\knowledge_evolution_resnet50\run_knowledge_evolution.py --device auto
```

只跑指定组：

```powershell
conda run --no-capture-output -n yolov8 python temp\knowledge_evolution_resnet50\run_knowledge_evolution.py --experiments CTRL-10 KE-V2-01 KE-V2-02 --device auto
```

## 产物

所有结果保存在：

```text
temp/knowledge_evolution_resnet50/runs_BaSic/ke_v2_matrix_<timestamp>/
```

每组包含：

```text
generation_01/
  best_val.pth
  final.pth
  start_validation_metrics.json
  history.json
  test_metrics.json
  generation_summary.json

generation_02/
  best_val.pth
  final.pth
  generation_transition_report.json
  start_validation_metrics.json
  ...
```

`best_val.pth` 一定保留；`final.pth` 默认也保留用于审计，但绝不用于传代。五组全部运行会占用数 GB checkpoint 空间。

汇总产物包括：

- 每组 `generation_summary.csv` 和 `SUMMARY.md`；
- `generation_trends.png`；
- `reset_shock_recovery.png`；
- 全矩阵 `all_generation_summary.csv`；
- `ke_vs_control_comparison.csv`；
- `matrix_test_accuracy.png`；
- 自动按“准确率至少 +1pp、Macro-F1 上升、MAE/QWK 不恶化”给出配对裁决。
