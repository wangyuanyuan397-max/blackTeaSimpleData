# ResNet50 Knowledge Evolution（独立临时实验）

这个目录只验证 Knowledge Evolution 的核心机制，不修改 `src/`，也不实现论文后半段的瘦身子网络：

```text
标准 ResNet50
  -> 每代正常训练 30 epoch
  -> 固定 KELS mask 的 FIT 权重原样继承
  -> 同一 mask 的 RESET 权重重新随机初始化
  -> BN 参数/统计量与所有 bias 原样继承
  -> 重建 optimizer 和 scheduler
  -> 下一代继续训练全部参数
```

默认设置是：

- 数据：`datasets_01234_BaSic`
- 模型：ImageNet 预训练 ResNet50（与现有 baseline 条件一致）
- `sr=0.8`
- 3 generations
- 30 epochs/generation
- AdamW、学习率、weight decay、cosine scheduler 和数据增强沿用当前 BaSiC 408 baseline
- 每一代都跑满相同 epoch 数，不使用 early stopping

注意：`sr=0.8` 不等于只重置 20% 卷积权重。普通中间卷积的 FIT 区约为 `0.8 × 0.8 = 64%`，RESET 区约为 36%。程序会把真实的逐层比例写入 `fixed_kels_mask_summary.json`。

## 先检查

不下载预训练权重、不创建运行目录：

```powershell
conda run --no-capture-output -n yolov8 python temp\knowledge_evolution_resnet50\run_knowledge_evolution.py --dry-run
```

只检查 KELS reset 本身（不读数据）：

```powershell
conda run --no-capture-output -n yolov8 python temp\knowledge_evolution_resnet50\smoke_test_ke_core.py
```

## 默认正式实验

```powershell
conda run --no-capture-output -n yolov8 python temp\knowledge_evolution_resnet50\run_knowledge_evolution.py
```

同时补做普通 ResNet50 连续训练 90 epochs 的关键对照：

```powershell
conda run --no-capture-output -n yolov8 python temp\knowledge_evolution_resnet50\run_knowledge_evolution.py --run-continuous-control
```

可覆盖主要参数，例如：

```powershell
conda run --no-capture-output -n yolov8 python temp\knowledge_evolution_resnet50\run_knowledge_evolution.py --split-rate 0.9 --generations 5 --epochs-per-generation 30
```

默认不保留 `.pth`，测试和结果归档完成后删除，以免多代 ResNet50 checkpoint 占满磁盘。需要保留时加 `--keep-pth`。

## 结果目录

所有产物都留在：

```text
temp/knowledge_evolution_resnet50/runs_BaSic/
```

一次运行会包含：

- `run_manifest.json`：实验口径与种子规则
- `fixed_kels_mask_summary.json`：固定 mask 的逐层 FIT/RESET 审计
- `generation_XX/`：每代曲线、历史、最佳验证 checkpoint 的测试指标、预测与混淆矩阵
- `generation_XX/generation_reset_report.json`：FIT 未变化、RESET 已变化、BN 未变化的换代审计
- `generation_summary.csv`、`SUMMARY.md`、`generation_trends.png`：跨代汇总
- `continuous_control_090_epochs/`：仅在指定 `--run-continuous-control` 时生成

代际传递使用“上一代最后一个 epoch 的模型”，而每代汇报使用“该代最佳验证 checkpoint”。脚本在测试前先保存最终状态，避免最佳 checkpoint 反过来改变进化链。
