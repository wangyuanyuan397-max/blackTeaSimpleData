# Moderate ↔ Over 多模型临时边界诊断

该文件夹是完全独立的临时实验入口，不修改 `tools/train_batch.py`、正式 YAML、模型注册表或
原有 `runs`。右键运行 `run_experiments.py` 后，结果只写入本目录的 `results/`。

## 模型列表

`MODEL_CONFIG_LIST` 默认包含四个模型 YAML：

1. `mambaout_tiny`
2. `resnet50`
3. `convnext_tiny`
4. `safnet_imagenet`

每个模型会分别运行四组兼容的边界实验：

1. `four_class_ce`：四分类 CE 对照。
2. `binary_moderate_over_ce`：每个固定 split 只保留 Moderate/Over，并映射为 0/1。
3. `four_class_ce_mo_weight1.5`：四分类不变，类别权重为 `[1, 1, 1.5, 1.5]`。
4. `four_class_ce_mo_logit_margin_m0.3_lam0.1`：只对真实标签 2/3 增加 logit margin。

因此默认生成 `4 个模型 × 4 个实验 = 16` 组模型 YAML 结果。原有
`logistic_normal_cdf_t23_search` 依赖专用 EfficientNetV2-S 概率输出，不能公平套用普通分类
YAML，所以仍作为独立第 17 组运行；它用验证集从 `{0.65, 0.70, 0.75, 0.80, 0.85}` 选择
t23，随后只测试一次。

公共训练设置为固定数据集、seed 2026、150 epoch、AdamW、学习率 `1e-4`、weight decay
`5e-4`、2 epoch warmup、cosine scheduler、patience 30。默认不保存 PTH。

每组产物独立保存在：

```text
results/batch_<时间>/<模型名>/<实验名>/
```

批次根目录另外生成 `summary.csv` 和 `summary.html`。

## 运行

在 PyCharm 中直接右键运行：

```text
temp/moderate_over_boundary_diagnostics/run_experiments.py
```

先检查所有模型而不下载预训练权重、不训练：

```powershell
python temp/moderate_over_boundary_diagnostics/run_experiments.py --dry-run --device cpu
```

只跑指定模型和实验：

```powershell
python temp/moderate_over_boundary_diagnostics/run_experiments.py --models resnet50 convnext_tiny --experiments four_class_ce
```

列出可选项：

```powershell
python temp/moderate_over_boundary_diagnostics/run_experiments.py --list-models --list-experiments
```

首次正式运行可能需要下载各 YAML 指定的预训练权重。确实需要保留最佳权重时添加
`--keep-pth`；否则最佳权重只在内存中用于最终测试，指标、训练历史、混淆矩阵和 HTML 报告
仍会分别保留。
