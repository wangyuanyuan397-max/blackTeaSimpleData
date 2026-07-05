# Moderate ↔ Over 临时边界诊断

该文件夹是完全独立的临时实验入口，不修改 `tools/train_batch.py`、正式 YAML、模型注册表或
原有 `runs`。右键运行 `run_experiments.py` 后，结果只写入本目录的 `results/`。

代码内固定五组单变量实验：

1. `stage5_ce`：四分类 Stage5 CE 对照。
2. `binary_moderate_over_stage5_ce`：每个固定 split 只保留 Moderate/Over，并映射为 0/1。
3. `stage5_ce_mo_weight1.5`：四分类不变，类别权重为 `[1, 1, 1.5, 1.5]`。
4. `stage5_ce_mo_logit_margin_m0.3_lam0.1`：只对真实标签 2/3 增加 logit margin。
5. `logistic_normal_cdf_t23_search`：训练一个 Logistic-Normal 模型，用验证集从
   `{0.65, 0.70, 0.75, 0.80, 0.85}` 选择 t23，随后只测试一次。

公共训练设置与正式实验一致：固定数据集、seed 2026、150 epoch、AdamW、学习率 `1e-4`、
weight decay `5e-4`、2 epoch warmup、cosine scheduler、patience 30。默认不保存 PTH。

## 运行

在 PyCharm 中直接右键运行：

```text
temp/moderate_over_boundary_diagnostics/run_experiments.py
```

先检查而不训练：

```powershell
python temp/moderate_over_boundary_diagnostics/run_experiments.py --dry-run --device cpu
```

只跑指定实验：

```powershell
python temp/moderate_over_boundary_diagnostics/run_experiments.py --experiments stage5_ce_mo_weight1.5
```

确实需要权重时添加 `--keep-pth`；默认最佳权重只保存在内存，测试完成后释放。
