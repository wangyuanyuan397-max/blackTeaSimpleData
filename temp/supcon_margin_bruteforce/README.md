# SupCon + Margin Brute-Force Sweep

这批实验把监督对比学习 SupCon 和余弦 margin 分类头接到 MixNet-S 上。

默认数据和输出：

- backbone: `mixnet_s`
- dataset: `datasets_01234_BaSic`
- common config: `temp/supcon_margin_bruteforce/common_01234_basic_408_train.yaml`
- output root: `temp/supcon_margin_bruteforce/runs_BaSic`

完整搜索网格：

- `margin`: 0, 0.05, 0.1, 0.15, 0.2
- `scale`: 16, 30, 64
- `temperature`: 0.05, 0.1, 0.2
- `lambda_supcon`: 0, 0.25, 0.5, 1.0
- `lr`: 1e-4, 3e-4, 1e-3
- `projector_out`: 64, 128, 256
- `classifier_feature`: `projected`, `raw`

`lambda_supcon=0` 时温度不参与损失，生成器只保留 `temperature=0.1`，避免重复实验。

总量：1 个 baseline + 2700 个 SupCon/Margin 配置。

生成配置：

```powershell
python temp\supcon_margin_bruteforce\generate_supcon_margin_configs.py --phase all
```

先跑 smoke：

```powershell
conda run --no-capture-output -n yolov8 python temp\supcon_margin_bruteforce\train_supcon_margin_bruteforce.py --phase smoke --generate --device auto --dry-run
conda run --no-capture-output -n yolov8 python temp\supcon_margin_bruteforce\train_supcon_margin_bruteforce.py --phase smoke --device auto
```

跑完整暴力队列：

```powershell
conda run --no-capture-output -n yolov8 python temp\supcon_margin_bruteforce\train_supcon_margin_bruteforce.py --phase all --generate --device auto
```

汇总结果：

```powershell
python temp\supcon_margin_bruteforce\summarize_supcon_margin_results.py
```

注意：

- 队列脚本会强制追加 `--discard-pth`，这批实验不保留 `.pth` 权重文件。
- baseline 使用普通单视图训练增强。
- SupCon/Margin 配置会把训练增强覆盖为 `two_view_patch_train_224`，训练输入为 `[B,2,C,H,W]`，验证/测试仍为普通单图输入。
