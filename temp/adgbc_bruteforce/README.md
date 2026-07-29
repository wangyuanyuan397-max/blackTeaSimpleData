# AD-GBC Brute-Force Sweep

This experiment treats AD-GBC as a feature cleaner after the timm backbone
feature map and before global average pooling.

Default backbone and data:

- backbone: `mixnet_s`
- dataset: `datasets_01234_BaSic`
- common config: `temp/adgbc_bruteforce/common_01234_basic_408_train.yaml`
- output root: `temp/adgbc_bruteforce/runs_BaSic`

Generate configs:

```powershell
python temp\adgbc_bruteforce\generate_adgbc_configs.py --phase all
```

Full grid:

- `K`: 8, 16, 32, 64
- `tau`: 1.0, 0.5, 0.1
- `lambda_w_div`: 0, 0.01, 0.05, 0.1
- `beta_scale_con`: 0, 0.05, 0.1
- training strategy: `finetune_all`, `warmup5_then_finetune`, `adgbc_head_only`, `head_only`

Total: 1 baseline + 576 AD-GBC configs.

Queue the smoke grid first:

```powershell
conda run -n yolov8 python temp\adgbc_bruteforce\train_adgbc_bruteforce.py --phase smoke --generate --device auto --dry-run
conda run -n yolov8 python temp\adgbc_bruteforce\train_adgbc_bruteforce.py --phase smoke --device auto
```

Queue the full brute-force run:

```powershell
conda run -n yolov8 python temp\adgbc_bruteforce\train_adgbc_bruteforce.py --phase all --generate --device auto
```

Summarize completed runs:

```powershell
python temp\adgbc_bruteforce\summarize_adgbc_results.py
```

The summarizer writes:

- `adgbc_all_results.csv`
- `adgbc_summary_by_k.csv`
- `adgbc_summary_by_tau.csv`
- `adgbc_summary_by_lambda_w.csv`
- `adgbc_summary_by_beta_scale.csv`
- `adgbc_summary_by_training_strategy.csv`

Training strategies:

- `finetune_all`: train backbone, AD-GBC, and head end to end.
- `warmup5_then_finetune`: freeze backbone for 5 epochs while AD-GBC + head train, then unfreeze all at the same LR.
- `adgbc_head_only`: freeze the timm base backbone; train AD-GBC + head.
- `head_only`: freeze timm base and AD-GBC; train only the final head as a negative control.
