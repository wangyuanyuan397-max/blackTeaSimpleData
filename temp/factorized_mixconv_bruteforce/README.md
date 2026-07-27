# Factorized MixConv Brute Force

This experiment keeps the `fixed_timm_mixnet_s` baseline training settings fixed
and changes only selected MixNet-S depthwise branches:

```text
KxK depthwise conv -> 1xK depthwise conv -> Kx1 depthwise conv
```

There is no BatchNorm or activation between the two asymmetric convolutions.
The 3x3 branches are always kept unchanged. The F-code bit order is:

```text
FABC = [5x5, 7x7, 9x9]
0 = original KxK
1 = factorized 1xK -> Kx1
```

Configs:

```text
F000: []
F001: [9]
F010: [7]
F011: [7, 9]
F100: [5]
F101: [5, 9]
F110: [5, 7]
F111: [5, 7, 9]
```

Each config is run with seeds `42`, `3407`, and `2026`, for 24 total runs.

Generate YAML configs:

```powershell
python temp\factorized_mixconv_bruteforce\generate_factorized_mixconv_configs.py
```

Dry-run checks:

```powershell
conda run -n yolov8 python temp\factorized_mixconv_bruteforce\train_factorized_mixconv_bruteforce.py --dry-run --device cpu
```

Run all 24 experiments:

```powershell
conda run -n yolov8 python temp\factorized_mixconv_bruteforce\train_factorized_mixconv_bruteforce.py --device auto
```

Run a subset:

```powershell
conda run -n yolov8 python temp\factorized_mixconv_bruteforce\train_factorized_mixconv_bruteforce.py --models F001_seed42 F011_seed42 --device auto
```

Outputs stay under:

```text
temp/factorized_mixconv_bruteforce/runs_BaSic
```

After training, the runner writes:

```text
factorized_mixconv_all_results.csv
factorized_mixconv_summary_by_config.csv
```
