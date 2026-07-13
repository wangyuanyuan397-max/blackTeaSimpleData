# Soft supervision diagnostics

This temporary experiment runner compares four-class soft-supervision methods on
`datasets_split_patches`:

- `ce_baseline`
- `label_smoothing_eps0.1`
- `bootstrap_soft_beta0.8_warm30`
- `self_distill_t2_alpha0.7`

Default full run:

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto
```

Dry run:

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --dry-run --device cpu --num-workers 0
```

Run one model:

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --models mambaout_tiny
```

Run one experiment:

```powershell
conda run -n yolov8 python temp\soft_supervision_diagnostics\run_soft_supervision_experiments.py --device auto --experiments self_distill_t2_alpha0.7
```

Results are saved under:

```text
temp/soft_supervision_diagnostics/results/
```

Checkpoints are not kept unless `--keep-pth` is explicitly provided.
