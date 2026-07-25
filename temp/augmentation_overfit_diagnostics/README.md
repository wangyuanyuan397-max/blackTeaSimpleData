# MixNet-S Transform Sweep

Temporary online augmentation sweep for the 01234 fixed-grid 408 baseline.

The runner changes only `data.train_transform`. It keeps:

- model: `configs/fixed_split_01234_models/fixed_timm_mixnet_s.yaml`
- common training config: `configs/fixed_split_01234_grid30_408_train.yaml`
- validation/test transform: deterministic `patch_eval_224`, `image_size=408`
- dataset split, optimizer, scheduler, loss, epochs, and batch size unchanged

## Edit Transform Recipes

Edit only `TRANSFORM_EXPERIMENT_LIST` in:

```powershell
temp\augmentation_overfit_diagnostics\run_augmentation_experiments.py
```

Each item contains `name`, `description`, `train_transform`, and `random_seed`.

## Local Checks

Do not run full training locally. Use lightweight checks only:

```powershell
python -m py_compile temp\augmentation_overfit_diagnostics\run_augmentation_experiments.py
python temp\augmentation_overfit_diagnostics\run_augmentation_experiments.py --dry-run --device cpu --num-workers 0
```

Optional preview export:

```powershell
python temp\augmentation_overfit_diagnostics\run_augmentation_experiments.py --dry-run --device cpu --num-workers 0 --preview-transforms
```

Previews are written to:

```text
temp/augmentation_overfit_diagnostics/preview_transforms/
```

## Server Run

First screening run:

```powershell
python temp\augmentation_overfit_diagnostics\run_augmentation_experiments.py --device auto --num-workers 4 --seeds 2026 --discard-pth
```

Run selected transforms only:

```powershell
python temp\augmentation_overfit_diagnostics\run_augmentation_experiments.py --device auto --num-workers 4 --seeds 2026 --experiments baseline_current_flip geom_noise_medium geom_strong --discard-pth
```

Three-seed confirmation after screening:

```powershell
python temp\augmentation_overfit_diagnostics\run_augmentation_experiments.py --device auto --num-workers 4 --seeds 2026 3407 42 --experiments baseline_current_flip geom_noise_medium geom_strong --discard-pth
```

## Results

All outputs go to:

```text
temp/augmentation_overfit_diagnostics/results/
```

Main comparison files:

- `transform_sweep_<timestamp>_summary.csv`
- `transform_sweep_<timestamp>_summary.html`
- `batch_<timestamp>_summary.html`

Focus on test metrics and overfit indicators:

- `accuracy`
- `macro_f1`
- `qwk`
- `mae`
- `train_val_gap_at_best`
- `final_train_val_gap`
- `val_peak_to_final_drop`
