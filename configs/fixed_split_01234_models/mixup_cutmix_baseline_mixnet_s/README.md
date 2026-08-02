# MixUp/CutMix baseline MixNet-S sweep

This folder tests batch-level MixUp and CutMix training strategies on the
baseline `timm` MixNet-S model for the 01234 BaSiC/grid30/408 dataset.

Common training settings inherit from:

```text
configs/fixed_split_01234_BaSic_grid30_408_train.yaml
```

Included configs:

- `mixnet_s_mixup_a0p5_seed2026`: MixUp only, `mixup_alpha=0.5`.
- `mixnet_s_cutmix_a1p0_seed2026`: CutMix only, `cutmix_alpha=1.0`.
- `mixnet_s_mixup_cutmix_m0p5_c1p0_seed2026`: randomly switches between MixUp and CutMix per batch.

These augmentations are applied only during training, after the DataLoader
returns a batch. Validation and testing still use the normal single-image eval
pipeline from the common config.
