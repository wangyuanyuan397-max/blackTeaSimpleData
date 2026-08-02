# AdamNorm x selected 18-model sweep

This folder contains 36 BaSiC/grid30/408 experiments for `tools/train_batch_01234.py`.

Common training settings inherit from:

```text
configs/fixed_split_01234_BaSic_grid30_408_train.yaml
```

AdamNorm recipes:

- `adamnorm_lr0p0001` from `adamnorm_stagemask_001101_k357_lr0p0001`: `type=adamnorm`, `lr=0.0001`, `gamma=0.95`.
- `adamnorm_lr0p0003` from `adamnorm_stagemask_001101_k357_lr0p0003`: `type=adamnorm`, `lr=0.0003`, `gamma=0.95`.

Target model families:

- 3 Fourier high/low MixNet-S variants.
- 4 deformable-attention MixNet-S variants.
- 8 MixNet-S structure-search variants.
- 3 SupCon/Margin recipes.

For each target, the target YAML supplies model/data/loss metadata. The optimizer section is replaced by AdamNorm.

Generated configs:

- `adamnorm_lr0p0001__fourier_d2_s3b2`
- `adamnorm_lr0p0001__fourier_d9_s4_all`
- `adamnorm_lr0p0001__fourier_d1_s2b3`
- `adamnorm_lr0p0001__D11000_seed2026`
- `adamnorm_lr0p0001__D10101_seed2026`
- `adamnorm_lr0p0001__D00000_seed42`
- `adamnorm_lr0p0001__D11011_seed2026`
- `adamnorm_lr0p0001__stagemask_001101_k357`
- `adamnorm_lr0p0001__10_p10_only_s0_k357`
- `adamnorm_lr0p0001__stagemask_000101_k357`
- `adamnorm_lr0p0001__p03_stride2_k357_g3_softmax`
- `adamnorm_lr0p0001__stagemask_001001_k357`
- `adamnorm_lr0p0001__13_p13_only_s3_k357`
- `adamnorm_lr0p0001__stagemask_011010_k357`
- `adamnorm_lr0p0001__12_p12_only_s2_k357`
- `adamnorm_lr0p0001__supm_mixnet_s_m0p1_s64_t0p1_ls0_lr0p001_p128_projected`
- `adamnorm_lr0p0001__supm_mixnet_s_m0p05_s30_t0p1_ls1_lr0p0003_p128_projected`
- `adamnorm_lr0p0001__supm_mixnet_s_m0p05_s30_t0p05_ls1_lr0p0003_p128_projected`
- `adamnorm_lr0p0003__fourier_d2_s3b2`
- `adamnorm_lr0p0003__fourier_d9_s4_all`
- `adamnorm_lr0p0003__fourier_d1_s2b3`
- `adamnorm_lr0p0003__D11000_seed2026`
- `adamnorm_lr0p0003__D10101_seed2026`
- `adamnorm_lr0p0003__D00000_seed42`
- `adamnorm_lr0p0003__D11011_seed2026`
- `adamnorm_lr0p0003__stagemask_001101_k357`
- `adamnorm_lr0p0003__10_p10_only_s0_k357`
- `adamnorm_lr0p0003__stagemask_000101_k357`
- `adamnorm_lr0p0003__p03_stride2_k357_g3_softmax`
- `adamnorm_lr0p0003__stagemask_001001_k357`
- `adamnorm_lr0p0003__13_p13_only_s3_k357`
- `adamnorm_lr0p0003__stagemask_011010_k357`
- `adamnorm_lr0p0003__12_p12_only_s2_k357`
- `adamnorm_lr0p0003__supm_mixnet_s_m0p1_s64_t0p1_ls0_lr0p001_p128_projected`
- `adamnorm_lr0p0003__supm_mixnet_s_m0p05_s30_t0p1_ls1_lr0p0003_p128_projected`
- `adamnorm_lr0p0003__supm_mixnet_s_m0p05_s30_t0p05_ls1_lr0p0003_p128_projected`
