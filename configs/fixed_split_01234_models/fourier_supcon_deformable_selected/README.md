# Fourier x SupCon/Margin and Deformable selected sweep

This folder contains 21 BaSiC/grid30/408 experiments for `tools/train_batch_01234.py`.

Common training settings inherit from:

```text
configs/fixed_split_01234_BaSic_grid30_408_train.yaml
```

Fourier sources:

- `fourier_d1_s2b3`
- `fourier_d9_s4_all`
- `fourier_d2_s3b2`

Targets:

- 3 SupCon/Margin recipes from `configs/fixed_split_01234_models/supcon_margin_selected`.
- 4 deformable-attention MixNet-S variants: `D11000_seed2026`, `D10101_seed2026`, `D00000_seed42`, and `D11011_seed2026`.

Combination rules:

- Fourier x SupCon/Margin keeps the SupCon/Margin `data`, `loss`, `optimizer`, projector, and cosine-margin head, and swaps the backbone to `mixnet_s_fourier`.
- Fourier x Deformable keeps the deformable stage settings and linear classifier head, and uses `mixnet_s_fourier_deformable` so Fourier filters and deformable attention are both active.

Generated configs:

- `fourier_d1_s2b3__supm_mixnet_s_m0p1_s64_t0p1_ls0_lr0p001_p128_projected`
- `fourier_d1_s2b3__supm_mixnet_s_m0p05_s30_t0p1_ls1_lr0p0003_p128_projected`
- `fourier_d1_s2b3__supm_mixnet_s_m0p05_s30_t0p05_ls1_lr0p0003_p128_projected`
- `fourier_d1_s2b3__D11000_seed2026`
- `fourier_d1_s2b3__D10101_seed2026`
- `fourier_d1_s2b3__D00000_seed42`
- `fourier_d1_s2b3__D11011_seed2026`
- `fourier_d9_s4_all__supm_mixnet_s_m0p1_s64_t0p1_ls0_lr0p001_p128_projected`
- `fourier_d9_s4_all__supm_mixnet_s_m0p05_s30_t0p1_ls1_lr0p0003_p128_projected`
- `fourier_d9_s4_all__supm_mixnet_s_m0p05_s30_t0p05_ls1_lr0p0003_p128_projected`
- `fourier_d9_s4_all__D11000_seed2026`
- `fourier_d9_s4_all__D10101_seed2026`
- `fourier_d9_s4_all__D00000_seed42`
- `fourier_d9_s4_all__D11011_seed2026`
- `fourier_d2_s3b2__supm_mixnet_s_m0p1_s64_t0p1_ls0_lr0p001_p128_projected`
- `fourier_d2_s3b2__supm_mixnet_s_m0p05_s30_t0p1_ls1_lr0p0003_p128_projected`
- `fourier_d2_s3b2__supm_mixnet_s_m0p05_s30_t0p05_ls1_lr0p0003_p128_projected`
- `fourier_d2_s3b2__D11000_seed2026`
- `fourier_d2_s3b2__D10101_seed2026`
- `fourier_d2_s3b2__D00000_seed42`
- `fourier_d2_s3b2__D11011_seed2026`
