# MixNet-S Partial High/Low Fourier Sweep

This folder contains the 01234 BaSiC/grid30/408 experiment queue for a
lightweight Fourier variant of the official timm `mixnet_s` backbone.

The backbone keeps the official MixNet-S blocks, kernels, SE modules, residual
paths, and channel counts unchanged. Each enabled block inserts a partial-channel
high/low Fourier filter after the depthwise mixed convolution activation and
immediately before the original SE module:

```text
conv_dw -> bn2 + act -> Partial High/Low Fourier -> SE -> conv_pwl
```

Common settings:

- `frequency_ratio: 0.25`: only the last 25% channels enter the Fourier branch.
- `low_frequency_radius_ratio: 0.35`: radial FFT cutoff for low-frequency bins.
- `residual_scale_init: 0.0`: starts as the pretrained official MixNet-S path.
- `random_seed: 2026`: matches the 01234 BaSiC batch baseline.

Queue:

- `fourier_d0_baseline_mixnet_s`: official MixNet-S control, no Fourier block.
- `fourier_d1_s2b3`: `blocks[2][3]`.
- `fourier_d2_s3b2`: `blocks[3][2]`.
- `fourier_d3_s4b2`: `blocks[4][2]`.
- `fourier_d4_s5b2`: `blocks[5][2]`.
- `fourier_d5_s2b3_s3b2`: `blocks[2][3] + blocks[3][2]`.
- `fourier_d6_s2b3_s4b2`: recommended main variant.
- `fourier_d7_s3b2_s4b2`: two 14x14-stage positions.
- `fourier_d8_s2b3_s3b2_s4b2`: three-position variant.
- `fourier_d9_s4_all`: all three blocks in `blocks[4]`.

Run via:

```powershell
python tools\train_batch_01234.py
```
