# Lite-Mono-Inspired MixNet-S Steps

This queue keeps official timm `mixnet_s` as the baseline and tests the
Lite-Mono-inspired ideas one group at a time on the 01234 BaSiC/grid30/408 split.

Design rules:

- Baseline is official `mixnet_s`.
- Stage-LGFI is inserted after a whole `model.blocks[i]` stage, not inside every
  `InvertedResidual`.
- LGFI uses residual scaling initialized to zero: `Y = X + gamma * LGFI(X)`.
- Hybrid Dilated MixConv keeps dense `3x3/5x5` branches by default and replaces
  only `7x7/9x9/11x11` with `3x3` dilated branches.
- CDC stage rewrites are separate experiments because they change official
  MixNet-S more aggressively.

Queue groups:

- `lm00`: official MixNet-S baseline.
- `lm01`-`lm07`: stage-level LGFI and SE/LGFI overlap checks.
- `lm08`-`lm12`: Hybrid Dilated MixConv and full CDC-style stage rewrites.
- `lm13`-`lm14`: low-priority pooled RGB and cross-stage residual checks.
- `lm15`-`lm18`: selected combinations.

Run via:

```powershell
python tools\train_batch_01234.py
```
