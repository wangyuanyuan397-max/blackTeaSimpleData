# MixNet-S Regularization / MixUp / CutMix Ablation

This folder contains the runnable 01234 ablation queue for the local timm
environment.

The requested `mixnet_xs` and `mixnet_xxs` names are not available in the
currently installed timm package. Local timm lists `mixnet_s`, `mixnet_m`,
`mixnet_l`, `mixnet_xl`, `mixnet_xxl`, and the `tf_mixnet_*` variants. To avoid
creating configs that fail at model construction, this queue uses `mixnet_s` as
the available MixNet backbone.

Run with:

```powershell
conda run -n yolov8 python tools\train_batch_01234.py
```

Useful checks:

```powershell
conda run -n yolov8 python tools\train_batch_01234.py --list-models
conda run -n yolov8 python tools\train_batch_01234.py --dry-run
```

Experiment order:

1. Baseline: no extra regularization, no batch mixing.
2. Strong regularization: Dropout=0.5, label smoothing=0.1, weight decay=1e-2.
3. Regularization + MixUp: MixUp alpha=0.2.
4. Regularization + CutMix: CutMix alpha=0.2.
5. Regularization + mixed augmentation: randomly switches MixUp/CutMix with
   50% CutMix probability, both alpha=0.2.
