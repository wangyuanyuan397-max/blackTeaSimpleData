# MixNet-S SAA Three-Seed Reproduction

This queue reproduces the official baseline and the three candidates retained
from the 25-run SAA screen with seeds `42`, `2026`, and `3407`:

- `baseline`: official timm MixNet-S.
- `down_e1_g4`: accuracy/efficiency candidate (`saa06`).
- `down_e2_g1`: balanced Accuracy, Macro-F1, MAE, and QWK candidate (`saa07`).
- `mid_e1_g1`: mid-stage/generalization candidate (`saa09`).

All non-seed settings remain fixed by
`configs/fixed_split_01234_BaSic_grid30_408_train.yaml`. The queue is ordered
by seed so an interrupted batch still leaves complete within-seed comparisons.

This queue is retained as a completed experiment record. The current
`tools/train_batch_01234.py` entry point targets `mixnet_glrf/`, so the commands
below describe the original runner state and are not the active queue now:

```powershell
python tools\train_batch_01234.py --list-models
python tools\train_batch_01234.py --dry-run --device cpu
python tools\train_batch_01234.py
```

Each successful run saves `test_predictions.csv`, keyed by absolute image path,
with labels, predictions, logits, and probabilities. Batch completion writes:

- `mixnet_saa_fusion_summary.csv`: all 12 individual runs.
- `mixnet_saa_seed_reproduction_summary.csv`: per-family mean and sample SD.

Use the prediction files for paired McNemar or paired bootstrap comparisons;
do not compare rows by file order—join them by `image_path`.

After all 12 runs finish, generate the paired statistics automatically:

```powershell
python tools\analyze_saa_seed_predictions.py
```

This writes `mixnet_saa_paired_statistics.csv` under the runs root. Accuracy
and MAE receive paired bootstrap 95% intervals; Accuracy also receives the
exact two-sided McNemar p value. Macro-F1 and QWK point deltas are included.
