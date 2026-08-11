# MixNet-S final diagnostic RePr queue

This is the last diagnostic RePr-PWProj experiment. The old R01-R18 YAML files
remain in this directory as completed records, but `CONFIG_NAMES.txt` now runs
only R00 and R19-R23.

The target definition is unchanged: `mid` is residual stage S4,
`all_residual` is every residual inverted block, and only MixedConv PW
projection filters are ranked. Ranking remains global, branch-aware
inter-filter redundancy; every cycle recomputes the ranking. Restoration still
uses orthogonal reinitialization at scale 0.1, resets selected projection BN
state and clears corresponding optimizer state.

## Queue

| ID | Scope | Prune | Full S1 | Sparse S2 | Cycles |
|---|---|---:|---:|---:|---:|
| R00 | baseline | - | - | - | - |
| R19 | mid | 10% | 5 | 3 | 3 |
| R20 | mid | 20% | 5 | 3 | 3 |
| R21 | all_residual | 10% | 5 | 3 | 3 |
| R22 | all_residual | 20% | 5 | 3 | 3 |
| R23 | mid | 20% | 7 | 3 | 3 |

For R19-R22 the transition is full E1-E5, sparse E6-E8, full E9-E13,
sparse E14-E16, full E17-E21, sparse E22-E24, then ordinary full training
from E25 to the original early stop or epoch limit. R23 restores after E10,
E20 and E30; its first post-RePr-eligible checkpoint is E11.

The transition order is always train, validate, checkpoint, then prune or
restore. Sparse-end validation therefore measures the masked network before
reinitialization; newly restored filters must train for one full epoch before
they can enter post-RePr selection.

## Checkpoints and outputs

Each RePr run keeps two checkpoints:

- `best_global.pth`: best validation checkpoint from any phase, beginning at E1.
- `best_post_repr.pth`: best full-network checkpoint after at least one complete
  prune/sparse/reinitialize/restore cycle and one subsequent full epoch.

Both receive the same fixed test evaluation. `test_metrics.json` stores nested
global and post-RePr Accuracy, Macro-F1, MAE, QWK, +/-1 accuracy, error counts,
class accuracy and confusion matrices. Separate global/post confusion-matrix
CSVs are also written. The primary batch result is the post-RePr test result.

Per-epoch history records phase, cycle, completed cycles, train/validation
metrics, train-val gap, pruned count, redundancy mean/median/P90/max and both
best-refresh flags. Cycle history additionally records before-prune, sparse-end,
first-full-after-restore and best-full-after-restore validation snapshots plus
selection overlap between consecutive rankings.

Run through the shared 01234 entry point:

```powershell
python tools\train_batch_01234.py --repr-diagnostic --list-models
python tools\train_batch_01234.py --repr-diagnostic --dry-run --device cpu
python tools\train_batch_01234.py --repr-diagnostic
```

All common data, pretrained-weight, optimizer, learning-rate, scheduler,
augmentation, CE, split, seed, early-stopping and maximum-epoch settings remain
inherited from `fixed_split_01234_BaSic_grid30_408_train.yaml`.
