# MixNet-S Ortho-Shot adaptation

This directory contains a training-only Ortho-Shot adaptation for the fixed
five-class MixNet-S task. It keeps the official timm `mixnet_s` inference
graph, pretrained weights, dataset splits, cross-entropy objective and common
optimizer/scheduler settings unchanged. Few-shot support/query/task episodes
are intentionally not introduced.

## Baseline audit

The common `patch_train_224` transform is not augmentation-free: at 408 input
it performs resize, horizontal flip at 0.5, vertical flip at 0.5, tensor
conversion and ImageNet normalization. `orthoshot_patch_train` has exactly
those defaults, so its disabled form is equivalent to the old transform.

This matters for the combination ablation: a nominal `+ HFlip` job would
duplicate the baseline. Phase 3 therefore uses explicit no-HFlip, baseline
flip and rotation variants instead of submitting duplicate jobs. O03 also
serves as the projection-DBT/no-warmup case that was repeated as O11/O16 in
the draft matrix.

## Experiment phases

- `phase1_dbt`: 12 canonical jobs. This is the active batch queue. It scans
  projection lambda, target scope, depthwise DBT and five-epoch lambda warmup.
- `phase2_single_augmentation`: 16 jobs. Each new augmentation is isolated
  with DBT and MaxUp disabled.
- `phase3_combined_augmentation`: 7 non-duplicate CutMix/SelfMix/flip/rotation
  combinations.
- `phase4_maxup`: 5 jobs. MaxUp takes a per-sample maximum across candidates;
  the DBT term, if later combined, is evaluated once per optimizer step.

MaxUp configurations reduce the physical batch and increase gradient
accumulation so the base effective batch remains 32 and the concatenated
forward batch remains 32: `16 x 2 candidates` or `8 x 4 candidates`.

The final F00-F05 comparison is deliberately not hard-coded before screening:
it must be instantiated from the measured best DBT, augmentation and MaxUp
settings, otherwise the label “Best” would be misleading.

## Run

The active Phase-1 queue is:

```powershell
python tools\train_batch_01234.py --list-models
python tools\train_batch_01234.py --dry-run --device cpu
python tools\train_batch_01234.py
```

Run later phases through the same entry point, for example:

```powershell
python tools\train_batch_01234.py --orthoshot-phase phase2_single_augmentation --dry-run --device cpu
python tools\train_batch_01234.py --orthoshot-phase phase3_combined_augmentation
python tools\train_batch_01234.py --orthoshot-phase phase4_maxup
```

Keeping the phases sequential prevents selecting a final combination from
unmeasured assumptions.

Batch completion writes `mixnet_orthoshot_summary.csv`. Besides accuracy,
macro metrics and model complexity, it records CE, raw/weighted orthogonal
loss, orthogonal-to-CE ratio, selected target counts, expansion/projection
filter cosine, augmentation frequency or MaxUp winner frequency, physical
batch, effective base batch and forward batch.
