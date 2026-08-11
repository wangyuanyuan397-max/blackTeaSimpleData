# MixNet-S RePr-PWProj

This isolated queue evaluates cyclic RePr on MixNet-S residual projection
filters. It does not replace MixConv, depthwise kernels, expansion layers, SE,
the classifier head, the common dataset config, or any existing backbone.

The queue contains one unchanged timm MixNet-S baseline and 18 controlled
variants:

- scope: `mid` (S4), `late` (S5), or `all_residual` (all residual blocks);
- global prune ratio: 10%, 20%, or 30%, with a 40% per-block cap;
- cycle schedule: 10 full + 5 sparse epochs, or 20 full + 10 sparse epochs;
- three prune/restore cycles and orthogonal reintroduction at scale 0.1.

Filters are ranked branch-wise inside each MixedConv projection and selected
globally. Selected channels are masked after projection BatchNorm, so tensor
shapes never change. On restoration, projection rows are orthogonally
reinitialized, selected BatchNorm state is reset, and matching optimizer state
is cleared.

Sparse validation epochs are logged but cannot replace `best_model.pth`.
Early stopping starts only after the third restore. This guarantees that the
selected checkpoint is a complete, unmasked network and that all three cycles
are actually executed.

This queue remains as a completed experiment record. The active
`tools/train_batch_01234.py` runner now targets the independent
`mixnet_orthoshot/phase1_dbt/` queue.

Its historical command was:

```powershell
python tools\train_batch_01234.py --list-models
python tools\train_batch_01234.py --dry-run --device cpu
python tools\train_batch_01234.py
```

Batch completion writes `mixnet_repr_summary.csv` under the common runs root.
Per-epoch controller state is also stored in `history.json`, and the final test
metrics include the target blocks, completed cycles, filter-redundancy values,
and reinitialization history.
