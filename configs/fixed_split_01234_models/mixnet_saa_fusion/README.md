# MixNet-S Scale-Aware Aggregation Fusion

This isolated 25-run queue keeps the official timm `mixnet_s` kernels and all
pretrained layers, then inserts SAA after depthwise MixConv `BN + SiLU` and
before the original SE module.

The four targets are:

- `all`: every official multi-scale depthwise MixConv block.
- `down`: multi-scale depthwise MixConv blocks with stride 2.
- `mid`: stage S4 (the 14x14 semantic stage at 224 input; 26x26 at 408).
- `late`: stage S5 (the 7x7 semantic stage at 224 input; 13x13 at 408).

The queue contains one official baseline, 16 residual-SAA combinations
(`4 targets x 2 expansions x 2 inter-group settings`), four pure replacement
runs, and four bounded modulation runs. `saa11_res_p3_mid_e2_g1` was the
pre-screening candidate recorded before these results were available.

SAA branch heads are always inferred from the unmodified MixConv kernel list.
For timm 1.0.25, official MixNet-S uses two through five heads depending on the
block; no three-head assumption is hard-coded.

The original 25 configurations and the follow-up three-seed queue remain here
as completed experiment records. `tools/train_batch_01234.py` now targets the
separate `mixnet_repr/` queue; these SAA files are no longer the active runner
queue.

All training, data, optimizer, scheduler, early-stopping, pretrained-weight,
and split settings come from
`configs/fixed_split_01234_BaSic_grid30_408_train.yaml`; only the model
structure changes between these YAML files.
