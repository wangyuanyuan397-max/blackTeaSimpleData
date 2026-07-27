# MixNet-S Explicit Scale Interaction

This isolated experiment targets the specific concern that MixConv scale
branches have no explicit interaction before their outputs are merged.

It does not modify:

- `configs/fixed_split_01234_models/fixed_timm_mixnet_s.yaml`
- the shared baseline training hyperparameters
- any permanent registry file under `src/`

The local runner imports `mixnet_explicit_scale_interaction_backbone.py` at
runtime and registers `mixnet_s_explicit_scale_interaction` only for these runs.

## Implemented Operators

- `baseline`
  - Plain `timm` MixNet-S through the local wrapper.
- `scale_attention`
  - Adds per-sample scalar attention over scale branches before concat.
- `small_to_large_guidance`
  - Adds zero-initialized 1x1 residual projections from smaller-kernel branch
    outputs to larger-kernel branch outputs.
- `large_to_small_guidance`
  - Adds zero-initialized 1x1 residual projections from larger-kernel branch
    outputs back to smaller-kernel branch outputs.
- `weighted_sum`
  - Replaces pure concat with a weighted sum over full-channel branch
    projections.  Each projection starts as an identity slice, so the initial
    behavior is close to concat.
- `cross_residual_bidir`
  - Adds bidirectional cross-scale residual projections before concat.
- `full_concat_interaction`
  - Combines scale attention and bidirectional residual connections before
    concat.
- `full_weighted_interaction`
  - Combines scale attention, bidirectional residual connections, and weighted
    full-channel summation.

For residual/guidance operators, `edge_mode=adjacent` connects neighboring
scale branches.  `edge_mode=all` connects every legal source-target scale pair.

## Generate Configs

First-round operators:

```powershell
python temp\mixnet_explicit_scale_interaction\generate_explicit_scale_interaction_configs.py --phase operators
```

Operators plus adjacent/all edge-mode variants:

```powershell
python temp\mixnet_explicit_scale_interaction\generate_explicit_scale_interaction_configs.py --phase all
```

After choosing a strong operator, generate stage-level probes:

```powershell
python temp\mixnet_explicit_scale_interaction\generate_explicit_scale_interaction_configs.py --phase stage_single --interaction-type cross_residual_bidir
python temp\mixnet_explicit_scale_interaction\generate_explicit_scale_interaction_configs.py --phase stage_mask --interaction-type cross_residual_bidir
```

Generate block-level probes:

```powershell
python temp\mixnet_explicit_scale_interaction\generate_explicit_scale_interaction_configs.py --phase block_single --interaction-type cross_residual_bidir
```

Generate local subset enumeration:

```powershell
python temp\mixnet_explicit_scale_interaction\generate_explicit_scale_interaction_configs.py --phase block_subset --interaction-type cross_residual_bidir --subset-blocks S3B0 S3B1 S3B2 S4B0 --max-subset-size 4
```

## Run

Smoke check:

```powershell
conda run -n yolov8 python temp\mixnet_explicit_scale_interaction\train_explicit_scale_interaction.py --phase smoke --dry-run --device cpu
```

Run first-round operators:

```powershell
conda run -n yolov8 python temp\mixnet_explicit_scale_interaction\train_explicit_scale_interaction.py --phase operators --device auto
```

Run edge-mode variants:

```powershell
conda run -n yolov8 python temp\mixnet_explicit_scale_interaction\train_explicit_scale_interaction.py --phase edge_modes --device auto
```

Results go to:

```text
temp/mixnet_explicit_scale_interaction/runs_BaSic
```

Each successful run writes `mixnet_explicit_scale_interaction` metadata into
`test_metrics.json`.  The batch runner also writes:

```text
explicit_scale_interaction_summary.csv
```
