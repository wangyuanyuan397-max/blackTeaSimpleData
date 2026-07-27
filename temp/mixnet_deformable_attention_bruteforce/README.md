# MixNet-S Deformable Attention Brute Force

This isolated experiment keeps the `fixed_timm_mixnet_s` training settings fixed
and searches only one variable:

```text
Which MixNet-S spatial stages receive DeformableAttention2d?
```

Bit order:

```text
Dabcde = [S0, S1, S2, S3, S4]
0 = no deformable attention
1 = insert after the last MixNet block at that spatial resolution
```

For the current 408 input, the discovered MixNet-S resolution groups are expected
to be approximately:

```text
S0: 204 x 204
S1: 102 x 102
S2:  51 x 51
S3:  26 x 26
S4:  13 x 13
```

The insertion point is discovered by actually forwarding a dummy tensor through
timm MixNet-S and grouping consecutive blocks with the same spatial resolution.
This avoids hard-coding fragile block indices.

Generate all 96 YAML configs:

```powershell
python temp\mixnet_deformable_attention_bruteforce\generate_deform_configs.py
```

Smoke-test the backbone:

```powershell
conda run -n yolov8 python temp\mixnet_deformable_attention_bruteforce\smoke_test_deform_backbone.py --input-size 224 --device cpu
```

List the generated configs:

```powershell
conda run -n yolov8 python temp\mixnet_deformable_attention_bruteforce\train_deform_sweep.py --generate --list-models
```

Run the full 32 x 3 sweep:

```powershell
conda run -n yolov8 python temp\mixnet_deformable_attention_bruteforce\train_deform_sweep.py --device auto
```

Run a small subset:

```powershell
conda run -n yolov8 python temp\mixnet_deformable_attention_bruteforce\train_deform_sweep.py --models D00000_seed42 D00100_seed42 D11111_seed42 --device auto
```

Outputs stay under:

```text
temp/mixnet_deformable_attention_bruteforce/runs_grid30_408
```

After training, the runner writes:

```text
deform_attention_all_results.csv
deform_attention_summary_by_config.csv
deform_attention_single_stage_summary.csv
deform_attention_stage_frequency_above_baseline.csv
```

The summary ranking uses validation metrics and the configured complexity
penalty. Test metrics are reported but not used for structure selection.
