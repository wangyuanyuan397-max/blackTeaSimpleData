# MixNet-S Channel Shuffle Search

This directory contains the isolated brute-force search for channel fusion after
MixNet-S MixConv blocks.  It is separate from `temp/mixnet_structure_search`,
which searches MixConv kernel placement and kernel sizes.

The search keeps the dataset and training settings fixed, then changes only:

- `fusion_type`
- `target_groups`
- `partial_ratio`
- `stage_mask`
- `block_indices`
- `shuffle_mode`

## Files

- `mixnet_channel_shuffle_backbone.py`
  - Registers `mixnet_s_channel_shuffle_search`.
  - Implements `baseline`, `group`, `shuffle_group`, `group_shuffle`,
    `shuffle_dense`, `extra_group`, `extra_shuffle_group`, and `partial_mix`.
  - Records actual per-block group fallback metadata.
- `generate_channel_shuffle_configs.py`
  - Generates model YAML files under `configs/`.
- `train_mixnet_channel_shuffle_search.py`
  - Loads the local backbone, forwards configs to `tools/train_batch.py`, and
    writes channel-shuffle metadata into `test_metrics.json`.
- `common_01234_basic_408_train.yaml`
  - Fixed BaSiC 408 training settings.

## Generate Configs

First batch, the clean replacement comparison:

```powershell
python temp\mixnet_channel_shuffle_search\generate_channel_shuffle_configs.py --phase operators
```

Additive structures:

```powershell
python temp\mixnet_channel_shuffle_search\generate_channel_shuffle_configs.py --phase additive
```

Default all currently means `operators + additive`:

```powershell
python temp\mixnet_channel_shuffle_search\generate_channel_shuffle_configs.py --phase all
```

After picking a good operator, generate stage-level search configs:

```powershell
python temp\mixnet_channel_shuffle_search\generate_channel_shuffle_configs.py --phase stage_single --fusion-type shuffle_group --target-groups 4
python temp\mixnet_channel_shuffle_search\generate_channel_shuffle_configs.py --phase stage_mask --fusion-type shuffle_group --target-groups 4
```

Generate block-level probes:

```powershell
python temp\mixnet_channel_shuffle_search\generate_channel_shuffle_configs.py --phase block_single --fusion-type shuffle_group --target-groups 4
```

Generate local block subset enumeration only for a small chosen block window:

```powershell
python temp\mixnet_channel_shuffle_search\generate_channel_shuffle_configs.py --phase block_subset --fusion-type shuffle_group --target-groups 4 --subset-blocks S3B0 S3B1 S3B2 S4B0 --max-subset-size 4
```

## Run

Smoke check:

```powershell
conda run -n yolov8 python temp\mixnet_channel_shuffle_search\train_mixnet_channel_shuffle_search.py --phase smoke --dry-run --device cpu
```

Run the first 11 replacement experiments:

```powershell
conda run -n yolov8 python temp\mixnet_channel_shuffle_search\train_mixnet_channel_shuffle_search.py --phase operators --device auto
```

Run additive experiments:

```powershell
conda run -n yolov8 python temp\mixnet_channel_shuffle_search\train_mixnet_channel_shuffle_search.py --phase additive --device auto
```

Run a custom config directory:

```powershell
conda run -n yolov8 python temp\mixnet_channel_shuffle_search\train_mixnet_channel_shuffle_search.py --config-dir temp\mixnet_channel_shuffle_search\configs\stage_mask --device auto
```

Results go to:

```text
temp/mixnet_channel_shuffle_search/runs_BaSic
```

Each successful run writes `mixnet_channel_shuffle` metadata into
`test_metrics.json`.  The batch runner also writes:

```text
channel_shuffle_search_summary.csv
```

## Recommended Order

1. Run `operators`.
2. Pick the best 2-3 replacement configs using validation metrics only.
3. Run `additive` as a separate comparison group.
4. For the best config, run `stage_single`, then `stage_mask`.
5. Run `block_single` only inside promising stages.
6. Run `block_subset` only on a small local block window.
7. Re-train final candidates with multiple seeds and report `mean +- std`.
