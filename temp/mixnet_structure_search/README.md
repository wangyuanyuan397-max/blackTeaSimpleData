# MixNet-S Structure Search

This experiment keeps the BaSiC grid30 dataset and all training settings fixed,
then changes only MixNet-S depthwise/MixConv placement, kernel sets, and optional
scale gates.

Dataset and training config:

- dataset: `datasets_01234_BaSic`
- common config: `temp/mixnet_structure_search/common_01234_basic_408_train.yaml`
- output root: `temp/mixnet_structure_search/runs_BaSic`
- `.pth` checkpoints are not written for search runs

Generate configs:

```powershell
python temp\mixnet_structure_search\generate_mixnet_search_configs.py --phase all
```

Run the three smoke configs:

```powershell
conda run -n yolov8 python temp\mixnet_structure_search\train_mixnet_structure_search.py --phase smoke --dry-run --device cpu
```

Run the first 20 placement experiments:

```powershell
conda run -n yolov8 python temp\mixnet_structure_search\train_mixnet_structure_search.py --phase position --device auto
```

Available phases:

- `position`: P00-P19, fixed `K357` except original and all-3x3 baseline.
- `stage_mask`: all 64 stage masks with `K357`.
- `kernel_continuous`: default top-position templates crossed with K35/K357/K3579/K357911.
- `gates`: G0-G3 templates on `P03_STRIDE2_K357`.

Follow-up batches:

```powershell
python temp\mixnet_structure_search\generate_mixnet_followup_configs.py --phase all
```

This writes:

- `configs/followup_seed_repro`: 4 structures x seeds 42/3407/2026.
- `configs/followup_champion_gates`: only-S0 and S2+S3+S5 crossed with G0/G1/G2/G3.
- `configs/followup_s235_kernel_grid`: S2/S3/S5 crossed with K3/K35/K357/K3579.

Queue all three priorities in one run:

```powershell
conda run -n yolov8 python temp\mixnet_structure_search\train_mixnet_structure_search.py --phase followup_all --generate-followup --device auto
```

`followup_all` runs the three directories in priority order and writes the
follow-up CSV summaries at the end.

Run priority 1, the multi-seed reproduction:

```powershell
conda run -n yolov8 python temp\mixnet_structure_search\train_mixnet_structure_search.py --config-dir temp\mixnet_structure_search\configs\followup_seed_repro --device auto
python temp\mixnet_structure_search\summarize_mixnet_followup_results.py
```

Run priority 2, only the six new champion-gate configs if the two G0 runs are
already available:

```powershell
conda run -n yolov8 python temp\mixnet_structure_search\train_mixnet_structure_search.py --config-dir temp\mixnet_structure_search\configs\followup_champion_gates --models gate_only_s0_k357_g1_static gate_only_s0_k357_g2_sigmoid gate_only_s0_k357_g3_softmax gate_s235_k357_g1_static gate_s235_k357_g2_sigmoid gate_s235_k357_g3_softmax --device auto
python temp\mixnet_structure_search\summarize_mixnet_followup_results.py
```

Run priority 3, the S2/S3/S5 stage-specific kernel grid:

```powershell
conda run -n yolov8 python temp\mixnet_structure_search\train_mixnet_structure_search.py --config-dir temp\mixnet_structure_search\configs\followup_s235_kernel_grid --device auto
python temp\mixnet_structure_search\summarize_mixnet_followup_results.py
```

The follow-up summarizer writes:

- `mixnet_followup_all_results.csv`
- `mixnet_seed_repro_summary.csv`
- `mixnet_champion_gate_summary.csv`
- `mixnet_s235_kernel_grid_summary.csv`

The actual configurable backbone is registered as `mixnet_s_search`. It starts
from `timm.create_model("mixnet_s", pretrained=true, num_classes=0)`, replaces
only the planned `conv_dw` modules, and initializes changed depthwise kernels
from the pretrained kernels by centered crop or centered zero-padding.

`train_mixnet_structure_search.py` forwards to `tools/train_batch.py` with
`--discard-pth` enforced and replaces disk checkpoints with an in-memory best
state dict for final test evaluation. Passing `--keep-pth` is rejected so large
structure search batches do not write `.pth` files under the runs directory.
