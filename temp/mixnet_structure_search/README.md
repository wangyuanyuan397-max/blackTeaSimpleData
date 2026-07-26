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

The actual configurable backbone is registered as `mixnet_s_search`. It starts
from `timm.create_model("mixnet_s", pretrained=true, num_classes=0)`, replaces
only the planned `conv_dw` modules, and initializes changed depthwise kernels
from the pretrained kernels by centered crop or centered zero-padding.

`train_mixnet_structure_search.py` forwards to `tools/train_batch.py` with
`--discard-pth` enforced and replaces disk checkpoints with an in-memory best
state dict for final test evaluation. Passing `--keep-pth` is rejected so large
structure search batches do not write `.pth` files under the runs directory.
