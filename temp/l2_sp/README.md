# L2-SP MixNet-S baseline

This temporary experiment keeps the original baseline configs unchanged:

- `configs/fixed_split_01234_models/fixed_timm_mixnet_s.yaml`
- `configs/fixed_split_01234_grid30_408_train.yaml`

Run a config/data check:

```powershell
conda run -n yolov8 python temp\l2_sp\train_l2_sp_mixnet_s.py --dry-run
```

Run training:

```powershell
conda run -n yolov8 python temp\l2_sp\train_l2_sp_mixnet_s.py
```

The L2-SP settings are in `temp/l2_sp/fixed_timm_mixnet_s_l2sp_alpha001.yaml`:

- `alpha: 0.01`
- `normalize: mean`
- `constrained_weight_decay: 0.0`
- `include_prefixes: ["backbone."]`
- `exclude_prefixes: ["head.", "aux_head.", "classifier."]`

With `normalize: mean`, the L2-SP term is averaged by constrained parameter
count. If you change it to `sum`, reduce `alpha` substantially.

The constrained backbone parameter group uses `weight_decay=0.0` so it is pulled
toward the pretrained starting point by L2-SP instead of being simultaneously
pulled toward zero by ordinary weight decay. Unconstrained parameters such as the
linear head keep the optimizer weight decay from the common training config.
