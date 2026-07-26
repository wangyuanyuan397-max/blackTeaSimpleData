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
- `include_prefixes: ["backbone."]`
- `exclude_prefixes: ["head.", "aux_head.", "classifier."]`

With `normalize: mean`, the L2-SP term is averaged by constrained parameter
count. If you change it to `sum`, reduce `alpha` substantially.
