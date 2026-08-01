# MixNet structure x deformable attention selected sweep

This folder contains the selected 01234 BaSiC/grid30/408 MixNet-S experiments for `tools/train_batch_01234.py`.

Common training settings inherit from:

```text
configs/fixed_split_01234_BaSic_grid30_408_train.yaml
```

Included configs:

- 2 standalone deformable-attention MixNet-S models: `D11000_seed2026`, `D10101_seed2026`.
- 16 combined models: 8 selected `mixnet_structure_selected` kernel plans crossed with those 2 deformable-attention stage masks.

For the combined models, `model.backbone.type` is `mixnet_s_deformable`, with the original structure-search `kernel_plan` plus the selected `deform_stage_ids` and deformable-attention parameters.
