# SupCon/Margin x MixNet structure selected sweep

This folder contains 24 selected 01234 BaSiC/grid30/408 experiments for `tools/train_batch_01234.py`.

Common training settings inherit from:

```text
configs/fixed_split_01234_BaSic_grid30_408_train.yaml
```

Included configs:

- 3 selected `supcon_margin_selected` recipes.
- 8 selected `mixnet_structure_selected` kernel plans.
- 24 combined YAMLs total.

For each combined model, the SupCon/Margin recipe supplies `data.train_transform`, `model.type`, `projector`, `classifier_feature`, cosine-margin `head`, `loss`, and `optimizer.lr`; the MixNet structure recipe supplies `model.backbone`.
