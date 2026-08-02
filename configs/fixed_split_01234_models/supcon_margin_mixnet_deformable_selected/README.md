# SupCon/Margin x MixNet-S deformable-attention selected sweep

This folder contains 8 selected 01234 BaSiC/grid30/408 experiments for `tools/train_batch_01234.py`.

Common training settings inherit from:

```text
configs/fixed_split_01234_BaSic_grid30_408_train.yaml
```

Included configs:

- 2 selected `supcon_margin_selected` recipes.
- 4 MixNet-S backbone variants from `temp/mixnet_deformable_attention_bruteforce`: `D11000_seed2026`, `D10101_seed2026`, `D00000_seed42`, and `D11011_seed2026`.
- 8 combined YAMLs total.

For each combined model, the SupCon/Margin recipe supplies `data.train_transform`, `model.type`, `projector`, `classifier_feature`, cosine-margin `head`, `loss`, and `optimizer.lr`; the D config supplies the MixNet-S backbone and random seed.
