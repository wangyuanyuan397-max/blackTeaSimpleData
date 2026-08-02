# ViT / Swin on datasets_01234_BaSic

This folder contains the two Transformer backbone runs for the BaSiC
preprocessed 01234 grid30 408-patch dataset.

The shared training settings are inherited from
`configs/fixed_split_01234_BaSic_grid30_408_train.yaml`, whose `dataset_root`
is `datasets_01234_BaSic`.

Models:

- `fixed_timm_vit_small_patch16_224_BaSic_grid30_408`: `vit_small_patch16_224`
- `fixed_timm_swin_tiny_patch4_window7_224_BaSic_grid30_408`:
  `swin_tiny_patch4_window7_224`

The source dataset remains the BaSiC 408-patch dataset. These configs override
only the transform size to 224 so the standard ImageNet pretrained ViT/Swin
checkpoints can run reliably.
