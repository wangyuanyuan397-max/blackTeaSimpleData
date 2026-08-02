# ViT / Swin on datasets_01234_grid30_408

This folder contains the two Transformer backbone runs for the 01234 grid30
408-patch dataset.

The shared training settings are inherited from
`configs/fixed_split_01234_grid30_408_train.yaml`.

Models:

- `fixed_timm_vit_small_patch16_224_grid30_408`: `vit_small_patch16_224`
- `fixed_timm_swin_tiny_patch4_window7_224_grid30_408`:
  `swin_tiny_patch4_window7_224`

Both configs keep the project classifier head and use ImageNet pretrained timm
backbones. The source dataset remains the 408-patch dataset, while these two
model configs override the transform size to 224 to match the standard ViT/Swin
ImageNet pretrained checkpoints.
