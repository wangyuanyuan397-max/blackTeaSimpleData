# SAFNet paper/source comparison

This isolated queue is selected through `tools/train_batch_01234.py` and uses
`configs/fixed_split_01234_BaSic_grid30_408_train.yaml` as its common config.

Models:

1. `paper_resnet50`: ImageNet-pretrained torchvision ResNet50.
2. `paper_mobilenet_v3_large`: ImageNet-pretrained torchvision MobileNetV3-Large.
3. `paper_convnext_tiny`: ImageNet-pretrained torchvision ConvNeXt-Tiny.
4. `paper_safnet_scratch`: source-faithful SCAM-ResNet50 + AMSAFF SAFNet,
   initialized from scratch as in the supplied Python source.

All models share the five classes 00/10/20/30/40, 408 input, seed 2026,
150 epochs, cross-entropy loss, AdamW and cosine scheduling.

```powershell
python tools\train_batch_01234.py --safnet-comparison --list-models
python tools\train_batch_01234.py --safnet-comparison --dry-run --device cpu
python tools\train_batch_01234.py --safnet-comparison --device cuda
```
