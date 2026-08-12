# SAFNet paper/source comparison queue

`tools/train_batch.py` currently runs these four models in order:

1. `fixed_resnet50`: torchvision ResNet50 with ImageNet weights.
2. `fixed_mobilenet_v3_large`: torchvision MobileNetV3-Large with ImageNet weights.
3. `fixed_convnext_tiny`: torchvision ConvNeXt-Tiny with ImageNet weights. The request's
   “convnet” is interpreted as the repository's existing ConvNeXt comparison.
4. `fixed_safnet_scratch`: the final SCAM-ResNet50 + AMSAFF model jointly defined by
   the supplied `SCAM_ResNet.py` and `safnet.py`, initialized from scratch.

All four use `configs/fixed_split_01234_BaSic_grid30_408_train.yaml`, including
the same fixed five-class 00/10/20/30/40 train/validation/test split, 408 input,
seed 2026, cross-entropy loss, AdamW optimizer and cosine schedule. This makes the local results directly
comparable to one another. They are not a numerical reproduction of the paper's
87.03% result because the paper uses a different five-class dataset, synthetic
DDPM/ESRGAN augmentation, and SGD with an initial learning rate of 0.01.

Inspect the queue and data:

```powershell
python tools\train_batch.py --list-models
python tools\train_batch.py --dry-run --device cpu
```

Run all models:

```powershell
python tools\train_batch.py --device cuda
```

Run only selected models:

```powershell
python tools\train_batch.py --device cuda --models fixed_resnet50 fixed_safnet_scratch
```
