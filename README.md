# Simplified Fermentation Classification Framework

This directory is an independent, reduced copy of the main project. It keeps
the YAML configuration, model registry, component builder, trainer, evaluator,
and the frequently used backbones and losses.

## Entry points

```bash
python tools/train.py --config configs/full6fold_server_b16a2/baselines/resnet50_strict_ce.yaml
python tools/evaluate.py --config CONFIG.yaml --checkpoint runs/EXP/best_model.pth
```

## Dataset adapter

Dataset storage belongs under `datasets/`. The data adapter is intentionally
not copied because the new dataset will not use the current project's LOGOCV
and patch-voting protocol.

Before training, add `src/data/loader.py` and register the required dataset and
transforms with `DATASETS` and `TRANSFORMS`. The adapter must expose
`build_dataloader(...)` and return train, validation, and test datasets whose
items contain at least `(image, label, path)`.

## Scope

The copy excludes LOGOCV runners, traditional-feature runners, paper plotting,
monitoring, Google Drive upload, and the large collection of one-off attention
ablations.
