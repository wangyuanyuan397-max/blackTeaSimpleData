# MixNet-S SimSiam SSL Experiment

This temporary experiment targets:

- dataset: `datasets_01234_BaSic`
- classes: `00`, `10`, `20`, `30`, `40`
- backbone: `timm` model `mixnet_s`
- stage 1: unlabeled SimSiam self-supervised pretraining
- stage 2: supervised five-class finetuning from the SSL backbone

All new code lives in this folder and writes run artifacts under
`temp/mixnet_simsiam_ssl/runs/`.

## 1. Pretrain Without Labels

Default command. It uses only `datasets_01234_BaSic/train` as unlabeled images,
so validation/test images are not seen before evaluation.

```powershell
python temp\mixnet_simsiam_ssl\train_simsiam_pretrain.py
```

The main output is:

```text
temp/mixnet_simsiam_ssl/runs/simsiam_mixnet_s_basic408/mixnet_s_simsiam_backbone.pth
```

If GPU memory is tight, use smaller batches:

```powershell
python temp\mixnet_simsiam_ssl\train_simsiam_pretrain.py --batch-size 8
```

If you explicitly want to use all splits as unlabeled images for SSL:

```powershell
python temp\mixnet_simsiam_ssl\train_simsiam_pretrain.py --ssl-splits train val test
```

That can improve representation learning, but it is a transductive setting
because the test images are seen without labels.

## 2. Finetune With Labels

Run after the SSL checkpoint exists:

```powershell
python temp\mixnet_simsiam_ssl\finetune_mixnet_classifier.py
```

Important outputs:

```text
temp/mixnet_simsiam_ssl/runs/finetune_mixnet_s_basic408/best_model.pth
temp/mixnet_simsiam_ssl/runs/finetune_mixnet_s_basic408/summary.json
temp/mixnet_simsiam_ssl/runs/finetune_mixnet_s_basic408/history.csv
```

`summary.json` contains best validation metrics and final test metrics.

## 3. Finetune Tuning Sweep

Your current SSL run looks more underfit than overfit: train and validation
accuracy are both around 0.60. The first tuning pass therefore keeps the SSL
checkpoint fixed and relaxes the supervised finetuning stage:

- weaker train augmentation: `resize_flip` or milder crop
- lower weight decay
- higher AdamW learning rate
- optionally faster classifier-head learning rate
- optional checkpoint selection by `val_qwk` for the ordinal task

List recommended variants:

```powershell
python temp\mixnet_simsiam_ssl\run_finetune_sweep.py --list
```

Preview commands without running:

```powershell
python temp\mixnet_simsiam_ssl\run_finetune_sweep.py --dry-run --exclude-sgd
```

Run the recommended AdamW variants first:

```powershell
python temp\mixnet_simsiam_ssl\run_finetune_sweep.py --exclude-sgd
```

Run one selected variant:

```powershell
python temp\mixnet_simsiam_ssl\run_finetune_sweep.py --variant resize_lr3e4_wd1e4_acc
```

The sweep writes:

```text
temp/mixnet_simsiam_ssl/runs/tuning/sweep_results.csv
```

Each variant also has its own `summary.json`, `history.csv`, and
`best_model.pth` under `temp/mixnet_simsiam_ssl/runs/tuning/{variant_name}/`.

## Fast Smoke Checks

These commands only check that the pipeline can instantiate models and batches:

```powershell
python temp\mixnet_simsiam_ssl\train_simsiam_pretrain.py --dry-run --image-size 224 --batch-size 4 --num-workers 0 --max-samples 8
```

For finetune smoke testing without a real SSL checkpoint:

```powershell
python temp\mixnet_simsiam_ssl\finetune_mixnet_classifier.py --dry-run --ssl-checkpoint none --image-size 224 --batch-size 4 --eval-batch-size 8 --num-workers 0 --max-samples 20
```

## Notes

- Default `image_size=408` matches the current BaSiC/grid30 MixNet baseline.
- The SSL script does not require `lightly`; SimSiam loss and heads are implemented
  directly in PyTorch.
- Windows multiprocessing requires pickle-safe dataloader worker helpers. The
  scripts support `--num-workers 4`; if your local machine still has worker
  startup trouble, pass `--num-workers 0` to run single-process loading.
- By default SSL starts from random MixNet-S weights. Add `--imagenet-pretrained`
  if you want ImageNet initialization before SSL.
- Finetuning defaults to AdamW with `lr=1e-4`, aligned with the existing BaSiC
  baseline config. Use `--optimizer sgd --lr 0.001` to try the SGD style from
  the pasted sketch.
