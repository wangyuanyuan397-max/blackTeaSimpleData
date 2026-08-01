# AdamNorm LR sweep for 01234 BaSiC grid30 408

These configs test the local `adamnorm` optimizer with a fixed MixNet-S
structure and a small learning-rate sweep.

Common settings are inherited from:

```text
configs/fixed_split_01234_BaSic_grid30_408_train.yaml
```

Controlled variables:

- Dataset, split, seed, epochs, batch sizes, transforms, scheduler, loss,
  patience, TTA, and PTH cleanup stay in the common config.
- Model structure is fixed to `stagemask_001101_k357`.
- Optimizer type is `adamnorm`.
- AdamNorm `gamma` is fixed at `0.95`.
- Only `optimizer.lr` changes across these YAML files.

Run through:

```powershell
python tools\train_batch_01234.py
```
