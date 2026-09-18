# Kubernetes runtime configurations

These files contain container/PVC paths and are intended for Kubernetes Jobs.
They are kept separate from local experiment configurations so that local
Windows paths and cluster-mounted paths cannot be confused.

For `fixed_split_01234_grid30_408_train.yaml`:

- input dataset: `/data/datasets/datasets_01234_grid30_408`
- persistent outputs: `/data/runs/datasets_01234_grid30_408`
- input PVC: `black-tea-dataset`
- output PVC: `black-tea-runs`

This is a common batch-training configuration. A Kubernetes Job must also
select one or more model YAML files through the batch training entry point.

