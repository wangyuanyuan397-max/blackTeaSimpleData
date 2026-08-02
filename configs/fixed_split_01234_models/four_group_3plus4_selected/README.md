# Four-group 3-way and 4-way selected combinations

Generated for the 01234 BaSiC grid30 408 fixed split.

Groups:
- Fourier: 3 configs.
- Deformable attention: 4 configs.
- SupCon + margin: 3 configs.
- AdamNorm: 2 configs.

Combination plan:
- Three groups at a time, max one item per group: 36 F+D+S, 24 F+D+A, 18 F+S+A, 24 D+S+A = 102 configs.
- Four groups at a time, max one item per group: 72 configs.
- Total: 174 configs.

Files:
- `CONFIG_NAMES.txt` is the run queue used by `tools/train_batch_01234.py`.
- `combinations.csv` maps each short config name back to its full source configs.

Merge rules:
- Fourier and deformable attention are merged into the backbone; F+D uses `mixnet_s_fourier_deformable`.
- SupCon + margin supplies the two-view train transform, projector, cosine-margin head, loss, and its lr when AdamNorm is not selected.
- AdamNorm overrides the optimizer type/lr/gamma. Other shared optimizer settings continue to inherit from the common training YAML.
