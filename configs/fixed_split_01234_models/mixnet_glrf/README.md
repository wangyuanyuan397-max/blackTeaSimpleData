# MixNet-S Global-Local Reliability Fusion

This isolated queue tests evidence acquisition and fusion without modifying
the internal MixNet-S blocks or the cross-entropy loss.

The historical baseline actually consumes 408x408 tensors. To preserve that
baseline exactly, every GLRF variant keeps the full global view at 408x408;
the two local views use deterministic/random 288 crops resized to 224x224.
All views share one timm `mixnet_s` backbone.

The six controlled variants are:

- `A0`: ordinary 408x408 MixNet-S baseline.
- `A1`: GLRF wrapper, global view only; checks interface equivalence.
- `A2`: two local views; mean local fusion and mean global/local fusion.
- `A3`: reliability-weighted locals and mean global/local fusion.
- `A4`: mean locals and learned global/local gate.
- `A5`: reliability-weighted locals and learned global/local gate.

A2-A5 use physical batch 8 with four-step gradient accumulation, preserving
the baseline effective batch size of 32. Validation and test local crops use
fixed top-left and bottom-right anchors so early stopping and final metrics are
reproducible. Training resamples both local crops on every access.

These six configurations remain here as completed experiment records.
`tools/train_batch_01234.py` now targets the separate
`mixnet_orthoshot/phase1_dbt/` queue, so the GLRF files are no longer the
active runner queue.

Each GLRF test prediction CSV adds global/local weights, both local reliability
weights, branch predictions, and their disagreement flag. Test metrics include
overall and per-class gate means and disagreement rates. Batch completion also
writes `mixnet_glrf_summary.csv` under the common runs root.
