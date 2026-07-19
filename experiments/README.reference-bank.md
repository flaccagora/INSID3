# Exhaustive surgical-instrument reference bank

`run_reference_bank.py` consumes the catalogs produced by the clevis
reference-case GUI. It runs every reference independently so an unrelated
majority cannot suppress a rare instrument, splits predictions into instances,
and fuses duplicate masks and clevis predictions.

```bash
python experiments/run_reference_bank.py \
  --references /inputs/reference_cases.json \
  --targets /inputs/evaluation_targets.json \
  --output-dir output/reference_bank_pilot \
  --model-size base \
  --image-size 768 \
  --calibration-count 5
```

The first `--calibration-count` targets select mask and keypoint confidence
thresholds. Remaining targets are the held-out evaluation split. Results include
instance precision/recall/F1, matched mask IoU, clevis pixel/normalized error,
PCK, per-detection confidence, and supporting reference names.

Reference catalogs use `clevis.insid3-reference-case` version 1. Evaluation
catalogs use `clevis.insid3-evaluation-target` version 1. Catalog entries point
to self-contained case manifests, and all image/mask paths are relative to those
manifest directories.
