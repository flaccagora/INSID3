# SurgVU instrument and clevis smoke experiment

This fixed one-shot episode tests two INSID3 capabilities on the same surgical
instrument in SurgVU case 123:

- full instrument segmentation from a reference image and instrument mask;
- clevis semantic correspondence from one reference clevis point.

The target instrument mask and target clevis point are used only for scoring.
They are not provided to INSID3 during prediction.

The exported inputs are committed beside this file under `data/`. To reproduce
them from the clevis repository's video and RLE cache, run from the repository
root:

```bash
.venv/bin/python submodules/INSID3/experiments/clevis_case123/prepare_data.py
```

Run INSID3 in its existing container:

```bash
cd submodules/INSID3
docker compose run --rm insid3 \
  python experiments/clevis_case123/run_experiment.py
```

The experiment deliberately defaults to DINOv3-base because that checkpoint is
already present in `pretrain/`. Results are written to
`experiments/clevis_case123/output/`: a raw predicted mask, segmentation and
keypoint visualizations, and `metrics.json` with instrument IoU and clevis point
error.
