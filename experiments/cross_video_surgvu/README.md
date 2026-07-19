# Modular cross-video INSID3 experiment

This episode uses a masked instrument from SurgVU case 123 as context and a
target frame from case 122. The target identity is the right-hand USM3
MegaSutureCut Needle Driver; other visible instruments are distractors. The two
frames differ in video, anatomy, viewpoint, lighting, and instrument type.

Prepare the checked-in inputs again from the parent clevis repository:

```bash
.venv/bin/python submodules/INSID3/experiments/cross_video_surgvu/prepare_data.py
```

Run the generic manifest-driven experiment inside the existing container:

```bash
cd submodules/INSID3
docker compose run --rm insid3 \
  python experiments/run_episode.py \
  experiments/cross_video_surgvu/episode.json
```

## Reusing the runner

Copy `episode.json` and change its paths. Paths are resolved relative to the
manifest. `references` accepts one or more masked images for segmentation.
`correspondence.reference_index` selects the single reference used for
keypoint matching.

Both target annotations are optional:

- add `target.mask` to compute segmentation IoU;
- add `correspondence.target_keypoints_xy` to compute keypoint error.

Optional annotations are held out for scoring and are never passed to INSID3.
Use `--output-dir` to keep results from multiple runs separate.
