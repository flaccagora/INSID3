# INSID3 — Docker Environment

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quick Start

```bash
# Build image
docker compose build

# Run inference (interactive)
docker compose run --rm insid3 python inference_segmentation.py \
    --dataset coco --exp-name insid3-coco
```

## Data & Weights

Place pretrained DINOv3 weights in `pretrain/` and datasets in `data/` — both are mounted into the container.

## Interactive Shell

```bash
docker compose run --rm insid3 bash
```
