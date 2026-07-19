"""Run one INSID3 segmentation and clevis-correspondence episode."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make the runner work both as a file and from the Docker working directory.
INSID3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(INSID3_ROOT))

from models import build_insid3
from utils.visualization import (
    visualize_prediction_matching,
    visualize_prediction_segmentation,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--model-size", default="base", choices=("small", "base", "large"))
    parser.add_argument("--image-size", default=768, type=int)
    return parser.parse_args()


def binary_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def main() -> None:
    args = parse_args()
    with (EXPERIMENT_DIR / "episode.json").open() as stream:
        episode = json.load(stream)

    data_dir = EXPERIMENT_DIR / "data"
    output_dir = EXPERIMENT_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    reference = data_dir / episode["reference_image"]
    reference_mask = data_dir / episode["reference_mask"]
    target = data_dir / episode["target_image"]
    target_mask = data_dir / episode["target_mask"]

    model = build_insid3(
        model_size=args.model_size,
        image_size=args.image_size,
        device=args.device,
    ).to(args.device).eval()

    # INSID3's loader currently dispatches filesystem inputs only for ``str``.
    model.set_reference(str(reference), str(reference_mask))
    model.set_target(str(target))
    prediction = model.segment().detach().cpu().numpy().astype(bool)
    Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").save(
        output_dir / "target_instrument_prediction.png"
    )
    visualize_prediction_segmentation(
        reference,
        reference_mask,
        target,
        prediction,
        output_dir / "segmentation.png",
    )

    source_point = episode["reference_clevis_xy"]
    target_point = episode["target_clevis_xy"]
    model.set_reference(str(reference))
    model.set_target(str(target))
    predicted_point = model.match(torch.tensor([source_point]), use_debiased=True)[0]
    visualize_prediction_matching(
        reference,
        [source_point],
        target,
        predicted_point.unsqueeze(0),
        output_dir / "clevis_keypoint.png",
    )

    truth = binary_mask(target_mask)
    intersection = np.logical_and(prediction, truth).sum()
    union = np.logical_or(prediction, truth).sum()
    predicted_xy = predicted_point.detach().cpu().tolist()
    metrics = {
        "instrument_iou": float(intersection / union) if union else 1.0,
        "clevis_error_px": math.dist(predicted_xy, target_point),
        "clevis_error_normalized_by_image_diagonal": math.dist(predicted_xy, target_point)
        / math.hypot(*Image.open(target).size),
        "predicted_clevis_xy": predicted_xy,
        "target_clevis_xy": target_point,
        "model_size": args.model_size,
        "image_size": args.image_size,
    }
    with (output_dir / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
