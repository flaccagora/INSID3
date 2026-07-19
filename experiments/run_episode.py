"""Manifest-driven INSID3 segmentation and keypoint experiment runner."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

INSID3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSID3_ROOT))

from models import build_insid3  # noqa: E402
from utils.visualization import (  # noqa: E402
    visualize_prediction_matching,
    visualize_prediction_segmentation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an INSID3 episode described by a JSON manifest."
    )
    parser.add_argument("episode", type=Path, help="Episode JSON manifest")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--model-size", default="base", choices=("small", "base", "large"))
    parser.add_argument("--image-size", default=768, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def resolve_path(manifest_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def binary_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def load_episode(path: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = path.resolve()
    with manifest_path.open() as stream:
        episode = json.load(stream)
    if not episode.get("references"):
        raise ValueError("episode must contain at least one entry in 'references'")
    if "target" not in episode or "image" not in episode["target"]:
        raise ValueError("episode must contain target.image")
    return episode, manifest_path.parent


def run(args: argparse.Namespace) -> dict[str, Any]:
    episode, manifest_dir = load_episode(args.episode)
    output_dir = (args.output_dir or manifest_dir / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    references = episode["references"]
    target_spec = episode["target"]
    target_image = resolve_path(manifest_dir, target_spec["image"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_insid3(
        model_size=args.model_size,
        image_size=args.image_size,
        device=args.device,
    ).to(args.device).eval()
    metrics: dict[str, Any] = {
        "episode": episode.get("name", args.episode.stem),
        "model_size": args.model_size,
        "image_size": args.image_size,
        "seed": args.seed,
    }

    masked_references = [ref for ref in references if ref.get("mask")]
    if masked_references:
        for reference in masked_references:
            model.set_reference(
                str(resolve_path(manifest_dir, reference["image"])),
                str(resolve_path(manifest_dir, reference["mask"])),
            )
        model.set_target(str(target_image))
        prediction = model.segment().detach().cpu().numpy().astype(bool)
        Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").save(
            output_dir / "target_segmentation_prediction.png"
        )
        first_ref = masked_references[0]
        visualize_prediction_segmentation(
            resolve_path(manifest_dir, first_ref["image"]),
            resolve_path(manifest_dir, first_ref["mask"]),
            target_image,
            prediction,
            output_dir / "segmentation.png",
        )
        if target_spec.get("mask"):
            truth = binary_mask(resolve_path(manifest_dir, target_spec["mask"]))
            intersection = np.logical_and(prediction, truth).sum()
            union = np.logical_or(prediction, truth).sum()
            metrics["segmentation_iou"] = float(intersection / union) if union else 1.0
        else:
            metrics["segmentation_iou"] = None

    correspondence = episode.get("correspondence")
    if correspondence:
        reference_index = int(correspondence.get("reference_index", 0))
        reference = references[reference_index]
        source_points = correspondence["reference_keypoints_xy"]
        model.set_reference(str(resolve_path(manifest_dir, reference["image"])))
        model.set_target(str(target_image))
        predicted_points = model.match(
            torch.tensor(source_points), use_debiased=correspondence.get("debiased", True)
        )
        visualize_prediction_matching(
            resolve_path(manifest_dir, reference["image"]),
            source_points,
            target_image,
            predicted_points,
            output_dir / "keypoints.png",
        )
        predicted_xy = predicted_points.detach().cpu().tolist()
        metrics["predicted_keypoints_xy"] = predicted_xy
        target_points = correspondence.get("target_keypoints_xy")
        if target_points is not None:
            errors = [math.dist(pred, truth) for pred, truth in zip(predicted_xy, target_points)]
            target_size = Image.open(target_image).size
            metrics["keypoint_errors_px"] = errors
            metrics["mean_keypoint_error_px"] = sum(errors) / len(errors)
            metrics["mean_keypoint_error_normalized_by_image_diagonal"] = (
                metrics["mean_keypoint_error_px"] / math.hypot(*target_size)
            )
        else:
            metrics["keypoint_errors_px"] = None

    with (output_dir / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
