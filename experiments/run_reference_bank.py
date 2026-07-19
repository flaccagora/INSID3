"""Exhaustive multi-reference INSID3 instance segmentation and clevis matching."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

INSID3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSID3_ROOT))

from experiments.reference_bank import (  # noqa: E402
    Candidate,
    associate_keypoint,
    connected_candidates,
    fuse_candidates,
    mask_iou,
)
from models import build_insid3  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an exhaustive INSID3 reference bank")
    parser.add_argument("--references", type=Path, required=True, help="reference_cases.json")
    parser.add_argument("--targets", type=Path, required=True, help="evaluation_targets.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--model-size", default="base", choices=("small", "base", "large"))
    parser.add_argument("--image-size", default=768, type=int)
    parser.add_argument("--calibration-count", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def _load_catalog(path: Path, expected_format: str) -> list[tuple[dict, Path]]:
    path = path.resolve()
    raw = json.loads(path.read_text())
    rows = raw if isinstance(raw, list) else [raw]
    loaded = []
    for row in rows:
        metadata_path = path.parent / row["path"] if "path" in row else path
        metadata = json.loads(metadata_path.read_text()) if metadata_path != path else row
        if metadata.get("format") != expected_format or metadata.get("version") != 1:
            raise ValueError(f"Unsupported manifest at {metadata_path}")
        loaded.append((metadata, metadata_path.parent))
    return loaded


def _mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def predict_candidates(model, references, target_image: Path) -> list[Candidate]:
    width, height = Image.open(target_image).size
    min_area = max(64, int(round(width * height * 0.0001)))
    all_candidates = []
    for reference, reference_dir in references:
        image = reference_dir / reference["image"]
        mask = reference_dir / reference["mask"]
        model.set_reference(str(image), str(mask))
        model.set_target(str(target_image))
        prediction, confidence = model.segment_with_confidence()
        candidates = connected_candidates(
            prediction.detach().cpu().numpy(),
            confidence.detach().cpu().numpy(),
            reference_name=reference["name"],
            min_area=min_area,
        )
        model.set_reference(str(image))
        model.set_target(str(target_image))
        point, point_confidence = model.match_with_confidence(
            torch.tensor([reference["clevis_xy"]]), use_debiased=True
        )
        associate_keypoint(
            candidates,
            tuple(point[0].detach().cpu().tolist()),
            float(point_confidence[0].detach().cpu()),
            max_distance=0.05 * math.hypot(width, height),
        )
        all_candidates.extend(candidates)
    return all_candidates


def _truth(target: dict, target_dir: Path) -> list[tuple[np.ndarray, tuple[float, float]]]:
    return [
        (_mask(target_dir / instance["mask"]), tuple(instance["clevis_xy"]))
        for instance in target["instances"]
    ]


def evaluate(detections, truth, iou_threshold: float = 0.5) -> dict:
    if detections and truth:
        ious = np.asarray([[mask_iou(det.mask, item[0]) for item in truth] for det in detections])
        pred_idx, truth_idx = linear_sum_assignment(1.0 - ious)
        matches = [(int(p), int(t), float(ious[p, t])) for p, t in zip(pred_idx, truth_idx) if ious[p, t] >= iou_threshold]
    else:
        matches = []
    tp = len(matches)
    precision = tp / len(detections) if detections else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = []
    diagonal = math.hypot(*truth[0][0].shape[::-1]) if truth else 1.0
    for pred_index, truth_index, _iou in matches:
        if detections[pred_index].clevis_xy is not None:
            errors.append(math.dist(detections[pred_index].clevis_xy, truth[truth_index][1]))
    keypoint_predictions = sum(det.clevis_xy is not None for det in detections)
    keypoint_tp = len(errors)
    keypoint_precision = (
        keypoint_tp / keypoint_predictions
        if keypoint_predictions
        else (1.0 if not truth else 0.0)
    )
    keypoint_recall = keypoint_tp / len(truth) if truth else 1.0
    keypoint_f1 = (
        2 * keypoint_precision * keypoint_recall / (keypoint_precision + keypoint_recall)
        if keypoint_precision + keypoint_recall
        else 0.0
    )
    return {
        "instrument_tp": tp,
        "instrument_predictions": len(detections),
        "instrument_truth": len(truth),
        "instrument_precision": precision,
        "instrument_recall": recall,
        "instrument_f1": f1,
        "matched_mean_iou": float(np.mean([row[2] for row in matches])) if matches else None,
        "keypoint_tp": keypoint_tp,
        "keypoint_predictions": keypoint_predictions,
        "keypoint_truth": len(truth),
        "keypoint_precision": keypoint_precision,
        "keypoint_recall": keypoint_recall,
        "keypoint_f1": keypoint_f1,
        "keypoint_errors_px": errors,
        "keypoint_mean_error_px": float(np.mean(errors)) if errors else None,
        "keypoint_mean_error_normalized": float(np.mean(errors) / diagonal) if errors else None,
        "pck_01": float(np.mean(np.asarray(errors) <= 0.01 * diagonal)) if errors else None,
        "pck_02": float(np.mean(np.asarray(errors) <= 0.02 * diagonal)) if errors else None,
        "pck_05": float(np.mean(np.asarray(errors) <= 0.05 * diagonal)) if errors else None,
    }


def calibrate(raw_targets, count: int) -> tuple[float, float]:
    development = raw_targets[:count]
    if not development:
        return 0.2, 0.0
    best = (-1.0, 0.2, 0.0)
    for mask_threshold in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        for point_threshold in (0.0, 0.2, 0.4, 0.6, 0.8):
            scores = []
            for candidates, truth in development:
                detections = fuse_candidates(
                    candidates,
                    mask_confidence_threshold=mask_threshold,
                    keypoint_confidence_threshold=point_threshold,
                )
                metrics = evaluate(detections, truth)
                scores.append((metrics["instrument_f1"] + metrics["keypoint_f1"]) / 2)
            score = float(np.mean(scores))
            if score > best[0]:
                best = (score, mask_threshold, point_threshold)
    return best[1], best[2]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    references = _load_catalog(args.references, "clevis.insid3-reference-case")
    targets = _load_catalog(args.targets, "clevis.insid3-evaluation-target")
    model = build_insid3(
        model_size=args.model_size, image_size=args.image_size, device=args.device
    ).to(args.device).eval()

    raw = []
    for target, target_dir in targets:
        image = target_dir / target["image"]
        raw.append((predict_candidates(model, references, image), _truth(target, target_dir)))
    mask_threshold, point_threshold = calibrate(raw, min(args.calibration_count, len(raw)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_results = []
    aggregate_metrics = []
    for index, ((target, _target_dir), (candidates, truth)) in enumerate(zip(targets, raw)):
        detections = fuse_candidates(
            candidates,
            mask_confidence_threshold=mask_threshold,
            keypoint_confidence_threshold=point_threshold,
        )
        target_dir = args.output_dir / target["name"]
        target_dir.mkdir(exist_ok=True)
        rows = []
        for detection_index, detection in enumerate(detections):
            mask_name = f"prediction_{detection_index:02d}.png"
            Image.fromarray(detection.mask.astype(np.uint8) * 255, mode="L").save(target_dir / mask_name)
            rows.append({
                "mask": mask_name,
                "confidence": detection.confidence,
                "clevis_xy": detection.clevis_xy,
                "clevis_confidence": detection.clevis_confidence,
                "supporting_references": detection.supporting_references,
            })
        metrics = evaluate(detections, truth)
        split = "calibration" if index < args.calibration_count else "evaluation"
        target_results.append({"name": target["name"], "split": split, "detections": rows, "metrics": metrics})
        if split == "evaluation":
            aggregate_metrics.append(metrics)

    aggregate_keys = (
        "instrument_precision",
        "instrument_recall",
        "instrument_f1",
        "matched_mean_iou",
        "keypoint_precision",
        "keypoint_recall",
        "keypoint_f1",
        "keypoint_mean_error_px",
        "keypoint_mean_error_normalized",
        "pck_01",
        "pck_02",
        "pck_05",
    )
    evaluation_metrics = {
        key: float(np.mean([m[key] for m in aggregate_metrics if m[key] is not None]))
        if any(m[key] is not None for m in aggregate_metrics)
        else None
        for key in aggregate_keys
    }
    summary = {
        "format": "insid3.reference-bank-results",
        "version": 1,
        "reference_count": len(references),
        "target_count": len(targets),
        "model_size": args.model_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "mask_confidence_threshold": mask_threshold,
        "keypoint_confidence_threshold": point_threshold,
        "evaluation_metrics": evaluation_metrics,
        "targets": target_results,
    }
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "targets"}, indent=2))


if __name__ == "__main__":
    main()
