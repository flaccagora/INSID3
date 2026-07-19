"""Pure post-processing utilities for exhaustive INSID3 reference banks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class Candidate:
    mask: np.ndarray
    mask_confidence: float
    reference_name: str
    keypoint_xy: tuple[float, float] | None = None
    keypoint_confidence: float | None = None


@dataclass
class Detection:
    mask: np.ndarray
    confidence: float
    clevis_xy: tuple[float, float] | None
    clevis_confidence: float | None
    supporting_references: list[str] = field(default_factory=list)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 1.0


def connected_candidates(
    mask: np.ndarray,
    confidence: np.ndarray,
    *,
    reference_name: str,
    min_area: int,
) -> list[Candidate]:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    candidates = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        if int(component.sum()) < min_area:
            continue
        candidates.append(
            Candidate(
                mask=component,
                mask_confidence=float(np.asarray(confidence)[component].mean()),
                reference_name=reference_name,
            )
        )
    return candidates


def associate_keypoint(
    candidates: list[Candidate],
    point_xy: tuple[float, float],
    confidence: float,
    *,
    max_distance: float,
) -> None:
    if not candidates:
        return
    x = int(round(point_xy[0]))
    y = int(round(point_xy[1]))
    for candidate in candidates:
        if 0 <= y < candidate.mask.shape[0] and 0 <= x < candidate.mask.shape[1] and candidate.mask[y, x]:
            candidate.keypoint_xy = point_xy
            candidate.keypoint_confidence = confidence
            return
    best = None
    for candidate in candidates:
        ys, xs = np.nonzero(candidate.mask)
        if len(xs) == 0:
            continue
        distance = float(np.sqrt((xs - point_xy[0]) ** 2 + (ys - point_xy[1]) ** 2).min())
        if best is None or distance < best[0]:
            best = (distance, candidate)
    if best is not None and best[0] <= max_distance:
        best[1].keypoint_xy = point_xy
        best[1].keypoint_confidence = confidence


def weighted_geometric_median(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    estimate = np.average(points, axis=0, weights=weights)
    for _ in range(64):
        distances = np.linalg.norm(points - estimate, axis=1)
        if np.any(distances < 1e-6):
            return points[int(np.argmin(distances))]
        adjusted = weights / np.maximum(distances, 1e-6)
        updated = np.sum(points * adjusted[:, None], axis=0) / adjusted.sum()
        if np.linalg.norm(updated - estimate) < 1e-3:
            return updated
        estimate = updated
    return estimate


def fuse_candidates(
    candidates: list[Candidate],
    *,
    mask_confidence_threshold: float,
    keypoint_confidence_threshold: float,
    nms_iou: float = 0.5,
    vote_threshold: float = 0.5,
) -> list[Detection]:
    remaining = sorted(
        [candidate for candidate in candidates if candidate.mask_confidence >= mask_confidence_threshold],
        key=lambda candidate: candidate.mask_confidence,
        reverse=True,
    )
    groups: list[list[Candidate]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        keep = []
        for candidate in remaining:
            if mask_iou(seed.mask, candidate.mask) >= nms_iou:
                group.append(candidate)
            else:
                keep.append(candidate)
        groups.append(group)
        remaining = keep

    detections = []
    for group in groups:
        weights = np.asarray([max(item.mask_confidence, 1e-6) for item in group])
        mask_stack = np.stack([item.mask for item in group]).astype(np.float32)
        fused_mask = np.tensordot(weights, mask_stack, axes=(0, 0)) / weights.sum() >= vote_threshold
        point_items = [
            item
            for item in group
            if item.keypoint_xy is not None
            and item.keypoint_confidence is not None
            and item.keypoint_confidence >= keypoint_confidence_threshold
        ]
        clevis = None
        clevis_confidence = None
        if point_items:
            point_weights = np.asarray([max(item.keypoint_confidence, 1e-6) for item in point_items])
            clevis_arr = weighted_geometric_median(
                np.asarray([item.keypoint_xy for item in point_items], dtype=np.float64),
                point_weights,
            )
            clevis = (float(clevis_arr[0]), float(clevis_arr[1]))
            clevis_confidence = float(np.average(
                [item.keypoint_confidence for item in point_items], weights=point_weights
            ))
        detections.append(
            Detection(
                mask=fused_mask,
                confidence=float(max(item.mask_confidence for item in group)),
                clevis_xy=clevis,
                clevis_confidence=clevis_confidence,
                supporting_references=sorted({item.reference_name for item in group}),
            )
        )
    return detections
