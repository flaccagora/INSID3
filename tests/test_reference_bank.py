import numpy as np

from experiments.reference_bank import (
    Candidate,
    associate_keypoint,
    connected_candidates,
    fuse_candidates,
    mask_iou,
    weighted_geometric_median,
)


def test_connected_components_and_keypoint_association():
    mask = np.zeros((30, 40), dtype=bool)
    mask[2:12, 3:15] = True
    mask[17:28, 25:38] = True
    confidence = np.where(mask, 0.8, 0.0)
    candidates = connected_candidates(mask, confidence, reference_name="ref", min_area=20)
    assert len(candidates) == 2
    associate_keypoint(candidates, (30.0, 22.0), 0.9, max_distance=5.0)
    assert [candidate.keypoint_xy for candidate in candidates] == [None, (30.0, 22.0)]


def test_duplicate_masks_and_points_are_fused():
    first = np.zeros((20, 20), dtype=bool)
    second = np.zeros((20, 20), dtype=bool)
    first[4:14, 5:15] = True
    second[5:15, 5:15] = True
    assert mask_iou(first, second) > 0.5
    detections = fuse_candidates(
        [
            Candidate(first, 0.9, "a", (9.0, 8.0), 0.8),
            Candidate(second, 0.8, "b", (10.0, 9.0), 0.9),
        ],
        mask_confidence_threshold=0.2,
        keypoint_confidence_threshold=0.2,
    )
    assert len(detections) == 1
    assert detections[0].supporting_references == ["a", "b"]
    assert 9.0 <= detections[0].clevis_xy[0] <= 10.0


def test_unrelated_and_low_confidence_candidates_are_not_merged():
    left = np.zeros((20, 30), dtype=bool)
    right = np.zeros((20, 30), dtype=bool)
    left[2:12, 2:10] = True
    right[3:13, 20:28] = True
    detections = fuse_candidates(
        [Candidate(left, 0.9, "left"), Candidate(right, 0.1, "right")],
        mask_confidence_threshold=0.2,
        keypoint_confidence_threshold=0.0,
    )
    assert len(detections) == 1
    assert detections[0].supporting_references == ["left"]


def test_weighted_geometric_median_resists_outlier():
    point = weighted_geometric_median(
        np.asarray([[10.0, 10.0], [11.0, 10.0], [100.0, 100.0]]),
        np.asarray([1.0, 1.0, 0.1]),
    )
    assert point[0] < 20 and point[1] < 20
