"""Export the fixed SurgVU reference/target episode into the INSID3 checkout."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[3]
VIDEO = REPO_ROOT / "data/surgvu_smoke/sample_videos/case123/case123.mp4"
MASK_ROOT = REPO_ROOT / "data/cache_surgvu_smoke/1_segmentation/instrument/case_123/0"
DATA_DIR = EXPERIMENT_DIR / "data"

# Track 1 is the lower-right monopolar curved scissors in both frames.
FRAMES = {"reference": 660, "target": 900}
TRACK_ID = 1


def load_mask(frame_idx: int) -> np.ndarray:
    mask_path = MASK_ROOT / str(frame_idx) / f"{TRACK_ID}.npz"
    with np.load(mask_path, allow_pickle=False) as encoded:
        rle = {
            "size": encoded["size"].tolist(),
            "counts": encoded["counts"].tobytes(),
        }
    return mask_utils.decode(rle).astype(np.uint8)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {VIDEO}")

    try:
        for role, frame_idx in FRAMES.items():
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {frame_idx} from {VIDEO}")
            if not cv2.imwrite(str(DATA_DIR / f"{role}.png"), frame_bgr):
                raise RuntimeError(f"Could not write {role}.png")
            Image.fromarray(load_mask(frame_idx) * 255, mode="L").save(
                DATA_DIR / f"{role}_instrument_mask.png"
            )
    finally:
        capture.release()

    print(f"Exported case123 frames 660 and 900 (instrument track 1) to {DATA_DIR}")


if __name__ == "__main__":
    main()
