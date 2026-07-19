"""Prepare a cross-video SurgVU episode from clevis repository inputs."""

from pathlib import Path
from shutil import copy2

import cv2


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[3]
SAME_CLIP_DATA = EXPERIMENT_DIR.parent / "clevis_case123" / "data"
TARGET_VIDEO = REPO_ROOT / "data/surgvu_smoke/sample_videos/case122/case122.mp4"
TARGET_FRAME = 600


def main() -> None:
    data_dir = EXPERIMENT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    copy2(SAME_CLIP_DATA / "reference.png", data_dir / "reference_case123_f660.png")
    copy2(
        SAME_CLIP_DATA / "reference_instrument_mask.png",
        data_dir / "reference_case123_f660_mask.png",
    )

    capture = cv2.VideoCapture(str(TARGET_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {TARGET_VIDEO}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
        ok, frame = capture.read()
        if not ok or not cv2.imwrite(str(data_dir / "target_case122_f600.png"), frame):
            raise RuntimeError(f"Could not export frame {TARGET_FRAME} from {TARGET_VIDEO}")
    finally:
        capture.release()
    print(f"Prepared cross-video episode in {data_dir}")


if __name__ == "__main__":
    main()
