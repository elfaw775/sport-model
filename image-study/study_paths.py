"""Repository data, pinned weights and writable local experiment directories."""
import os
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
PUBLISHED = PACKAGE / "published"
DATA = Path(os.environ.get("SPORT_XRAY_DATA", REPO / "training/data/sporthorse_v2")).expanduser().resolve()
WORK = Path(os.environ.get("SPORT_XRAY_RUN", REPO / "work/image-study")).expanduser().resolve()
CACHE = Path(os.environ.get("SPORT_XRAY_CACHE",
    Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "sports-horse-xray")).expanduser().resolve()
DETECTOR_RUN = Path(os.environ.get("SPORT_XRAY_DETECTOR_RUN", WORK / "detector-baseline")).expanduser().resolve()
YOLO_WEIGHTS = Path(os.environ.get("SPORT_XRAY_YOLO", CACHE / "weights/yolo11s.pt")).expanduser().resolve()
CV = WORK / "cv84"


def locked_input(name):
    """Fresh holdout locks hash both scripts and generated artifacts."""
    return (PACKAGE if "/" not in name else WORK) / name
