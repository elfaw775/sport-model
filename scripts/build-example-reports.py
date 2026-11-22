"""Recompute the two public read-only examples. Run with --check to verify the published results."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import detector, screening
from backend.demo_library import demo_library

EXAMPLES = (("18", "18-正位.jpg"), ("75", "75-正位 p1骨折.jpg"))
OUTPUT = ROOT / "backend/static/example-reports.json"


def reference_boxes(label, width, height):
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        category, *coords = line.split()
        if category != "0":
            raise ValueError(f"Unexpected training category in {label.name}: {category}")
        cx, cy, w, h = map(float, coords)
        boxes.append({"x1": max(0.0, (cx - w / 2) * width), "y1": max(0.0, (cy - h / 2) * height),
                      "x2": min(width, (cx + w / 2) * width), "y2": min(height, (cy + h / 2) * height)})
    return boxes


def build(labels):
    # Use the same rotation, clipping, class filter and precision as training.
    subprocess.run([sys.executable, ROOT / "training/scripts/coco2yolo.py",
                    ROOT / "training/source/coco/instances_default.json", "--out", labels, "--keep", "病变"], check=True)
    catalog, files = demo_library(ROOT / "training/data/sporthorse_v2")
    by_name = {files[row["filename"]].name: row for row in catalog}
    examples = []
    for identifier, filename in EXAMPLES:
        row = by_name[filename]
        path = files[row["filename"]]
        with Image.open(path) as image:
            film = screening.classify(image)
            original = ImageOps.exif_transpose(image)
            width, height = original.size
            detections = detector.detect(original)
        examples.append({"id": identifier, "filename": filename, "public_filename": row["filename"],
                         "projection": row["projection"], "width": width, "height": height, "screening": film,
                         "model": detector.MODEL_ID, "threshold": detector.THRESHOLD, "detections": detections,
                         "reference_boxes": reference_boxes(labels / f"{path.stem}.txt", width, height),
                         "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return examples


def matches(saved, actual):
    if type(saved) is not type(actual):
        return False
    if isinstance(actual, dict):
        return saved.keys() == actual.keys() and all(matches(saved[key], value) for key, value in actual.items())
    if isinstance(actual, list):
        return len(saved) == len(actual) and all(matches(a, b) for a, b in zip(saved, actual))
    if isinstance(actual, float):
        return math.isclose(saved, actual, rel_tol=1e-5, abs_tol=1e-6)
    return saved == actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the existing JSON without overwriting it")
    args = parser.parse_args()
    work = ROOT / "work/qa/example-reports"
    work.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=work) as labels:
        reports = build(Path(labels))
    if args.check:
        if not matches(json.loads(OUTPUT.read_text(encoding="utf-8")), reports):
            raise SystemExit("Published example reports differ from the source images, labels or models.")
        print(f"Verified {len(reports)} reports against the source images, labels and ONNX models.")
    else:
        OUTPUT.write_text(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote {len(reports)} reports to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
