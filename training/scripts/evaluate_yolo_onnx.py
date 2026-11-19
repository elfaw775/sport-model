"""Evaluate a hash-pinned, single-class 640px YOLO ONNX on the fixed horse splits."""
import argparse
import json
from pathlib import Path

from PIL import Image

from compare_detectors import WORK, score_run
from backend import detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    session = detector._session(args.model, args.sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    for split in ("val", "test"):
        predictions = {}
        for path in sorted((WORK / "dataset/images" / split).iterdir()):
            with Image.open(path) as image:
                tensor, scale, left, top = detector._prepare(image)
                output, = session.run(None, {session.get_inputs()[0].name: tensor})
                predictions[path.name] = detector._decode(output, scale, left, top, *image.size, threshold=.01)
        (args.output / f"{split}_predictions.json").write_text(json.dumps(predictions, ensure_ascii=False))
    (args.output / "metrics.json").write_text(json.dumps(score_run(args.output), indent=2) + "\n")


if __name__ == "__main__":
    main()
