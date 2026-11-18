"""Score last-epoch checkpoints and the shipped detector against the corrected labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work/model-comparison"
sys.path.insert(0, str(ROOT))


def truth(split):
    result = {}
    for path in (WORK / "dataset/images" / split).iterdir():
        with Image.open(path) as image:
            width, height = image.size
        lines = (WORK / "dataset/labels" / split / (path.stem + ".txt")).read_text().splitlines()
        boxes = []
        for line in lines:
            cls, cx, cy, bw, bh = map(float, line.split())
            assert cls == 0
            boxes.append(((cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height))
        result[path.name] = boxes
    return result


def iou(box, target):
    left, top = max(box[0],target[0]), max(box[1],target[1])
    right, bottom = min(box[2],target[2]), min(box[3],target[3])
    overlap = max(0, right-left) * max(0, bottom-top)
    area = (box[2]-box[0])*(box[3]-box[1]) + (target[2]-target[0])*(target[3]-target[1])
    return overlap / max(area-overlap, 1e-12)


def counts(predictions, labels, cutoff):
    assert predictions.keys() == labels.keys()
    tally = Counter(tp=0,fp=0,fn=0)
    for name, targets in labels.items():
        remaining = list(targets)
        for hit in sorted(predictions[name], key=lambda item: -item["score"]):
            if hit["score"] < cutoff:
                continue
            box = [hit[key] for key in ("x1","y1","x2","y2")]
            overlaps = [iou(box,target) for target in remaining]
            if overlaps and max(overlaps) >= .5:
                tally["tp"] += 1
                remaining.pop(overlaps.index(max(overlaps)))
            else:
                tally["fp"] += 1
        tally["fn"] += len(remaining)
    return dict(tally)


def shipped():
    from backend import detector
    detector.THRESHOLD = .01
    folder = WORK / "shipped"
    folder.mkdir(exist_ok=True)
    for split in ("val","test"):
        records = {}
        for path in (WORK / "dataset/images" / split).iterdir():
            with Image.open(path) as image:
                records[path.name] = detector.detect(image)
        (folder / (split + "_predictions.json")).write_text(json.dumps(records,ensure_ascii=False))


def compare():
    labels = {split:truth(split) for split in ("val","test")}
    results = {}
    for name in ("shipped","yolo","frcnn"):
        files = dict(val="predictions.json" if name == "frcnn" else "val_predictions.json",test="test_predictions.json")
        predicted = {split:json.loads((WORK / name / fname).read_text()) for split,fname in files.items()}
        # Fixed grid; validation only. Bound the mean number of false positives at one per film.
        grid = [.01,.025,.05,.075,.10,.15,.20,.25,.30,.40,.50,.60,.70,.80,.90,.95,.99,1.001]
        feasible = [(threshold, counts(predicted["val"],labels["val"],threshold)) for threshold in grid]
        feasible = [(threshold, score) for threshold,score in feasible if score["fp"] <= len(labels["val"])]
        threshold, _ = max(feasible, key=lambda candidate:(candidate[1]["tp"],-candidate[1]["fp"],-candidate[0]))
        results[name] = {"val_selected_cutoff":threshold}
        for key,cutoff in (("fixed_0.15",.15),("val_selected",threshold)):
            results[name][key] = {split:counts(predicted[split], labels[split], cutoff) for split in labels}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shipped",action="store_true",help="Export shipped ONNX predictions (backend dependencies required)")
    args = parser.parse_args()
    if args.shipped:
        shipped()
    else:
        report = json.dumps(compare(), indent=2)
        (WORK / "comparison.json").write_text(report + "\n")
        print(report)
