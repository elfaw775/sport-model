"""Score blinded Qwen judgments and archived YOLO predictions on the frozen horse splits."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median
from zipfile import ZipFile

from PIL import Image

from compare_detectors import ROOT, WORK, counts, truth


COMPARISON = ROOT / "training/results/comparison-2026-11-18/predictions.zip"
UNRESOLVED = ("uncertain", "parse_error", "inference_error")
AXES = ("x1", "y1", "x2", "y2")


def load_qwen(path):
    rows = {"val": {}, "test": {}}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        split, name = row["split"], row["filename"]
        if split not in rows or name in rows[split]:
            raise ValueError(f"Unexpected split or duplicate result: {split}/{name}")
        status = row["status"]
        if status not in ("positive", "negative", *UNRESOLVED):
            raise ValueError(f"Unrecognized status for {name}: {status}")
        if not isinstance(row["boxes"], list) or (row["boxes"] and status != "positive"):
            raise ValueError(f"Inconsistent box/status for {name}")
        with Image.open(WORK / "dataset/images" / split / name) as image:
            width, height = image.size
        boxes = []
        for box in row["boxes"]:
            if (len(box) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in box)
                    or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)):
                raise ValueError(f"Invalid normalized box for {name}: {box}")
            pixels = (v * dim / 1000 for v, dim in zip(box, (width, height, width, height)))
            boxes.append(dict(zip(AXES, pixels), score=1.0))  # Equal scores preserve generation order.
        rows[split][name] = (status, boxes)
    return rows


def load_yolo(archive, model, split, cutoff=0.15):
    predictions = json.loads(archive.read(f"{model}/{split}_predictions.json"))
    rows = {}
    for name, boxes in predictions.items():
        kept = [box for box in boxes if box["score"] >= cutoff]
        rows[name] = ("positive" if kept else "negative", kept)
    return rows


def train_median_box():
    corners = []
    for path in (WORK / "dataset/labels/train").glob("*.txt"):
        for line in path.read_text().splitlines():
            _, cx, cy, width, height = map(float, line.split())
            corners.append((cx-width/2, cy-height/2, cx+width/2, cy+height/2))
    return [median(values) for values in zip(*corners)]


def fixed_boxes(split, labels, corners):
    rows = {}
    for name in labels:
        with Image.open(WORK / "dataset/images" / split / name) as image:
            width, height = image.size
        box = dict(zip(AXES, (v * dim for v, dim in zip(corners, (width, height, width, height)))))
        rows[name] = ("positive", [dict(box, score=1.0)])
    return rows


def score(labels, rows):
    if labels.keys() != rows.keys():
        raise ValueError(f"Split mismatch: missing {labels.keys()-rows.keys()}, extra {rows.keys()-labels.keys()}")
    outcomes = Counter(tp=0, fp=0, tn=0, fn=0)
    unresolved = Counter({reason: 0 for reason in UNRESOLVED})
    positive_without_box = 0
    predictions = {}
    for name, targets in labels.items():
        status, boxes = rows[name]
        predictions[name] = boxes
        if status in UNRESOLVED:
            unresolved[status] += 1
        elif status == "positive":
            outcomes["tp" if targets else "fp"] += 1
            positive_without_box += int(not boxes)
        else:
            outcomes["fn" if targets else "tn"] += 1
    positives = sum(bool(boxes) for boxes in labels.values())
    negatives = len(labels) - positives
    sensitivity = outcomes["tp"] / positives
    specificity = outcomes["tn"] / negatives
    # All images remain in each denominator, including undecided or malformed answers.
    image_level = dict(reference_positive=positives, reference_negative=negatives, **outcomes,
                       unresolved=unresolved, unresolved_positive=positives-outcomes["tp"]-outcomes["fn"],
                       unresolved_negative=negatives-outcomes["tn"]-outcomes["fp"],
                       positive_without_box=positive_without_box, sensitivity=sensitivity, specificity=specificity,
                       accuracy=(outcomes["tp"] + outcomes["tn"]) / len(labels),
                       balanced_accuracy=(sensitivity + specificity) / 2)
    lesion_level = counts(predictions, labels, cutoff=0)
    lesion_level["reference_lesions"] = sum(len(targets) for targets in labels.values())
    lesion_level["predicted_boxes"] = sum(len(boxes) for boxes in predictions.values())
    return {"image_level": image_level, "lesion_level_iou50": lesion_level}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="Local Qwen normalized JSONL output")
    args = parser.parse_args()
    qwen = load_qwen(args.predictions)
    results = {}
    prior = train_median_box()
    archived_thresholds = json.loads(COMPARISON.with_name("metrics.json").read_text())
    with ZipFile(COMPARISON) as archive:
        for split in ("val", "test"):
            labels = truth(split)
            results[split] = {"qwen": score(labels, qwen[split]),
                              "deployed_yolo": score(labels, load_yolo(archive, "shipped", split)),
                              "corrected_yolo640": score(labels, load_yolo(archive, "yolo", split))}
            always_positive = {name: ("positive", []) for name in labels}
            results[split]["always_positive_image_only"] = {
                "image_level": score(labels, always_positive)["image_level"]}
            results[split]["train_median_box"] = {
                "normalized_xyxy": prior, **score(labels, fixed_boxes(split, labels, prior))}
            for key, model in (("deployed_yolo", "shipped"), ("corrected_yolo640", "yolo")):
                cutoff = archived_thresholds[model]["val_selected_cutoff"]
                results[split][key + "_val_selected"] = {
                    "cutoff": cutoff, **score(labels, load_yolo(archive, model, split, cutoff))}
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
