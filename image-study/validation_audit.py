"""Paired perturbations on positive validation films; never used for selection."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch

from study_paths import WORK
from evaluate import cases, choose_threshold, fit_probe, load_npz, probe_scores
from hf_features import FrozenEncoder, SOURCE
from local_patch import CORRECTED


def read_boxes(path, width, height):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        _, cx, cy, w, h = map(float, line.split())
        boxes.append((max(0, math.floor((cx-w/2)*width)), max(0, math.floor((cy-h/2)*height)),
                      min(width, math.ceil((cx+w/2)*width)), min(height, math.ceil((cy+h/2)*height))))
    return boxes


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def matched_background(gray, boxes):
    height, width = gray.shape
    placed = []
    # For each real box use a disjoint rectangle of identical size and similar mean gray.
    for x0, y0, x1, y1 in sorted(boxes, key=lambda b: -(b[2]-b[0])*(b[3]-b[1])):
        w, h = x1-x0, y1-y0
        ref = float(gray[y0:y1, x0:x1].mean())
        options = []
        for y in np.linspace(0, height-h, 17).astype(int):
            for x in np.linspace(0, width-w, 17).astype(int):
                other = (int(x), int(y), int(x+w), int(y+h))
                if not any(intersects(other, b) for b in boxes+placed):
                    options.append((abs(float(gray[y:y+h, x:x+w].mean())-ref), other))
        if not options:
            return None
        placed.append(min(options)[1])
    return placed


def blurred(image, boxes):
    result = image.copy()
    for box in boxes:
        radius = max(1, min(box[2]-box[0], box[3]-box[1])/6)
        result.paste(image.crop(box).filter(ImageFilter.GaussianBlur(radius=radius)), box)
    return result


def main():
    torch.set_num_threads(4)
    train, _ = load_npz("dinov2", "train", "X")
    head = fit_probe(train, cases("train")[1])
    val, _ = load_npz("dinov2", "val", "X")
    ids, labels = cases("val")
    original_scores = probe_scores(head, val)
    threshold = choose_threshold(labels, original_scores)
    paths = {p.stem:p for p in (SOURCE / "images/val").iterdir() if p.is_file()}
    encoder = FrozenEncoder("dinov2")
    records = []
    skipped = []
    for i, (stem, label) in enumerate(zip(ids, labels)):
        if not label:
            continue
        with Image.open(paths[stem]) as raw:
            if raw.getexif().get(274, 1) != 1:
                raise ValueError("Bounding box alignment requires stored orientation")
            image = raw.convert("L")
        boxes = read_boxes(CORRECTED/"labels/val"/(stem+".txt"), *image.size)
        control = matched_background(np.asarray(image), boxes)
        if not control:
            skipped.append(stem)
            continue
        variants = [image, blurred(image, boxes), blurred(image, control)]
        scores = probe_scores(head, encoder.encode(variants))
        if abs(scores[0] - original_scores[i]) > 1e-5:
            raise ValueError("Original score does not reproduce")
        records.append(dict(stem=stem, original=float(scores[0]), lesion_blur=float(scores[1]),
                            sham_blur=float(scores[2]), boxes=len(boxes)))
    matrix = np.array([[r["original"],r["lesion_blur"],r["sham_blur"]] for r in records])
    # Use cached scores for the unmodified threshold decision to avoid device batch round-off.
    matrix[:, 0] = [original_scores[ids.index(r["stem"])] for r in records]
    summary = dict(split="validation positive films with a disjoint matched rectangle", n=len(records),
                   skipped=skipped, threshold=threshold, positive_counts=(matrix>=threshold).sum(axis=0).tolist(),
                   median_score_change=(np.median(matrix[:, 1:]-matrix[:, :1], axis=0)).tolist(),
                   lesion_below_sham=int((matrix[:, 1]<matrix[:, 2]).sum()), rows=records)
    output = WORK / "results/validation-audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
