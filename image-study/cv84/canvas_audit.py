"""Change only external padding on the published DINOv2 five-fold comparison."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import fit_probe, probe_scores, write_json
from hf_features import FrozenEncoder, SIZE
from study_paths import DATA, PUBLISHED


def canvas(image, variant):
    gray = ImageOps.exif_transpose(image).convert("L")
    scale = SIZE / max(gray.size)
    size = tuple(max(1, round(axis * scale)) for axis in gray.size)
    content = gray.resize(size, Image.Resampling.BICUBIC)
    origin = tuple((SIZE-axis)//2 for axis in size)
    if variant == "aligned_start":
        origin = (0, 0)
    frame = Image.new("L", (SIZE, SIZE), 32 if variant == "gray_padding" else 0)
    frame.paste(content, origin)
    return frame


def run(output):
    folds = json.loads((PUBLISHED / "cv84/folds.json").read_bytes())
    partitions = json.loads((PUBLISHED / "cv84/detector-report.json").read_bytes())["folds"]
    with (PUBLISHED / "cv84/three-model-oof.csv").open(encoding="utf-8-sig", newline="") as f:
        reference = {row["image"]: row for row in csv.DictReader(f)}
    ids = folds["ids"]
    y = np.array(folds["labels"])
    files = {p.stem:p for p in (DATA/"images").glob("*/*") if p.is_file()}
    assert set(ids) == set(files) == set(reference) and len(ids) == 84
    encoder = FrozenEncoder("dinov2")
    variants = ("original", "gray_padding", "aligned_start")
    x = {}
    for name in variants:
        vectors = []
        for start in range(0, len(ids), 6):
            images = []
            for key in ids[start:start+6]:
                with Image.open(files[key]) as im:
                    images.append(canvas(im, name))
            vectors.append(encoder.encode(images))
        x[name] = np.concatenate(vectors)
    out = {k:np.full(len(ids), np.nan) for k in variants}
    for fold in partitions:
        fit, held = fold["fit_indices"], fold["test_indices"]
        head = fit_probe(x["original"][fit], y[fit])
        for name in variants:
            out[name][held] = probe_scores(head, x[name][held])
    threshold = np.array([float(reference[k]["dino_threshold"]) for k in ids])
    assert all(np.isfinite(scores).all() for scores in out.values())
    pred = {k:s >= threshold for k,s in out.items()}
    assert np.array_equal(pred["original"], [bool(int(reference[k]["dino"])) for k in ids])
    reference_scores = np.array([float(reference[k]["dino_score"]) for k in ids])
    drift = float(max(abs(reference_scores-out["original"])))
    if drift > 0.002:
        raise ValueError(f"Reference scores changed by {drift}")
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in variants:
        decisions = pred[name]
        summary[name] = dict(correct=int(sum(decisions==y)), missed=int(sum(~decisions & (y==1))),
                             false_alarms=int(sum(decisions & (y==0))), flips=int(sum(decisions!=pred["original"])))
    rows = [dict(image=k, truth=int(y[i]), fold=int(reference[k]["fold"]), threshold=threshold[i],
                 scores={name:float(out[name][i]) for name in variants}) for i,k in enumerate(ids)]
    write_json(output / "canvas-audit.json", dict(n=len(ids), design="fixed seed-42 matched-budget folds, frozen thresholds",
               max_baseline_score_drift=drift, score_rule="positive if score >= threshold", summary=summary, films=rows))
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
