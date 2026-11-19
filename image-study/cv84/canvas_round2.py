"""Paired held-out scores for fixed padding conditions and a fit-only augmentation candidate."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import fit_probe, probe_scores, unit
from hf_features import FrozenEncoder, SIZE
from nested import midpoint_threshold
from study_paths import DATA, PUBLISHED


ORIGINAL = "black_center"
# Position in quarters of available padding; floor reproduces the original centering.
CONDITIONS = {ORIGINAL: (0, 2), **{f"gray_{shade}": (shade, 2) for shade in (8, 16, 32, 64, 128)},
              **{f"align_{25*q}": (0, q) for q in (0, 1, 3, 4)}}
TRAIN_VIEWS = (ORIGINAL, "gray_16", "align_25", "align_75")
BASELINE, AUGMENTED = "locked_original", "fit_augmented"
PLAN = Path(__file__).with_name("canvas-round2-plan.md")
SOURCE = PUBLISHED / "cv84"


def make_content(path):
    with Image.open(path) as im:
        gray = ImageOps.exif_transpose(im).convert("L")
    scale = SIZE / max(gray.size)
    size = tuple(max(1, round(axis * scale)) for axis in gray.size)
    return gray.resize(size, Image.Resampling.BICUBIC)


def encode_conditions(contents):
    encoder = FrozenEncoder("dinov2")
    features = {}
    for name, (color, quarter) in CONDITIONS.items():
        chunks = []
        for start in range(0, len(contents), 6):
            batch = []
            for content in contents[start:start+6]:
                location = tuple((SIZE-axis)*quarter//4 for axis in content.size)
                frame = Image.new("L", (SIZE, SIZE), color)
                frame.paste(content, location)
                batch.append(frame)
            chunks.append(encoder.encode(batch))
        features[name] = np.concatenate(chunks)
        print(f"Encoded {name}", flush=True)
    return features


def augmented_head(features, fit, labels):
    views = np.concatenate([features[name][fit] for name in TRAIN_VIEWS])
    normalized = unit(views)
    scaler = StandardScaler().fit(normalized)
    model = LogisticRegression(C=1, class_weight="balanced", max_iter=3000, solver="lbfgs")
    model.fit(scaler.transform(normalized), np.tile(labels[fit], len(TRAIN_VIEWS)),
              sample_weight=np.full(len(views), 1/len(TRAIN_VIEWS)))
    return dict(mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_[0], intercept=model.intercept_)


def summarize(labels, predictions, original):
    return dict(n=len(labels), positives=int(np.sum(labels)), correct=int(np.sum(predictions == labels)),
                misses=int(np.sum(~predictions & (labels == 1))),
                false_alarms=int(np.sum(predictions & (labels == 0))),
                turned_positive=int(np.sum(predictions & ~original)),
                turned_negative=int(np.sum(~predictions & original)))


def run(output):
    foldmap = json.loads((SOURCE / "folds.json").read_bytes())
    ids, y = foldmap["ids"], np.array(foldmap["labels"])
    partitions = json.loads((SOURCE / "detector-report.json").read_bytes())["folds"]
    with (SOURCE / "three-model-oof.csv").open(encoding="utf-8-sig", newline="") as f:
        reference = {r["image"]: r for r in csv.DictReader(f)}
    files = {p.stem: p for p in (DATA / "images").glob("*/*") if p.is_file()}
    assert set(ids) == set(files) == set(reference) and len(ids) == len(set(ids)) == 84
    contents = [make_content(files[key]) for key in ids]
    axes = np.array(["left_right" if im.width < SIZE else "top_bottom" if im.height < SIZE else "none"
                     for im in contents])
    features = encode_conditions(contents)
    models = (BASELINE, AUGMENTED)
    scores = {m:{v:np.full(len(ids), np.nan) for v in CONDITIONS} for m in models}
    thresholds = {m:np.full(len(ids), np.nan) for m in models}
    folds = np.zeros(len(ids), dtype=int)
    fitted = {}
    for partition in partitions:
        fit, cal, held = (np.array(partition[key]) for key in ("fit_indices","calibration_indices","test_indices"))
        assert sorted(np.concatenate((fit, cal, held))) == list(range(len(ids)))
        fold = partition["fold"] + 1
        assert not folds[held].any() and all(int(reference[ids[i]]["fold"]) == fold for i in held)
        folds[held] = fold
        locked = np.array([float(reference[ids[i]]["dino_threshold"]) for i in held])
        assert np.unique(locked).size == 1
        base = fit_probe(features[ORIGINAL][fit], y[fit])
        aug = augmented_head(features, fit, y)
        threshold_aug = midpoint_threshold(y[cal], probe_scores(aug, features[ORIGINAL][cal]))
        fitted.update({f"fold{fold}_{field}":values for field,values in aug.items()})
        fitted[f"fold{fold}_threshold"] = np.array(threshold_aug)
        for model, head, cutoff in ((BASELINE, base, locked[0]), (AUGMENTED, aug, threshold_aug)):
            thresholds[model][held] = cutoff
            for view in CONDITIONS:
                scores[model][view][held] = probe_scores(head, features[view][held])
    assert np.all(folds > 0)
    decisions = {}
    for model in models:
        assert np.isfinite(thresholds[model]).all() and all(np.isfinite(v).all() for v in scores[model].values())
        decisions[model] = {v:s >= thresholds[model] for v,s in scores[model].items()}
    prior = json.loads((SOURCE / "canvas-audit.json").read_bytes())
    prior_rows = {row["image"]: row for row in prior["films"]}
    assert np.array_equal(decisions[BASELINE][ORIGINAL], [int(reference[k]["dino"]) for k in ids])
    for view, old_view in ((ORIGINAL, "original"), ("gray_32", "gray_padding"), ("align_0", "aligned_start")):
        old = np.array([prior_rows[k]["scores"][old_view] for k in ids])
        assert np.array_equal(decisions[BASELINE][view], old >= thresholds[BASELINE])
        assert np.max(np.abs(scores[BASELINE][view] - old)) < .002
    metrics, changes, stability = {}, {}, {}
    for model, variants in decisions.items():
        baseline = variants[ORIGINAL]
        metrics[model], changes[model] = {}, {}
        for name, d in variants.items():
            metrics[model][name] = {"all":summarize(y, d, baseline), "by_padding_axis":{}}
            for axis in sorted(set(axes)):
                subset = axes == axis
                metrics[model][name]["by_padding_axis"][axis] = summarize(y[subset],d[subset],baseline[subset])
            changes[model][name] = [ids[i] for i in np.flatnonzero(d != baseline)]
        matrix = np.stack(list(variants.values()))
        all_correct = np.all(matrix == y, axis=0)
        all_wrong = np.all(matrix != y, axis=0)
        flipped = np.any(matrix != baseline, axis=0)
        stability[model] = dict(always_correct=int(sum(all_correct)), always_wrong=int(sum(all_wrong)),
                                ever_flipped=int(sum(flipped)), flipped_ids=[ids[i] for i in np.flatnonzero(flipped)],
                                always_wrong_ids=[ids[i] for i in np.flatnonzero(all_wrong)])
    output.mkdir(parents=True, exist_ok=True)
    heads = output / "augmented-fold-heads.npz"
    np.savez_compressed(heads, **fitted)
    table = output / "scores.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "image", "truth", "fold", "padding_axis", "content_width", "content_height",
                    "threshold", *CONDITIONS])
        for model in models:
            for i,key in enumerate(ids):
                w.writerow([model,key,y[i],folds[i],axes[i],*contents[i].size,thresholds[model][i],
                            *(scores[model][v][i] for v in CONDITIONS)])
    summary = dict(films=len(y), positives=int(sum(y)), device=str(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"),
                   decision_rule="score >= model and fold threshold",
                   plan_sha256=sha256(PLAN.read_bytes()).hexdigest(), scores_sha256=sha256(table.read_bytes()).hexdigest(),
                   augmented_heads_sha256=sha256(heads.read_bytes()).hexdigest(),
                   conditions=CONDITIONS, candidate_training_views=TRAIN_VIEWS,
                   stability=stability, metrics=metrics, changed_from_same_models_black_center=changes)
    (output / "summary.json").write_bytes((json.dumps(summary, ensure_ascii=False, indent=2)+"\n").encode())
    print(json.dumps(stability, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
