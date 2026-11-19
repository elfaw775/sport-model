"""Re-score seven whole-film variants with the saved full-data and seed-42 outer-fold heads."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from folds import HERE, MANIFEST, load_rows
from nested import OUT, PRIMARY, features
from predict84 import encoder_for
from stability import VARIANTS, adjust, details, summarize

from evaluate import probe_scores, read_json, sha, write_json


@dataclass
class Plan:
    group: str
    fold: int
    winner: str
    threshold: float
    indices: list
    params: dict


def read_head(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Fitted head checksum mismatch for {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files}


def read_plans(folds):
    summary = read_json(OUT / "summary.json")
    if sha(HERE / "folds.json") != summary["folds_sha256"]:
        raise ValueError("The fitted run used different folds")
    saved = summary["refit84"]
    plans = [Plan("refit84", -1, saved["winner"], saved["threshold"],
                  list(range(len(folds["ids"]))), read_head(OUT / "refit84-head.npz", saved["head_sha256"]))]
    decisions = {int(r["fold"]):r for r in read_json(OUT / "inner_decisions.json")["42"]}
    for fold in folds["repeats"]["42"]:
        index = fold["fold"]
        row = decisions[index]
        if row["outer_test_indices"] != fold["test"]:
            raise ValueError(f"Holdout changed for outer fold {index}")
        params = read_head(OUT / f"seed42-fold{index}-head.npz", row["head_sha256"])
        winner = row["winner"]
        plans.append(Plan("oof42", index, winner, row["summaries"][winner]["threshold"], fold["test"], params))
    return plans


def load_images(rows):
    paths = {(p.parent.name, p.stem):p for p in (MANIFEST.parent / "images").glob("*/*") if p.is_file()}
    images = []
    for r in rows:
        with Image.open(paths[r["split"], r["image"]]) as im:
            images.append(im.convert("RGB"))
    return images


def score_variants(encoder, images, rows, plans, batch):
    indices = sorted({i for plan in plans for i in plan.indices})
    position = {i: j for j, i in enumerate(indices)}
    records = []
    for variant in VARIANTS:
        # Repeat passes through the GPU again. Only share vectors across heads for this variant.
        vectors = np.vstack([
            encoder.encode([adjust(images[i], variant) for i in indices[start:start + batch]])
            for start in range(0, len(indices), batch)
        ])
        if not np.isfinite(vectors).all():
            raise ValueError(f"Nonfinite features for {variant}")
        for plan in plans:
            scores = probe_scores(plan.params, vectors[[position[i] for i in plan.indices]])
            for i, score in zip(plan.indices, scores, strict=True):
                r = rows[i]
                records.append(dict(image=r["image"], old_split=r["split"], y=int(r["category"] == "pos"),
                    group=plan.group, fold=plan.fold, model=plan.winner, variant=variant, score=float(score),
                    threshold=plan.threshold, pred=int(score >= plan.threshold)))
    return records


def crosscheck(records, rows, plans):
    # Compare independently re-encoded original images against the material actually scored in nested CV.
    with (OUT / "outer_predictions.csv").open(encoding="utf-8-sig", newline="") as f:
        prior = [r for r in csv.DictReader(f) if r["seed"] == "42" and r["method"] == PRIMARY]
    if len(prior) != len(rows):
        raise ValueError("Original run's fold prediction count mismatch")
    published = {(r["id"], "oof42"):(float(r["score"]), int(r["pred"]), r["encoder"], int(r["fold"]))
                 for r in prior}
    ids = [r["image"] for r in rows]
    y = np.array([r["category"] == "pos" for r in rows], dtype=int)
    full, = [p for p in plans if p.group == "refit84"]
    X, _ = features(full.winner, ids, y)
    scores = probe_scores(full.params, X)
    published.update({(key, "refit84"):(s, int(s >= full.threshold), full.winner, -1)
                      for key,s in zip(ids, scores, strict=True)})
    observed = {(r["image"],r["group"]):r for r in records if r["variant"] == "original"}
    if set(observed) != set(published):
        raise ValueError("Original-image prediction keys differ")
    report = {}
    for group in ("refit84", "oof42"):
        deltas = []
        mismatches = []
        for key, (score, pred, model, fold) in published.items():
            if key[1] != group:
                continue
            r = observed[key]
            if (r["model"],r["fold"]) != (model,fold):
                raise ValueError(f"Original fold encoder assignment changed for {key}")
            deltas.append(abs(r["score"]-score))
            if pred != r["pred"]:
                mismatches.append(key[0])
        report[group] = dict(compared=len(deltas), max_abs_score_difference=max(deltas),
                             decisions_changed=mismatches)
        if mismatches or max(deltas) > 5e-4:
            raise RuntimeError(f"Fresh inference differs from the fitted study: {report}")
    return report


def run(batch_size):
    torch.set_num_threads(2)
    rows, folds = load_rows(), read_json(HERE / "folds.json")
    if [r["image"] for r in rows] != folds["ids"]:
        raise ValueError("Film ordering differs from fitted study")
    images = load_images(rows)
    plans = read_plans(folds)
    per_encoder = defaultdict(list)
    for plan in plans:
        per_encoder[plan.winner].append(plan)
    records, timings = [], {}
    for tag, heads in per_encoder.items():
        start = perf_counter()
        model = encoder_for(tag, "cuda")
        records.extend(score_variants(model, images, rows, heads, batch_size))
        torch.cuda.synchronize()
        timings[tag] = round(perf_counter() - start, 3)
        print(tag, timings[tag], "seconds", flush=True)
        del model
        torch.cuda.empty_cache()
    if len(records) != 2 * len(rows) * len(VARIANTS):
        raise ValueError("Unexpected number of scored variants")
    audit = crosscheck(records, rows, plans)
    report = dict(variants=list(VARIANTS), device=torch.cuda.get_device_name(), batch_size=batch_size,
                  seconds_including_encoder_load=timings, original_crosscheck=audit, summaries={})
    for group in ("refit84", "oof42"):
        for cohort in ("all84","original59"):
            subset = [r for r in records if r["group"] == group and (cohort == "all84" or r["old_split"] == "train")]
            report["summaries"][f"{group}_{cohort}"] = summarize(subset)
    report["encoder_per_head"] = {f"{p.group}_fold{p.fold}": p.winner for p in plans}
    details(HERE / "selected-perturbations.csv", records)
    write_json(HERE / "selected-stability.json", report)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or not torch.cuda.is_available():
        parser.error("Positive batch size and CUDA required")
    run(args.batch_size)
