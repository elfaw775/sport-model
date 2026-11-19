"""Paired lesion/sham box blur on positive films using their seed-42 held-out decision rules."""
import csv
from collections import Counter

import numpy as np
import torch
from PIL import Image

from selected_stability import HERE, MANIFEST, OUT, PRIMARY, encoder_for, load_rows, probe_scores, read_plans
from evaluate import read_json, write_json
from validation_audit import CORRECTED, blurred, matched_background, read_boxes


def run():
    torch.set_num_threads(2)
    rows = load_rows()
    folds = read_json(HERE / "folds.json")
    plans = [p for p in read_plans(folds) if p.group == "oof42"]
    paths = {(p.parent.name,p.stem):p for p in (MANIFEST.parent/"images").glob("*/*") if p.is_file()}
    with (OUT/"outer_predictions.csv").open(encoding="utf-8-sig", newline="") as f:
        prior = {r["id"]:r for r in csv.DictReader(f) if r["method"] == PRIMARY and r["seed"] == "42"}
    cases, skipped = [], []
    for model in sorted({p.winner for p in plans}):
        encoder = encoder_for(model, "cuda")
        for plan in (p for p in plans if p.winner == model):
            for i in plan.indices:
                r = rows[i]
                if r["category"] != "pos":
                    continue
                stem, split = r["image"],r["split"]
                with Image.open(paths[split,stem]) as src:
                    if src.getexif().get(274,1) != 1:
                        raise ValueError(f"Unexpected orientation: {stem}")
                    im = src.convert("L")
                b = read_boxes(CORRECTED/"labels"/split/(stem+".txt"),*im.size)
                if len(b) != int(r["class0_boxes"]) or any(x1<=x0 or y1<=y0 for x0,y0,x1,y1 in b):
                    raise ValueError(f"Invalid lesion annotations: {stem}")
                sham = matched_background(np.asarray(im),b)
                if sham is None:
                    skipped.append(dict(id=stem,fold=plan.fold,reason="No disjoint intensity-matched same-size placement"))
                    continue
                variants = [im, blurred(im,b), blurred(im,sham)]
                raw, lesion, control = map(float,probe_scores(plan.params,encoder.encode(variants)))
                original = float(prior[stem]["score"])
                if abs(original-raw) > 5e-4 or (raw >= plan.threshold) != bool(int(prior[stem]["pred"])):
                    raise ValueError(f"Original image decision changed for {stem}")
                t = plan.threshold
                cases.append(dict(id=stem, original_split=split, fold=plan.fold, encoder=model, threshold=t,
                    image_size=list(im.size), lesion_boxes=b, sham_boxes=sham,
                    original=original, lesion_blur=lesion, sham_blur=control, original_fresh=raw,
                    original_pos=original>=t, lesion_pos=lesion>=t, sham_pos=control>=t))
        del encoder
        torch.cuda.empty_cache()
    if len({r["id"] for r in cases+skipped}) != 68:
        raise ValueError("Unexpected number of positive films")
    x = np.array([[r["original"],r["lesion_blur"],r["sham_blur"]] for r in cases])
    report = dict(seed=42, group="nested_hf OOF positive films", eligible=len(cases), skipped=skipped,
        positive_counts={k:sum(r[k] for r in cases) for k in ("original_pos","lesion_pos","sham_pos")},
        lesion_minus_original_median=float(np.median(x[:,1]-x[:,0])),
        sham_minus_original_median=float(np.median(x[:,2]-x[:,0])),
        lesion_minus_sham_median=float(np.median(x[:,1]-x[:,2])),
        lesion_lower_than_sham=int(np.sum(x[:,1]<x[:,2])),
        max_original_score_abs_delta=max(abs(r["original"]-r["original_fresh"]) for r in cases),
        cross_tab_original_lesion_sham=dict(Counter("".join(str(int(r[k])) for k in
            ("original_pos","lesion_pos","sham_pos")) for r in cases)), flips={})
    for variant in ("lesion_pos","sham_pos"):
        for source in (True, False):
            report["flips"][f"{variant}_{int(source)}_to_{int(not source)}"] = [
                r["id"] for r in cases if r["original_pos"] == source and r[variant] != source]
    write_json(HERE/"oof_occlusion.json", {**report,"rows":cases})
    print({k:v for k,v in report.items() if k not in {"flips","skipped"}})


if __name__ == "__main__":
    run()
