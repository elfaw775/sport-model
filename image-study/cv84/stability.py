"""Measure repeatability on fitted films and perturbation robustness on held-out films."""
import argparse
import csv
from io import BytesIO
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageEnhance

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "training/scripts"))
from study_paths import CV, PUBLISHED, YOLO_WEIGHTS
from evaluate import DATA, RUN, fit_probe, probe_scores, read_json, sha, write_json
from detection_folds import LOCAL, balanced_cutoff
from benchmark_rfdetr import YOLO_SHA256, predictor, to_records
from compare_detectors import counts, truth

HEAD = PUBLISHED / "holdout/winning-head.npz"
LOCK = PUBLISHED / "holdout/decisions.json"

VARIANTS = ("original", "repeat", "brightness90", "brightness110", "contrast90", "contrast110", "jpeg85")


def adjust(im, name):
    if name in ("original", "repeat"):
        return im
    if name.startswith("brightness"):
        return ImageEnhance.Brightness(im).enhance(int(name[10:])/100)
    if name.startswith("contrast"):
        return ImageEnhance.Contrast(im).enhance(int(name[8:])/100)
    if name == "jpeg85":
        with BytesIO() as data:
            im.save(data, "JPEG", quality=85)
            data.seek(0)
            with Image.open(data) as decoded:
                return decoded.convert("RGB")
    raise ValueError(name)


def details(path, rows):
    with path.open("w", newline="",encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(items):
    by_id = {}
    for row in items:
        by_id.setdefault(row["image"], []).append(row)
    def tally(rows):
        return dict(tp=sum(r["y"] == 1 and r["pred"] == 1 for r in rows),
                    tn=sum(r["y"] == 0 and r["pred"] == 0 for r in rows),
                    fp=sum(r["y"] == 0 and r["pred"] == 1 for r in rows),
                    fn=sum(r["y"] == 1 and r["pred"] == 0 for r in rows))
    assert all([r["variant"] for r in rows] == list(VARIANTS) for rows in by_id.values())
    report = dict(by_variant={v:tally([r for r in items if r["variant"] == v]) for v in VARIANTS},
        all_variants_correct_positive=sum(rs[0]["y"] == 1 and all(r["pred"] == 1 for r in rs) for rs in by_id.values()),
        all_variants_correct_negative=sum(rs[0]["y"] == 0 and all(r["pred"] == 0 for r in rs) for rs in by_id.values()),
        changed_ids=[s for s,rs in by_id.items() if len({r["pred"] for r in rs}) > 1],
        identical_image_changed=sum(rs[0]["pred"] != rs[1]["pred"] for rs in by_id.values()),
        identical_image_max_score_change=max(abs(rs[0]["score"]-rs[1]["score"]) for rs in by_id.values()))
    if "box_tp" in items[0]:
        report["positive_with_localized_box_all_variants"] = sum(all(r["box_tp"]>0 for r in rs) for rs in by_id.values())
        report["by_variant_localization"] = {v:{k:sum(r[k] for r in items if r["variant"] == v)
                                                 for k in ("box_tp","box_fp","box_fn")} for v in VARIANTS}
    return report


def run_embed(ids, y, images):
    from ultralytics import YOLO
    from importlib.metadata import version
    weights = YOLO_WEIGHTS
    assert sha(weights) == YOLO_SHA256
    model = YOLO(str(weights),task="detect")
    end = len(model.model.yaml["backbone"])-1
    assert end == 10
    vectors = []
    for i in range(0,len(ids),4):
        chunk = model.predict(images[i:i+4], embed=[end], imgsz=640, rect=False,device=0,verbose=False)
        vectors.extend(t.detach().cpu().float().numpy().reshape(-1) for t in chunk)
    X = np.stack(vectors)
    assert X.shape[0] == 84 and np.isfinite(X).all()
    dest = CV / "features/yolo11s-coco.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, ids=ids, y=y, X=X)
    write_json(dest.with_suffix(".json"),dict(weight_sha256=sha(weights), file_sha256=sha(dest), layer=end,
          layer_type=type(model.model.model[end]).__name__, shape=list(X.shape), ultralytics=version("ultralytics"),
          source="https://github.com/ultralytics/assets/releases", resizing="640 square, Ultralytics letterbox",
          pretraining="COCO; no equine fine-tuning", task="average-pooled backbone with independently fitted probe"))


def run_dino(foldmap,ids,y,images):
    from hf_features import FrozenEncoder
    enc = FrozenEncoder("dinov2", device="cuda")
    vectors = {}
    for name in VARIANTS:
        vectors[name] = np.vstack([enc.encode([adjust(im,name) for im in images[i:i+6]]) for i in range(0,len(ids),6)])
    decision = read_json(LOCK)
    assert decision["primary_head_sha256" if "primary_head_sha256" in decision else "head_sha256"] == sha(HEAD)
    cutoff = decision["validation"]["dinov2"]["threshold"]
    fitted = np.flatnonzero(np.array(foldmap["historical_splits"]) == "train")
    def rows_for(indices, head, threshold, label, fold):
        rows=[]
        for name, X in vectors.items():
            scores = probe_scores(head,X[indices])
            rows.extend(dict(image=str(ids[i]),old_split=foldmap["historical_splits"][i],y=int(y[i]),
                model=label,fold=fold,variant=name,score=float(score),threshold=threshold,pred=int(score>=threshold))
                for i,score in zip(indices,scores,strict=True))
        return rows
    with np.load(HEAD) as head:
        rows=rows_for(fitted,head,cutoff,"old_dinov2", -1)
    oof = []
    X = vectors["original"].astype(np.float64)
    for fold in foldmap["repeats"]["42"]:
        fit = np.array(fold["train"])
        scores = np.full(len(y), np.nan)
        for inn in fold["inner_validation"]:
            tr = np.setdiff1d(fit, inn)
            scores[inn] = probe_scores(fit_probe(X[tr],y[tr]),X[inn])
        t = balanced_cutoff(y[fit], scores[fit])
        oof.extend(rows_for(fold["test"],fit_probe(X[fit],y[fit]),t,"oof_dinov2",fold["fold"]))
    details(CV / "dino-perturbations.csv",rows+oof)
    write_json(CV / "dino-stability.json", dict(fitted_original59=summarize(rows),oof_all84=summarize(oof),
               oof_original59=summarize([r for r in oof if r["old_split"]=="train"])))


def run_detector(foldmap,ids,y,images,paths):
    weights = RUN / "yolo11s/weights/best.pt"
    image_threshold = read_json(LOCK)["validation"]["retrained_yolo11s"]["threshold"]
    detect = predictor(weights,"yolo")
    targets = truth("train",RUN / "dataset")
    rows = []
    for i in np.flatnonzero(np.array(foldmap["historical_splits"]) == "train"):
        name = paths[ids[i]].name
        for var in VARIANTS:
            im = adjust(images[i],var)
            boxes = to_records(*detect(im),*im.size)
            score = max((r["score"] for r in boxes),default=0.)
            c = counts({name:boxes}, {name:targets[name]}, .15)
            rows.append(dict(image=str(ids[i]),y=int(y[i]),variant=var,score=score,threshold=image_threshold,
                      pred=int(score>=image_threshold),box_tp=c["tp"],box_fp=c["fp"],box_fn=c["fn"]))
    details(CV / "detector-fitted-perturbations.csv",rows)
    write_json(CV / "detector-fitted-stability.json",dict(weight_sha256=sha(weights), image_threshold=image_threshold,
               localization_threshold=.15,iou=.5,**summarize(rows)))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages",nargs="+",choices=["embed","dino","detector"])
    stages=parser.parse_args().stages
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", YOLO_CONFIG_DIR=str(LOCAL / "ultralytics"))
    import torch
    torch.set_num_threads(4)
    folds=read_json(CV / "folds.json")
    ids=np.array(folds["ids"]); y=np.array(folds["labels"])
    paths={p.stem:p for p in (DATA / "images").glob("*/*") if p.is_file()}
    assert set(ids) == set(paths)
    films=[]
    for key in ids:
        with Image.open(paths[key]) as im:
            films.append(im.convert("RGB"))
    if "embed" in stages:
        run_embed(ids,y,films)
    if "dino" in stages:
        run_dino(folds,ids,y,films)
    if "detector" in stages:
        run_detector(folds,ids,y,films,paths)
