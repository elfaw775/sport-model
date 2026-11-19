"""Train five independent lesion detectors, score unseen films, and retain per-film locations."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "training/scripts"))
from benchmark_rfdetr import YOLO_SHA256, coco_score, file_hash, fit_yolo, predictor, to_records
from compare_detectors import counts, truth
from prepare_comparison import prepare
from evaluate import DATA, RUN, write_json

from study_paths import CACHE as LOCAL, CV, YOLO_WEIGHTS


def balanced_cutoff(y, scores):
    values = np.unique(scores)
    levels = [np.nextafter(values[0], -np.inf), *(values[:-1] + np.diff(values)/2),
              np.nextafter(values[-1], np.inf)]
    def rank(t):
        pred = scores >= t
        tp = int(sum(pred & (y == 1)))
        tn = int(sum(~pred & (y == 0)))
        return tp * int(sum(y == 0)) + tn * int(sum(y == 1)), tp, tn, float(t)
    return float(max(levels, key=rank))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=CV / "detector-work")
    ap.add_argument("--labels", type=Path, help="Prepared, rotation-corrected YOLO dataset")
    args = ap.parse_args()
    if args.labels is None:
        args.labels = RUN / "dataset"
        if not (args.labels / "data.yaml").exists():
            args.labels = prepare(RUN)
    if not (args.labels / "data.yaml").is_file():
        ap.error("--labels must point to the output dataset from training/scripts/prepare_comparison.py")
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", WANDB_MODE="disabled",
                      YOLO_CONFIG_DIR=str(LOCAL / "ultralytics"), MPLCONFIGDIR=str(LOCAL / "matplotlib"))
    import torch
    torch.set_num_threads(4)
    foldmap = json.loads((CV / "folds.json").read_text(encoding="utf8"))
    folds, ids = foldmap["repeats"]["42"], foldmap["ids"]
    y = np.array(foldmap["labels"])
    official = YOLO_WEIGHTS
    assert file_hash(official) == YOLO_SHA256
    img = {p.stem:p for p in (DATA / "images").glob("*/*") if p.is_file()}
    lbl = {p.stem:p for p in (args.labels / "labels").glob("*/*.txt")}
    assert set(ids) == set(img) == set(lbl)
    true = {}
    for split in ("train", "val", "test"):
        true.update(truth(split, args.labels))
    reports, predictions, sizes = [], {}, {}
    for f in folds:
        num = f["fold"]
        folder = args.work / f"fold-{num}"
        folder.mkdir(parents=True, exist_ok=True)
        result = folder / "held-output.json"
        if not result.exists():
            ds = folder / "dataset"
            cal = f["inner_validation"][0]
            train = sorted(set(f["train"])-set(cal))
            for split, indices in (("train",train), ("val",cal)):
                for kind in ("images", "labels"):
                    target = ds / kind / split
                    target.mkdir(parents=True, exist_ok=True)
                    for i in indices:
                        src = (img if kind == "images" else lbl)[ids[i]]
                        shutil.copyfile(src, target / src.name)
            (ds / "data.yaml").write_text("path: " + json.dumps(str(ds.resolve()), ensure_ascii=False)
                + "\ntrain: images/train\nval: images/val\nnames: {0: lesion}\n",encoding="utf8")
            print(f"Starting outer fold {num}, fit {len(train)}, calibrate {len(cal)}, held {len(f['test'])}",flush=True)
            best = fit_yolo(official, ds, SimpleNamespace(epochs=32,seed=42,work=folder,yolo_batch=4))
            detect = predictor(best, "yolo")
            boxes, dims = {}, {}
            for i in [*cal,*f["test"]]:
                path = img[ids[i]]
                with Image.open(path) as im:
                    rgb = im.convert("RGB")
                boxes[path.name] = to_records(*detect(rgb), *rgb.size)
                dims[path.name] = rgb.size
            def score(i):
                return max((box["score"] for box in boxes[img[ids[i]].name]),default=0.)
            cutoff = balanced_cutoff(y[cal], np.array([score(i) for i in cal]))
            held = [img[ids[i]].name for i in f["test"]]
            write_json(result, dict(fold=num, fit_indices=train, calibration_indices=cal,
                     test_indices=f["test"], checkpoint_sha256=file_hash(best), weight=best.resolve().relative_to(args.work.resolve()).as_posix(),
                     threshold=cutoff, scores=[score(i) for i in f["test"]],
                     boxes={k:boxes[k] for k in held}, sizes={k:dims[k] for k in held}))
            del detect
            torch.cuda.empty_cache()
        record = json.loads(result.read_text(encoding="utf8"))
        assert record["test_indices"] == f["test"]
        predictions.update(record.pop("boxes"))
        sizes.update(record.pop("sizes"))
        reports.append(record)
        print("Finished fold",num,flush=True)
    assert set(true) == set(predictions) == set(sizes) and len(true) == 84
    rows = []
    for r in reports:
        for i, score in zip(r["test_indices"],r["scores"],strict=True):
            k = img[ids[i]].name
            loc = counts({k:predictions[k]}, {k:true[k]}, .15)
            rows.append(dict(image=ids[i], old_split=foldmap["historical_splits"][i], y=int(y[i]),
                        fold=r["fold"], score=score, threshold=r["threshold"], pred=int(score>=r["threshold"]),
                        box_tp=loc["tp"],box_fp=loc["fp"],box_fn=loc["fn"]))
    with (CV / "detector-oof.csv").open("w",newline="",encoding="utf-8-sig") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(CV / "detector-report.json",dict(folds=reports, localization_at_015=counts(predictions,true,.15),
          location_hit_films=sum(r["box_tp"]>0 for r in rows), coco=coco_score(true,predictions,sizes),
          image_tp=sum(r["pred"]*r["y"] for r in rows), image_tn=sum((1-r["pred"])*(1-r["y"]) for r in rows)))


if __name__ == "__main__":
    main()
