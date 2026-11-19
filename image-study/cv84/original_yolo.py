"""Refit the archived YOLO11s recipe on the same five outer folds as the corrected detector."""
import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from study_paths import CACHE, DATA, PUBLISHED, YOLO_WEIGHTS
from detection_folds import balanced_cutoff
from benchmark_rfdetr import YOLO_SHA256, file_hash, predictor, to_records
from evaluate import write_json
from nested import metrics, write_rows
from with_film_data import COMMIT as DATA_COMMIT

CONFIG = PUBLISHED / "cv84"
FIXED = 0.15
INPUT_HASHES = {"images":"a30abcbd60656d28110d024beb80a90a19f98a6bdc27a125bc1508ba37d2acba",
                "labels":"11279d98202c5278741158c73684cc3d47ffd21c2e0b5a560ec942c97e037e2a"}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def input_hash(files):
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p:p.relative_to(DATA).as_posix()):
        digest.update((path.relative_to(DATA).as_posix() + "\0" + file_hash(path) + "\n").encode())
    return digest.hexdigest()


def training_data(root, ids, images, labels, groups):
    for split, indices in groups.items():
        for kind, files in (("images", images), ("labels", labels)):
            dest = root / kind / split
            dest.mkdir(parents=True)
            for i in indices:
                src = files[ids[i]]
                shutil.copyfile(src, dest / src.name)
    config = root / "data.yaml"
    config.write_text("path: " + json.dumps(str(root), ensure_ascii=False) +
                      "\ntrain: images/train\nval: images/val\nnames: {0: lesion}\n", encoding="utf8")
    return config


def fit(folder, dataset, recipe):
    import torch
    from ultralytics import YOLO
    args = dict(recipe["train_args"])
    for key in ("save_dir", "model", "mode", "task"):
        args.pop(key, None)
    # 8.4.200 disables AMP on the original MPS device, regardless of the saved True request.
    args.update(data=str(dataset), device=0, project=str(folder), name="training",
                amp=False, exist_ok=True, plots=False)
    model = YOLO(str(YOLO_WEIGHTS), task="detect")
    begin = perf_counter()
    model.train(**args)
    elapsed = perf_counter() - begin
    best = folder / "best.pt"
    shutil.copyfile(model.trainer.best, best)
    epochs = model.trainer.csv.read_text(encoding="utf8").splitlines()
    with (folder / "results.csv").open("w", encoding="utf8") as output:
        output.write("\n".join(epochs) + "\n")
    shutil.copyfile(model.trainer.save_dir / "args.yaml", folder / "args.yaml")
    curve = list(csv.DictReader(epochs))
    bad = {col:[int(row["epoch"]) for row in curve if not np.isfinite(float(row[col]))]
           for col in curve[0] if col.startswith(("val/", "train/", "metrics/"))}
    state = YOLO(str(best), task="detect").model.state_dict()
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Selected checkpoint contains non-finite weights")
    if bool(model.trainer.amp) or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("Training used reduced precision")
    info = dict(elapsed_train_seconds=elapsed, completed_epochs=len(epochs) - 1,
                nonfinite_curve_entries={col:indices for col,indices in bad.items() if indices},
                best_state_all_finite=True, best_state_tensors=len(state),
                amp_enabled=bool(model.trainer.amp), batch=model.trainer.batch_size)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best, info


def evaluate_fold(folder, split, ids, y, images, labels, recipe, provenance):
    f = split["fold"]
    train, cal, test = [split[key] for key in ("fit_indices", "calibration_indices", "test_indices")]
    print(f"Starting fold {f}: fit {len(train)}, calibration {len(cal)}, held {len(test)}", flush=True)
    with TemporaryDirectory(prefix="equine-original-fold-") as tmp:
        dataset = training_data(Path(tmp), ids, images, labels, {"train":train, "val":cal})
        best, info = fit(folder, dataset, recipe)
    predict = predictor(best, "yolo")
    start = perf_counter()
    boxes, sizes = {}, {}
    def score(indices):
        scores = []
        for i in indices:
            image_id = ids[i]
            with Image.open(images[image_id]) as im:
                rgb = im.convert("RGB")
            raw = predict(rgb)
            if not all(np.isfinite(array).all() for array in raw):
                raise ValueError(f"Invalid detector output for film {image_id}")
            boxes[image_id] = to_records(*raw, *rgb.size)
            sizes[image_id] = rgb.size
            scores.append(max((box["score"] for box in boxes[image_id]), default=0.0))
        return scores
    cal_scores = score(cal)
    threshold = balanced_cutoff(y[cal], np.array(cal_scores))
    test_scores = score(test)
    fit_scores = score(train)
    out = dict(fold=f, fit_indices=train, calibration_indices=cal, test_indices=test,
               fit_scores=fit_scores, calibration_scores=cal_scores, scores=test_scores,
               threshold=threshold, fixed_threshold=FIXED, checkpoint_sha256=file_hash(best),
               weight=f"fold-{f}/best.pt", elapsed_inference_seconds=perf_counter()-start,
               **info, **provenance)
    write_json(folder / "boxes.json", dict(boxes=boxes, sizes=sizes, indices="keyed by original film ID"))
    write_json(folder / "held-output.json", out)
    # Originals and staged films are in temporary directories. Retain one verified checkpoint per fold.
    training = (folder / "training").resolve()
    if training.parent == folder.resolve() and training.is_dir():
        shutil.rmtree(training)
    return out


def summarize(rows, *, independent_film_counts=True):
    y = np.array([r["y"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    summary = {p:metrics(y,scores,np.array([r[p] for r in rows])) for p in ("pred", "pred_fixed_015")}
    if not independent_film_counts:
        return {rule:{k:v for k,v in result.items() if not k.endswith("_ci95")}
                for rule,result in summary.items()}
    return summary


def summarize_fit_stability(work):
    """Count consistent detections across models that actually fitted each film."""
    with (work / "original-yolo-fit.csv").open(encoding="utf-8-sig", newline="") as f:
        fitting = list(csv.DictReader(f))
    records = []
    for image in sorted({row["image"] for row in fitting}):
        seen = [row for row in fitting if row["image"] == image]
        targets = {(int(row["y"]), row["old_split"]) for row in seen}
        folds = [int(row["fold"]) for row in seen]
        if len(targets) != 1 or len(set(folds)) != len(folds):
            raise ValueError("Conflicting film labels or duplicate fit-prediction rows")
        y, old_split = targets.pop()
        record = dict(image=image, y=y, old_split=old_split, fit_folds=folds, evaluations=len(seen))
        for rule in ("pred", "pred_fixed_015"):
            correct = sum(int(row[rule]) == y for row in seen)
            if len(seen) == 1:
                state = "single_correct" if correct else "single_wrong"
            elif correct == len(seen):
                state = "always_correct"
            else:
                state = "always_wrong" if correct == 0 else "mixed"
            record[rule] = dict(correct=correct, state=state)
        records.append(record)
    def counts(rows):
        return {rule:dict(Counter(row[rule]["state"] for row in rows)) for rule in ("pred", "pred_fixed_015")}
    write_json(work / "original-yolo-fit-stability.json", dict(
        definition="Across models that included each film in fitting; one deterministic inference per fitted model.",
        covered_unique_films=len(records), single_evaluation_films=sum(row["evaluations"]==1 for row in records),
        pooled_by_film=counts(records), positive_by_film=counts([r for r in records if r["y"]]),
        negative_by_film=counts([r for r in records if not r["y"]]),
        historical_train_by_film=counts([r for r in records if r["old_split"]=="train"]), films=records))


def run(work):
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", WANDB_MODE="disabled",
                      YOLO_CONFIG_DIR=str(CACHE / "ultralytics"), MPLCONFIGDIR=str(CACHE / "matplotlib"))
    import torch
    import ultralytics
    from ultralytics import settings
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings.update({k:False for k in ("sync","hub","wandb","mlflow","comet_ml","clearml") if k in settings})
    recipe = read(CONFIG / "original-yolo-recipe.json")
    if ultralytics.__version__ != recipe["source_ultralytics_version"] or not torch.cuda.is_available():
        raise ValueError("Expected the archived Ultralytics version and a CUDA device")
    folds = read(CONFIG / "folds.json")
    split_report = read(CONFIG / "detector-report.json")
    ids, y = folds["ids"], np.array(folds["labels"])
    images = {p.stem:p for p in (DATA / "images").glob("*/*") if p.is_file()}
    labels = {p.stem:p for p in (DATA / "labels").glob("*/*.txt")}
    input_hashes = {"images":input_hash(images.values()), "labels":input_hash(labels.values())}
    if (len(ids) != 84 or set(ids) != set(images) or set(ids) != set(labels) or
        input_hashes != INPUT_HASHES or file_hash(DATA / "split_manifest.tsv") != folds["manifest_sha256"] or
        file_hash(YOLO_WEIGHTS) != YOLO_SHA256):
        raise ValueError("Published data, manifest, or COCO initialization mismatch")
    for i, key in enumerate(ids):
        if bool(labels[key].read_text(encoding="utf8").strip()) != bool(y[i]):
            raise ValueError(f"Film label disagrees with original box labels: {key}")
    source = dict(data_commit=DATA_COMMIT, source_checkpoint_sha256=recipe["source_checkpoint_sha256"],
                  initial_coco_sha256=YOLO_SHA256, original_recipe_sha256=file_hash(CONFIG / "original-yolo-recipe.json"),
                  split_report_sha256=file_hash(CONFIG / "detector-report.json"), manifest_sha256=folds["manifest_sha256"],
                  staged_input_hashes=input_hashes, cuda_matmul_tf32=False, cudnn_tf32=False,
                  threshold_definition="Historical fixed evaluation threshold 0.15; fold threshold is calibration-only.",
                  ultralytics_version=ultralytics.__version__, torch_version=str(torch.__version__),
                  cuda_version=torch.version.cuda, device=torch.cuda.get_device_name(0))
    records, rows, fitting = [], [], []
    for split, declared in zip(split_report["folds"], folds["repeats"]["42"], strict=True):
        num = split["fold"]
        t, c, h = (split[k] for k in ("fit_indices","calibration_indices","test_indices"))
        if (declared["fold"] != num or sorted(t+c) != sorted(declared["train"]) or
            c != declared["inner_validation"][0] or h != declared["test"] or sorted(t+c+h) != list(range(84))):
            raise ValueError(f"Incompatible published split in fold {num}")
        folder = work / f"fold-{num}"
        folder.mkdir(parents=True, exist_ok=True)
        result = folder / "held-output.json"
        if result.exists():
            r = read(result)
            if (any(r[k] != v for k,v in source.items()) or file_hash(folder / "best.pt") != r["checkpoint_sha256"] or
                    any(r[k] != split[k] for k in ("fold","fit_indices","calibration_indices","test_indices"))):
                raise ValueError(f"Cannot resume incompatible fold {num}")
        else:
            r = evaluate_fold(folder, split, ids, y, images, labels, recipe, source)
        records.append(r)
        for indices, scores, output in ((h,r["scores"],rows), (t,r["fit_scores"],fitting)):
            for i,s in zip(indices,scores,strict=True):
                output.append(dict(image=ids[i], fold=num, y=int(y[i]), score=s, threshold=r["threshold"],
                                   pred=int(s>=r["threshold"]), pred_fixed_015=int(s>=FIXED),
                                   old_split=folds["historical_splits"][i]))
        print(f"Finished fold {num}, {r['completed_epochs']} epochs in {r['elapsed_train_seconds']:.1f}s",flush=True)
    if Counter(r["image"] for r in rows) != Counter(ids):
        raise ValueError("Incomplete or repeated outer predictions")
    write_rows(work / "original-yolo-oof.csv", rows)
    write_rows(work / "original-yolo-fit.csv", fitting)
    summarize_fit_stability(work)
    write_json(work / "original-yolo-report.json", dict(folds=records, pooled=summarize(rows),
               per_fold={str(r["fold"]):summarize([x for x in rows if x["fold"]==r["fold"]]) for r in records},
               historical_splits={s:summarize([x for x in rows if x["old_split"]==s]) for s in ("train","val","test")},
               fit_pooled_repeated=summarize(fitting, independent_film_counts=False), provenance=source))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=CACHE / "original-yolo-cv-new")
    ap.add_argument("--summarize-fit", action="store_true", help="Recount stability from completed fit predictions")
    args = ap.parse_args()
    if args.summarize_fit:
        summarize_fit_stability(args.work.resolve())
    else:
        run(args.work.resolve())
