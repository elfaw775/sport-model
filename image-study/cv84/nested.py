"""Evaluate on repeated out-of-fold films; isolate thresholds and encoder selection inside train folds."""
import csv
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from folds import HERE, MANIFEST, SEEDS, build, load_rows, split_indices
from study_paths import PACKAGE
from evaluate import exact_interval, fit_probe, probe_scores, sha, write_json


HF = ("dinov2", "rad-dino", "resnet18", "bone-siglip", "biomedclip",
      "dinov2reg-base", "convnextv2-tiny")
YOLO = "yolo11s-coco"
CONTROLS = ("height", "all_positive")
PRIMARY = "nested_hf"
OUT = HERE / "results"


def features(name, ids, labels):
    directory = HERE / "features"
    path = directory / f"{name}.npz"
    if path.exists():
        paths = [path]
    else:
        paths = []
        for split in ("train", "val", "test"):
            local = directory / f"{name}-{split}.npz"
            paths.append(local if local.exists() else HERE.parent / "features" / local.name)
    observed_ids, y, blocks = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            new_ids = data["ids"].tolist()
            new_y, new_x = data["y"], data["X"]
            if len(new_ids) != len(new_x) or new_y.shape != (len(new_x),) or new_x.ndim != 2:
                raise ValueError(f"Inconsistent array shape in {path}")
            observed_ids.extend(new_ids)
            y.extend(new_y.tolist())
            blocks.append(new_x.astype(np.float64))
    if len(ids) != len(observed_ids) or set(ids) != set(observed_ids):
        raise ValueError(f"Duplicate or missing films for {name}")
    position = {image_id:i for i, image_id in enumerate(observed_ids)}
    order = [position[key] for key in ids]
    X = np.concatenate(blocks)[order]
    if not np.array_equal(np.asarray(y)[order], labels) or not np.isfinite(X).all():
        raise ValueError(f"Label mismatch or non-finite feature in {name}")
    return X, {str(p.relative_to(HERE.parent)):sha(p) for p in paths}


def heights(rows):
    images = MANIFEST.parent / "images"
    values = []
    for split in ("train", "val", "test"):
        if not (images / split).is_dir():
            raise FileNotFoundError(images / split)
    indexed = {(s, p.stem):p for s in ("train","val","test") for p in (images/s).iterdir() if p.is_file()}
    for row in rows:
        with Image.open(indexed[(row["split"], row["image"])]) as im:
            values.append(im.height)
    return np.asarray(values, dtype=float)


def midpoint_threshold(y, score):
    levels = np.unique(score)
    mids = levels[:-1] + (levels[1:] - levels[:-1])/2
    # Adjacent IEEE floats have no representable midpoint; the upper endpoint realizes the partition.
    mids = np.where(mids <= levels[:-1], levels[1:], mids)
    options = np.r_[np.nextafter(levels[0], -np.inf), mids, np.nextafter(levels[-1], np.inf)]
    n_pos, n_neg = int(sum(y == 1)), int(sum(y == 0))
    if min(n_pos, n_neg) == 0 or len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError("Threshold selection requires both classes and aligned finite scores")
    def rank(t):
        pred = score >= t
        tp = int(sum(pred & (y == 1)))
        tn = int(sum(~pred & (y == 0)))
        return tp*n_neg + tn*n_pos, tp, tn, float(t)
    return float(max(options, key=rank))


def rank_model(models, summaries):
    return max(models, key=lambda key:(summaries[key]["auc_pair_credit2"], Fraction(summaries[key]["ap_exact"])))


def exact_ranking(y, s):
    # Work with integer pair credits and exact grouped precision so numerical integration
    # round-off cannot choose between encoders that produced equivalent rankings.
    positive = s[y == 1][:,None]
    negative = s[y == 0][None,:]
    credit = int(2*np.count_nonzero(positive > negative) + np.count_nonzero(positive == negative))
    ap = Fraction(0)
    tp = n = 0
    for level in np.unique(s)[::-1]:
        batch = y[s == level]
        new_tp = int(np.count_nonzero(batch))
        tp += new_tp
        n += len(batch)
        ap += Fraction(new_tp*tp,n)
    ap /= int(sum(y))
    return dict(auc=credit/(2*positive.size*negative.size),ap=float(ap),
                auc_pair_credit2=credit,ap_exact=str(ap))


def fit_and_score(name, X, y, train, held):
    if name == "all_positive":
        return np.ones(len(held))
    if name == "height":
        reference = np.median(X[name][train][y[train] == 0])
        return np.abs(X[name][held] - reference)
    head = fit_probe(X[name][train], y[train])
    return probe_scores(head, X[name][held])


def inner_loop(X, y, outer_train, inner_held):
    # Held indices are global manifest indices, so each image can be checked directly against the public fold list.
    summaries = {}
    scores = {name:np.full(len(y), np.nan) for name in X}
    for held in inner_held:
        train = np.setdiff1d(outer_train, held)
        for name in X:
            scores[name][held] = fit_and_score(name, X, y, train, held)
    for name, values in scores.items():
        s = values[outer_train]
        truth = y[outer_train]
        if not np.all(np.isfinite(s)):
            raise ValueError(f"Unscored inner film: {name}")
        summaries[name] = exact_ranking(truth,s)
        summaries[name]["threshold"] = 1. if name == "all_positive" else midpoint_threshold(truth, s)
    return rank_model(HF, summaries), summaries, {name:v[outer_train].tolist() for name,v in scores.items()}


def metrics(y, score, pred):
    tp, tn = int(sum((pred == 1) & (y == 1))), int(sum((pred == 0) & (y == 0)))
    n = len(y)
    pos, neg = int(sum(y)), int(n-sum(y))
    fp, fn = neg-tn, pos-tp
    result = dict(n=n, tp=tp, fp=fp, tn=tn, fn=fn, balanced_accuracy=(tp/pos + tn/neg)/2,
                  roc_auc=float(roc_auc_score(y, score)), average_precision=float(average_precision_score(y, score)))
    for key, yes, denominator in (("accuracy",tp+tn,n), ("sensitivity",tp,pos),
                                  ("specificity",tn,neg), ("precision",tp,tp+fp)):
        result[key] = yes/denominator if denominator else None
        result[key+"_ci95"] = exact_interval(yes,denominator)
    return result


def paired(y, a, b):
    won, lost = int(sum((a == y) & (b != y))), int(sum((a != y) & (b == y)))
    p = float(binomtest(won, won+lost, .5).pvalue) if won + lost else 1.
    return dict(primary_only_correct=won, comparator_only_correct=lost, exact_mcnemar_p=p)


def repeat(seed, folds, X, y, ids, original):
    names = (*X, PRIMARY)
    scores = {n:np.full(len(y), np.nan) for n in names}
    pred = {n:np.full(len(y), -1, dtype=int) for n in names}
    outer_predictions, audit = [], []
    coverage = np.zeros(len(y), dtype=int)
    for fold in folds:
        train, held = np.array(fold["train"]), np.array(fold["test"])
        inner = [np.array(block) for block in fold["inner_validation"]]
        assert not set(train) & set(held) and np.array_equal(np.sort(np.concatenate(inner)), np.sort(train))
        coverage[held] += 1
        winner, summaries, inner_scores = inner_loop(X,y,train,inner)
        print(f"seed {seed} fold {fold['fold']} -> {winner}", flush=True)
        saved_head_hash = None
        for name in X:
            if seed == SEEDS[0] and name == winner:
                head = fit_probe(X[name][train],y[train])
                head_path = OUT / f"seed42-fold{fold['fold']}-head.npz"
                np.savez(head_path, **head)
                saved_head_hash = sha(head_path)
                scores[name][held] = probe_scores(head, X[name][held])
            else:
                scores[name][held] = fit_and_score(name, X, y, train, held)
            threshold = summaries[name]["threshold"]
            pred[name][held] = scores[name][held] >= threshold
        scores[PRIMARY][held] = scores[winner][held]
        pred[PRIMARY][held] = pred[winner][held]
        audit.append(dict(fold=fold["fold"], winner=winner, summaries=summaries,
                          head_sha256=saved_head_hash, outer_test_indices=held.tolist(),
                          oof_train_indices=train.tolist(), inner_oof_scores=inner_scores))
        for name in names:
            method = winner if name == PRIMARY else name
            for i in held:
                outer_predictions.append(dict(seed=seed, fold=fold["fold"], index=int(i), id=ids[i],
                    original_split=original[i], y=int(y[i]), method=name, encoder=method,
                    score=float(scores[name][i]), threshold=summaries[method]["threshold"], pred=int(pred[name][i])))
    if np.any(coverage != 1):
        raise ValueError(f"Invalid held-out coverage for seed {seed}")
    groups = {"all84":np.ones(len(y),dtype=bool), "original59":original == "train",
              "original13":original == "val", "original12":original == "test"}
    by_group = {k:{name:metrics(y[m],scores[name][m],pred[name][m]) for name in names} for k,m in groups.items()}
    comparisons = {name:paired(y,pred[PRIMARY],pred[name]) for name in (YOLO, "height", "all_positive")}
    return (dict(groups=by_group, paired=comparisons, selections=dict(Counter(row["winner"] for row in audit))),
            audit, outer_predictions)


def stability(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], r["index"]), []).append(r)
    records, totals = [], {}
    for (method,index), group in groups.items():
        group.sort(key=lambda x:SEEDS.index(x["seed"]))
        if tuple(r["seed"] for r in group) != SEEDS:
            raise ValueError(f"Repeated prediction count mismatch: {method}, {index}")
        p = [r["pred"] for r in group]
        agreed = len(set(p)) == 1
        rec = dict(method=method,id=group[0]["id"], historical_split=group[0]["original_split"],
                   y=group[0]["y"], agree=int(agreed),positive_calls=sum(p))
        for r in group:
            rec[f"pred_{r['seed']}"] = r["pred"]
        records.append(rec)
        tab = totals.setdefault(method, dict(n=0, agree=0, always_correct=0, always_wrong=0, flipped=0))
        tab["n"] += 1
        tab["agree"] += int(agreed)
        tab["always_correct"] += int(agreed and p[0] == rec["y"])
        tab["always_wrong"] += int(agreed and p[0] != rec["y"])
        tab["flipped"] += int(not agreed)
    return totals, records


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    if (OUT / "summary.json").exists():
        raise FileExistsError("Completed experiment already exists")
    folds = json.loads((HERE/"folds.json").read_text(encoding="utf8"))
    if folds != build():
        raise ValueError("Saved folds differ from preregistered generator")
    ids, y = folds["ids"], np.array(folds["labels"])
    original = np.array(folds["historical_splits"])
    X, hashes = {}, {}
    for model in (*HF, YOLO):
        X[model], hashes[model] = features(model, ids, y)
    X["height"] = heights(load_rows())
    X["all_positive"] = np.ones(len(y))
    OUT.mkdir(exist_ok=True)
    summary = dict(folds_sha256=sha(HERE/"folds.json"), source_sha256=sha(Path(__file__)),
                   plan_sha256=sha(PACKAGE/"cv84/plan.md"), source_feature_sha256=hashes, repeats={})
    results, decisions = [], {}
    for seed in SEEDS:
        report, decisions[seed], rows = repeat(seed,folds["repeats"][str(seed)],X,y,ids,original)
        summary["repeats"][seed] = report
        results.extend(rows)
    write_rows(OUT/"outer_predictions.csv", results)
    summary["stability"], stability_rows = stability(results)
    write_rows(OUT/"per_film_stability.csv", stability_rows)
    write_json(OUT/"inner_decisions.json", decisions)
    # Select a final head on OOF internal scores; do not use the outer test outcomes to pick an encoder.
    all_idx = np.arange(len(y))
    inner = [held for _,held in split_indices(y,4,1042)]
    winner, report, scores = inner_loop(X,y,all_idx,inner)
    head_path = OUT / "refit84-head.npz"
    np.savez(head_path, **fit_probe(X[winner],y))
    summary["refit84"] = dict(winner=winner, threshold=report[winner]["threshold"],
        inner_summary=report, head_sha256=sha(head_path), inner_validation=[v.tolist() for v in inner])
    write_json(OUT/"refit84-selection-scores.json", dict(ids=ids,scores=scores))
    write_json(OUT/"summary.json", summary)
    print(f"Saved {len(results)} out-of-fold records. Refit84 selects {winner}.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
