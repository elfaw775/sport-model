"""Prespecified filename-prefix sensitivity check, not verified horse grouping.

Group equal leading decimal integers (including leading-zero equivalents), give
non-matching films their own group. Fixed DINOv2-S and height, grouped outer 5
and inner 4 folds, shuffled seed 42 at both levels. Pick thresholds only from
pooled inner OOF predictions. This does not change the primary CV84 experiment.
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from nested import (HERE, MANIFEST, features, fit_and_score, heights, load_rows,
                    metrics, midpoint_threshold, sha, write_json, write_rows)


def group_key(image_id):
    match = re.match(r"^([0-9]+)", image_id)
    return f"number:{int(match[1])}" if match else f"singleton:{image_id}"


def splits(indices, y, group, k):
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    for a, b in splitter.split(indices, y[indices], group[indices]):
        train, held = indices[a], indices[b]
        if set(group[train]) & set(group[held]):
            raise ValueError("Group split across fit/held rows")
        if len(np.unique(y[train])) != 2:
            raise ValueError("A fit fold lacks a class")
        yield train, held


def balance(idx, y, groups):
    return {"n": len(idx), "pos": int(y[idx].sum()), "neg": int((y[idx]==0).sum()),
            "groups":len(np.unique(groups[idx]))}


def run():
    rows = load_rows()
    ids = [r["image"] for r in rows]
    y = np.array([int(r["category"]=="pos") for r in rows])
    group = np.array([group_key(key) for key in ids])
    unique = np.unique(group)
    X, hashes = features("dinov2", ids, y)
    X = {"dinov2":X, "height":heights(rows)}
    counts = Counter(group)
    aggregation = {
        "definition": __doc__, "source_sha256": sha(Path(__file__)), "manifest_sha256":sha(MANIFEST),
        "feature_sha256":hashes, "n_films":len(ids), "n_groups":len(unique),
        "positive_films":int(y.sum()), "negative_films":int(sum(y==0)),
        "singleton_no_numeric_prefix":[key for key,g in zip(ids,group) if g.startswith("singleton:")],
        "repeated_groups":{k:n for k,n in counts.items() if n>1},
        "pos_only_groups":sum(set(y[group==g])=={1} for g in unique),
        "neg_only_groups":sum(set(y[group==g])=={0} for g in unique),
        "mixed_groups":[g for g in unique if np.unique(y[group==g]).size>1],
        "historical_split_overlap_groups":[g for g in unique if len({r['split'] for r, rg in zip(rows, group) if rg==g})>1],
        "folds":[],
    }
    all_i = np.arange(len(ids))
    coverage = np.zeros(len(ids),dtype=int)
    scores = {method:np.zeros(len(ids)) for method in X}
    preds = {method:np.zeros(len(ids),dtype=int) for method in X}
    records = []
    for fold, (outer_train,outer_held) in enumerate(splits(all_i,y,group,5)):
        if len(np.unique(y[outer_held])) != 2:
            raise ValueError("An outer held fold lacks a class")
        coverage[outer_held] += 1
        record = dict(fold=fold, train=balance(outer_train,y,group),
                      held=balance(outer_held,y,group),inner_validation=[],thresholds={})
        inner_scores = {name:np.full(len(y), np.nan) for name in X}
        seen = np.zeros(len(y),dtype=int)
        for tr,held in splits(outer_train,y,group,4):
            seen[held] += 1
            record["inner_validation"].append(dict(test=held.tolist(),counts=balance(held,y,group)))
            for method in X:
                inner_scores[method][held] = fit_and_score(method,X,y,tr,held)
        if not np.all(seen[outer_train]==1) or np.any(seen[outer_held]):
            raise ValueError("Invalid inner coverage")
        for method in X:
            threshold = midpoint_threshold(y[outer_train], inner_scores[method][outer_train])
            record["thresholds"][method] = threshold
            score = fit_and_score(method, X, y, outer_train, outer_held)
            pred = score >= threshold
            scores[method][outer_held],preds[method][outer_held] = score,pred
            for j,i in enumerate(outer_held):
                records.append(dict(method=method,fold=fold,id=ids[i],prefix_group=group[i],y=int(y[i]),
                                    score=float(score[j]),threshold=threshold,pred=int(pred[j])))
        aggregation["folds"].append(record)
    if not np.all(coverage==1):
        raise ValueError("Incomplete outer coverage")
    aggregation["pooled"] = {name:metrics(y,scores[name],preds[name]) for name in X}
    aggregation["interval_note"] = "Exact binomial intervals from the shared metric code treat films as independent; shared prefixes may violate that assumption."
    write_rows(HERE/"prefix-group-predictions.csv", records)
    write_json(HERE/"prefix-groups.json", aggregation)
    for name, m in aggregation["pooled"].items():
        print(name, {k:m[k] for k in ("tp","tn","fp","fn","accuracy","balanced_accuracy","roc_auc")})
    print({k:aggregation[k] for k in ("n_groups","singleton_no_numeric_prefix","repeated_groups","pos_only_groups","neg_only_groups","mixed_groups","historical_split_overlap_groups")})


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
