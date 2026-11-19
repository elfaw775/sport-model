"""Training fit vs held-fold performance on the 59 training images of the historical DINOv2 probe."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from nested import HERE, SEEDS, fit_probe, metrics, midpoint_threshold, probe_scores, sha, split_indices, write_json
from study_paths import PUBLISHED
from evaluate import cases, load_npz

HEAD = PUBLISHED / "holdout/winning-head.npz"
LOCK = PUBLISHED / "holdout/decisions.json"


def run():
    development = json.loads(LOCK.read_text(encoding="utf8"))
    ids, y = cases("train")
    X, _ = load_npz("dinov2", "train", "X")
    if development["winner"] != "dinov2" or sha(HEAD) != development["head_sha256"]:
        raise RuntimeError("Historical classifier identity mismatch")
    historical_cutoff = development["validation"]["dinov2"]["threshold"]
    with np.load(HEAD, allow_pickle=False) as head:
        scores = probe_scores(head, X)
    doc = dict(ids=ids, y=y.tolist(), historical_fit=dict(threshold=historical_cutoff,
        scores=scores.tolist(), metrics=metrics(y,scores,scores>=historical_cutoff)), repeats={})
    for seed in SEEDS:
        oof, predicted = np.empty(len(y)), np.full(len(y), -1)
        folds = []
        for fold, (train, held) in enumerate(split_indices(y,5,seed)):
            inner_scores = np.full(len(train),np.nan)
            for a,b in split_indices(y[train],4,seed+1000+fold):
                inner_scores[b] = probe_scores(fit_probe(X[train[a]], y[train[a]]),X[train[b]])
            t = midpoint_threshold(y[train],inner_scores)
            oof[held] = probe_scores(fit_probe(X[train],y[train]),X[held])
            predicted[held] = oof[held]>=t
            folds.append(dict(fold=fold,train=train.tolist(),held=held.tolist(),threshold=t))
        assert np.all(predicted>=0)
        doc["repeats"][seed] = dict(folds=folds,scores=oof.tolist(),pred=predicted.tolist(),
                                   metrics=metrics(y,oof,predicted))
    write_json(HERE/"historical59.json",doc)
    print("fitted",doc["historical_fit"]["metrics"]["accuracy"],
          "OOF",[(seed,r["metrics"]["accuracy"]) for seed,r in doc["repeats"].items()])


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
