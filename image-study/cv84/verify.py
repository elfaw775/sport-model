"""Read back completed decisions, membership and score files and check the frozen-head predictions."""
import csv
import json

import numpy as np
from threadpoolctl import threadpool_limits

from nested import HERE, OUT, HF, PRIMARY, SEEDS, features, midpoint_threshold, probe_scores, sha


def check():
    folds = json.loads((HERE/"folds.json").read_text(encoding="utf8"))
    summary = json.loads((OUT/"summary.json").read_text(encoding="utf8"))
    decisions = json.loads((OUT/"inner_decisions.json").read_text(encoding="utf8"))
    y = np.asarray(folds["labels"])
    ids = folds["ids"]
    methods = {*summary["source_feature_sha256"], "height", "all_positive", PRIMARY}
    with (OUT/"outer_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(int(r["seed"]),r["method"],int(r["index"])):r for r in rows}
    expected = {(s,m,i) for s in SEEDS for m in methods for i in range(len(ids))}
    if expected != set(indexed) or len(rows) != len(indexed):
        raise ValueError("Missing or duplicate outer score")
    if sha(HERE/"folds.json") != summary["folds_sha256"]:
        raise ValueError("Fold definitions changed")
    for seed in SEEDS:
        for split, trained in zip(folds["repeats"][str(seed)], decisions[str(seed)], strict=True):
            fold = split["fold"]
            held, train = np.array(split["test"]), np.array(split["train"])
            assert train.tolist() == trained["oof_train_indices"] and trained["outer_test_indices"] == held.tolist()
            assert set(train).isdisjoint(held) and sorted(np.concatenate(split["inner_validation"])) == sorted(train)
            summaries = trained["summaries"]
            for name, values in trained["inner_oof_scores"].items():
                cutoff = 1. if name == "all_positive" else midpoint_threshold(y[train],np.array(values))
                assert summaries[name]["threshold"] == cutoff
            winner = trained["winner"]
            assert winner in HF
            for m in methods:
                encoder = winner if m == PRIMARY else m
                for i in held:
                    r = indexed[seed,m,i]
                    assert r["id"] == ids[i] and int(r["fold"])==fold and int(r["y"])==y[i]
                    assert float(r["threshold"]) == summaries[encoder]["threshold"]
                    assert (float(r["score"])>=float(r["threshold"])) == bool(int(r["pred"]))
                    if m == PRIMARY:
                        assert r["score"] == indexed[seed,winner,i]["score"]
            if seed == 42:
                path = OUT / f"seed42-fold{fold}-head.npz"
                assert sha(path) == trained["head_sha256"]
                X,_ = features(winner,ids,y)
                with np.load(path,allow_pickle=False) as head:
                    rebuilt = probe_scores(head,X[held])
                np.testing.assert_allclose(rebuilt,[float(indexed[seed,PRIMARY,i]["score"]) for i in held],
                                           atol=1e-12,rtol=0)
    print(f"Verified {len(rows)} unique outer rows, {len(SEEDS)*5} folds, all thresholds and 5 saved heads")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        check()
