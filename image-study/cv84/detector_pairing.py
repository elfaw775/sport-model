"""Verify identical fold membership before comparing the nested classifier and the trained detector."""
import csv
import json
from pathlib import Path

import numpy as np

from nested import HERE, metrics, paired, write_json


def compare(root):
    def rows(path):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    manifest = json.loads((root / "folds.json").read_text(encoding="utf8"))
    detector = {r["image"]:r for r in rows(root / "detector-oof.csv")}
    classifier = {r["id"]:r for r in rows(root / "results/outer_predictions.csv")
                  if r["seed"] == "42" and r["method"] == "nested_hf"}
    assert set(detector) == set(classifier) == set(manifest["ids"])
    fitted = json.loads((root / "detector-report.json").read_text(encoding="utf8"))
    for reference, actual in zip(manifest["repeats"]["42"], fitted["folds"], strict=True):
        tr, val = set(actual["fit_indices"]), set(actual["calibration_indices"])
        assert tr.isdisjoint(val) and (tr | val) == set(reference["train"])
        assert actual["test_indices"] == reference["test"]
        assert not (tr | val).intersection(actual["test_indices"])
        for i in actual["test_indices"]:
            key = manifest["ids"][i]
            assert int(detector[key]["fold"]) == int(classifier[key]["fold"]) == reference["fold"]
            assert int(detector[key]["y"]) == int(classifier[key]["y"]) == manifest["labels"][i]
    def subset(ids):
        y = np.array([int(detector[i]["y"]) for i in ids])
        scores = np.array([float(detector[i]["score"]) for i in ids])
        output = np.array([int(detector[i]["pred"]) for i in ids])
        correct = np.array([int(classifier[i]["pred"]) for i in ids])
        return {"detector":metrics(y,scores,output), "nested_hf_vs_detector":paired(y,correct,output)}
    report = {"seed":42, "all84":subset(manifest["ids"]),
              "original59":subset([i for i in manifest["ids"] if detector[i]["old_split"]=="train"]),
              "n_fit_calibration_test":[[len(f[k]) for k in ("fit_indices","calibration_indices","test_indices")]
                                        for f in fitted["folds"]]}
    write_json(root / "detector-paired.json", report)


if __name__ == "__main__":
    compare(HERE)
