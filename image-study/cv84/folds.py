"""Fixed nested stratified folds shared by all 84-film experiments."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_paths import CV as HERE, DATA

MANIFEST = DATA / "split_manifest.tsv"
SEEDS = (42, 73, 101)


def load_rows():
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 84 or len({row["image"] for row in rows}) != len(rows):
        raise ValueError("Expected 84 unique untrimmed image IDs")
    if any(row["category"] not in {"pos", "neg"} for row in rows):
        raise ValueError("Unexpected label")
    return rows


def split_indices(y, k, seed):
    splitter = StratifiedKFold(k, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def build():
    rows = load_rows()
    ids = np.asarray([r["image"] for r in rows])
    y = np.array([int(r["category"] == "pos") for r in rows])
    result = {"manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              "ids": ids.tolist(), "labels": y.tolist(), "historical_splits": [r["split"] for r in rows],
              "repeats": {}}
    for seed in SEEDS:
        folds = []
        for fold, (train, test) in enumerate(split_indices(y, 5, seed)):
            inner_seed = 1000 + seed + fold
            inner = [train[held].tolist() for _, held in split_indices(y[train], 4, inner_seed)]
            folds.append({"fold": fold, "train": train.tolist(), "test": test.tolist(),
                          "inner_seed": inner_seed, "inner_validation": inner})
        result["repeats"][str(seed)] = folds
    return result


if __name__ == "__main__":
    HERE.mkdir(parents=True, exist_ok=True)
    output = HERE / "folds.json"
    output.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
