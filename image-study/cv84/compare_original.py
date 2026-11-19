"""Audit the three models on the same 84 films and export their paired decisions."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import binomtest


PUBLISHED = Path(__file__).resolve().parents[1] / "published"


def metrics(y, decisions):
    tp = int(sum((y == 1) & (decisions == 1)))
    tn = int(sum((y == 0) & (decisions == 0)))
    correct = tp + tn
    interval = binomtest(correct, len(y)).proportion_ci()
    return dict(n=len(y), correct=correct, tp=tp, tn=tn, fp=int(sum(y == 0)) - tn,
                fn=int(sum(y == 1)) - tp, accuracy=correct / len(y),
                sensitivity=tp / int(sum(y == 1)), specificity=tn / int(sum(y == 0)),
                accuracy_ci95=[interval.low, interval.high])


def audit(original, published, output):
    splits = json.loads((published / "cv84/folds.json").read_bytes())
    with zipfile.ZipFile(published / "evidence.zip") as evidence:
        paired = json.loads(evidence.read("cv84/matched-budget.json"))
    detector_bytes = (published / "cv84/detector-report.json").read_bytes()
    detector = json.loads(detector_bytes)
    assert paired["ids"] == splits["ids"] and paired["y"] == splits["labels"]
    assert paired["order"] == ["dinov2", "yolo11s_detection"]
    ids = splits["ids"]
    y = np.array(splits["labels"])
    assert len(set(ids)) == len(y) == 84 and sum(y) == 68
    old = json.loads(original.read_bytes())
    assert len(old["folds"]) == len({f["checkpoint_sha256"] for f in old["folds"]}) == 5
    recipe_hash = hashlib.sha256((published / "cv84/original-yolo-recipe.json").read_bytes()).hexdigest()
    detector_hash = hashlib.sha256(detector_bytes).hexdigest()
    rows = {}
    for a, b, c, partition in zip(old["folds"], detector["folds"], paired["folds"],
                                   splits["repeats"]["42"], strict=True):
        assert a["fold"] == b["fold"] == c["fold"] == partition["fold"]
        assert a["amp_enabled"] is False and a["batch"] == 16 and 1 <= a["completed_epochs"] <= 200
        assert a["cuda_matmul_tf32"] is False and a["cudnn_tf32"] is False and a["best_state_all_finite"]
        assert a["original_recipe_sha256"] == recipe_hash and a["split_report_sha256"] == detector_hash
        assert a["fixed_threshold"] == 0.15 and a["manifest_sha256"] == splits["manifest_sha256"]
        train, cal, held = a["fit_indices"], a["calibration_indices"], a["test_indices"]
        assert train == b["fit_indices"] == c["fit"]
        assert cal == b["calibration_indices"] == c["calibration"]
        assert held == b["test_indices"] == c["held"] == partition["test"]
        assert sorted(train + cal + held) == list(range(84))
        assert len(cal) == 17 and len(held) in (16, 17)
        # Independently enumerate calibration partitions using their observed score ranks.
        scores = np.asarray(a["calibration_scores"], dtype=float)
        assert len(scores) == len(cal) and np.isfinite(scores).all()
        truth = y[cal]
        def operating_point(t):
            is_pos = scores >= t
            tp = int(sum(is_pos & (truth == 1)))
            tn = int(sum(~is_pos & (truth == 0)))
            return tp * int(sum(truth == 0)) + tn * int(sum(truth == 1)), tp, tn
        levels = np.unique(scores)
        partitions = [math.nextafter(levels[-1], math.inf), *levels]
        assert operating_point(a["threshold"]) == max(operating_point(t) for t in partitions)
        for i, score, new_score in zip(held, a["scores"], b["scores"], strict=True):
            dino_score = paired["scores"][0][i]
            assert i not in rows and all(math.isfinite(v) for v in (score, new_score, dino_score))
            assert paired["scores"][1][i] == new_score
            assert c["yolo_threshold"] == b["threshold"]
            new_pred = int(new_score >= b["threshold"])
            dino_pred = int(dino_score >= c["dino_threshold"])
            assert new_pred == paired["decisions"][1][i] and dino_pred == paired["decisions"][0][i]
            rows[i] = dict(image=ids[i], fold=a["fold"] + 1, historical_split=splits["historical_splits"][i],
                           label=int(y[i]), original_score=score, original_threshold=a["threshold"],
                           original=int(score >= a["threshold"]), original_fixed_015=int(score >= 0.15),
                           retrained_score=new_score, retrained_threshold=b["threshold"], retrained=new_pred,
                           dino_score=dino_score, dino_threshold=c["dino_threshold"], dino=dino_pred)
    assert set(rows) == set(range(len(y)))
    ordered = [rows[i] for i in range(len(y))]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "three-model-oof.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    models, paired_errors = {}, {}
    dino_correct = np.array([r["dino"] == r["label"] for r in ordered])
    for name in ("original", "retrained", "dino", "original_fixed_015"):
        decisions = np.array([r[name] for r in ordered])
        models[name] = metrics(y, decisions)
        if name in ("original", "retrained"):
            correct = decisions == y
            paired_errors[name] = dict(dino_only=int(sum(dino_correct & ~correct)),
                                       yolo_only=int(sum(~dino_correct & correct)))
    summary = dict(unit="film", seed=42, folds=5, positives=68, negatives=16, models=models,
                   paired_dino_correct=paired_errors, source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                   retrained_public_report_sha256=detector_hash)
    (output / "three-model-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=PUBLISHED / "cv84/original-yolo-report.json")
    parser.add_argument("--published", type=Path, default=PUBLISHED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.original, args.published, args.output), indent=2))
