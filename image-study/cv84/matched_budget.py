"""Post hoc fixed DINOv2 probe using the detector's identical fit, calibration and outer film IDs."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from nested import HERE, features, fit_probe, metrics, midpoint_threshold, paired, probe_scores, sha, write_json


def run():
    detector = HERE / "detector-report.json"
    report = json.loads(detector.read_text(encoding="utf8"))
    mapping = json.loads((HERE/"folds.json").read_text(encoding="utf8"))
    ids = mapping["ids"]
    y = np.array(mapping["labels"])
    X, feature_hashes = features("dinov2", ids, y)
    scores = np.full((2,len(y)),np.nan)
    predictions = np.full((2,len(y)),-1, dtype=int)
    coverage = np.zeros(len(y), dtype=int)
    folds = []
    for partition, base in zip(mapping["repeats"]["42"],report["folds"],strict=True):
        fit = np.asarray(base["fit_indices"])
        cal = np.asarray(base["calibration_indices"])
        held = np.asarray(base["test_indices"])
        assert partition["test"] == held.tolist() and partition["fold"] == base["fold"]
        assert set(fit).isdisjoint(cal) and set(held).isdisjoint(fit) and set(held).isdisjoint(cal)
        assert sorted(np.concatenate((fit,cal,held))) == list(range(len(y)))
        head = fit_probe(X[fit],y[fit])
        calibration = probe_scores(head,X[cal])
        t = midpoint_threshold(y[cal],calibration)
        scores[0,held] = probe_scores(head,X[held])
        predictions[0,held] = scores[0,held] >= t
        scores[1,held] = base["scores"]
        predictions[1,held] = scores[1,held] >= base["threshold"]
        coverage[held] += 1
        folds.append(dict(fold=base["fold"],fit=fit.tolist(),calibration=cal.tolist(),held=held.tolist(),
                          dino_threshold=t,calibration_scores=calibration.tolist(),yolo_threshold=base["threshold"]))
    assert np.all(coverage == 1) and np.isfinite(scores).all()
    results = {k:metrics(y,scores[i],predictions[i]) for i,k in enumerate(("dinov2","yolo11s_detection"))}
    write_json(HERE/"matched-budget.json",dict(design="post hoc seed-42, fixed DINOv2 Small probe, matched film IDs",
        detector_report_sha256=sha(detector), feature_sha256=feature_hashes, folds=folds,
        ids=ids,y=y.tolist(),scores=scores.tolist(),decisions=predictions.tolist(),order=list(results),
        models=results,paired=paired(y,*predictions)))
    print([(k, r["tp"],r["fp"],r["tn"],r["fn"],r["balanced_accuracy"],r["roc_auc"]) for k,r in results.items()])
    print(paired(y,*predictions))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
