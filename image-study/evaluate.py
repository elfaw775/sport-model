"""Fit two prespecified image probes, lock validation decisions, then score once."""
import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.special import expit
from scipy.stats import beta, binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
from study_paths import DATA, DETECTOR_RUN as RUN, WORK, locked_input

FEATURES = WORK / "features"
RESULTS = WORK / "results"
MANIFEST = DATA / "split_manifest.tsv"
HF = ("dinov2", "resnet18")
YOLO = ("archived_yolo11s", "retrained_yolo11s")
EXTRA = ("patch", "acquisition")
LOCK = RESULTS / "decisions.json"
HEAD = RESULTS / "winning-head.npz"
RAD_HEAD = RESULTS / "rad-dino-head.npz"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def cases(split):
    with MANIFEST.open(encoding="utf8", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["split"] == split]
    ids = [r["image"] for r in rows]
    if len(ids) != len(set(ids)) or any(r["category"] not in ("pos", "neg") for r in rows):
        raise ValueError("Invalid manifest")
    return ids, np.array([int(r["category"] == "pos") for r in rows], dtype=int)


def load_npz(name, split, field):
    ids, truth = cases(split)
    path = FEATURES / f"{name}-{split}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        actual = arrays["ids"].astype(str).tolist()
        if len(actual) != len(ids) or set(ids) != set(actual):
            raise ValueError(f"IDs differ from manifest in {path}")
        index = {x: i for i, x in enumerate(actual)}
        indices = [index[x] for x in ids]
        labels = arrays["y"][indices]
        values = arrays[field][indices].astype(np.float64)
    if not np.array_equal(labels, truth) or len(values) != len(labels) or not np.isfinite(values).all():
        raise ValueError(f"Invalid data in {path}")
    if values.ndim != (2 if field == "X" else 1):
        raise ValueError(f"Unexpected shape for {field} in {path}")
    return values, path


def unit(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1, norms)


def fit_probe(X, y):
    scaled = StandardScaler().fit(unit(X))
    clf = LogisticRegression(C=1, class_weight="balanced", max_iter=3000, solver="lbfgs")
    clf.fit(scaled.transform(unit(X)), y)
    return dict(mean=scaled.mean_, scale=scaled.scale_, coef=clf.coef_[0], intercept=clf.intercept_)


def probe_scores(head, X):
    X = np.asarray(X, dtype=np.float64)
    values = (unit(X) - head["mean"]) / head["scale"]
    return expit(values @ head["coef"] + head["intercept"])


def choose_threshold(y, scores):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or not np.isfinite(scores).all() or set(y) != {0, 1}:
        raise ValueError("Expected aligned finite scores and both binary classes")
    pos, neg = int(sum(y)), len(y) - int(sum(y))
    candidates = np.append(np.unique(scores), np.nextafter(max(scores), np.inf))
    def rank(t):
        pred = scores >= t
        tp, tn = int(sum(pred & (y == 1))), int(sum(~pred & (y == 0)))
        return tp * neg + tn * pos, tp, tn, float(t)
    return float(max(candidates, key=rank))


def exact_interval(k, n):
    if not n:
        return None
    lo = float(beta.ppf(.025, k, n-k+1)) if k else 0.
    hi = float(beta.ppf(.975, k+1, n-k)) if k < n else 1.
    return [lo, hi]


def metrics(y, scores, threshold):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != y.shape or y.ndim != 1 or not np.isfinite(scores).all() or set(y) != {0, 1}:
        raise ValueError("Expected aligned finite scores and both binary classes")
    pred = scores >= threshold
    tp, fp = int(sum(pred & (y == 1))), int(sum(pred & (y == 0)))
    fn, tn = int(sum(~pred & (y == 1))), int(sum(~pred & (y == 0)))
    m = dict(n=len(y), tp=tp, fp=fp, tn=tn, fn=fn, threshold=float(threshold))
    for name, k, n in [("accuracy", tp+tn, len(y)), ("sensitivity", tp, tp+fn),
                       ("specificity", tn, tn+fp), ("precision", tp, tp+fp)]:
        m[name] = k/n if n else None
        m[name+"_ci95"] = exact_interval(k, n)
    m["balanced_accuracy"] = (m["sensitivity"] + m["specificity"])/2
    m["roc_auc"] = float(roc_auc_score(y, scores))
    m["average_precision"] = float(average_precision_score(y, scores))
    return m


def baseline_scores(name, split):
    ids, _ = cases(split)
    if name == "all_positive":
        return np.ones(len(ids)), None
    if name in EXTRA:
        return load_npz(name, split, "scores")
    path = RUN / f"{name}-{split}-predictions.json"
    predictions = read_json(path)
    by_stem = {Path(k).stem:max((box["score"] for box in boxes), default=0.)
               for k, boxes in predictions.items()}
    if len(by_stem) != len(predictions) or set(by_stem) != set(ids):
        raise ValueError(f"Missing or duplicated films in {path}")
    return np.asarray([by_stem[k] for k in ids]), path


def training_results():
    ids, y = cases("train")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(ids, y))
    summary = {}
    scores = {}
    for name in HF:
        X, path = load_npz(name, "train", "X")
        score = np.empty(len(y))
        for train, held in folds:
            score[held] = probe_scores(fit_probe(X[train], y[train]), X[held])
        summary[name] = dict(roc_auc=float(roc_auc_score(y, score)),
                             average_precision=float(average_precision_score(y, score)),
                             feature_sha256=sha(path))
        scores[name] = score.tolist()
    winner = max(HF, key=lambda n: (summary[n]["roc_auc"], summary[n]["average_precision"]))
    return winner, summary, dict(ids=ids, y=y.tolist(), scores=scores,
                                 held_fold_indices=[held.tolist() for _, held in folds])


def lock(include_rad=False):
    RESULTS.mkdir(exist_ok=True, parents=True)
    if LOCK.exists() or (RESULTS / "test.json").exists():
        raise ValueError("Model selection was already locked")
    name, summary, oof = training_results()
    X, _ = load_npz(name, "train", "X")
    head = fit_probe(X, cases("train")[1])
    Xval, valpath = load_npz(name, "val", "X")
    ids, y = cases("val")
    sources = {name: (probe_scores(head, Xval), valpath)}
    secondary = {}
    if include_rad:
        Xrad, trainpath = load_npz("rad-dino", "train", "X")
        radhead = fit_probe(Xrad, cases("train")[1])
        rad_val, path = load_npz("rad-dino", "val", "X")
        sources["rad-dino"] = probe_scores(radhead, rad_val), path
        np.savez(RAD_HEAD, **radhead)
        secondary["rad-dino"] = dict(train_feature_sha256=sha(trainpath), head_sha256=sha(RAD_HEAD))
    sources.update({b:baseline_scores(b, "val") for b in (*YOLO, *EXTRA, "all_positive")})
    report = {}
    scores = {}
    for model, (score, path) in sources.items():
        t = 1. if model == "all_positive" else choose_threshold(y, score)
        report[model] = metrics(y, score, t)
        report[model]["score_input_sha256"] = sha(path) if path else None
        scores[model] = score.tolist()
    np.savez(HEAD, **head)
    # Verify the portable parameters, independently of the sklearn object and on-disk pickle formats.
    with np.load(HEAD, allow_pickle=False) as obj:
        np.testing.assert_array_equal(probe_scores(obj, Xval), sources[name][0])
    write_json(RESULTS / "train_oof.json", oof)
    write_json(RESULTS / "val_predictions.json", dict(ids=ids, y=y.tolist(), scores=scores))
    artifact_names = ["local_models/patch-cnn.pt", "local_models/acquisition.json", "hf_features.py",
                      "hf_assets.py", "local_patch.py", "results/train_oof.json", "results/val_predictions.json"]
    artifact_names.extend(f"features/{model}-train-val.json" for model in (*HF, *secondary))
    output = dict(locked_at=datetime.now(timezone.utc).isoformat(), winner=name, train_oof=summary,
                  exploratory_heads=secondary,
                  validation=report, head_sha256=sha(HEAD), plan_sha256=sha(HERE / "evaluation-plan.md"),
                  manifest_sha256=sha(MANIFEST), eval_script_sha256=sha(Path(__file__)),
                  artifact_sha256={rel:sha(locked_input(rel)) for rel in artifact_names},
                  versions=dict(sklearn=sklearn.__version__, scipy=scipy.__version__, numpy=np.__version__))
    write_json(LOCK, output)
    print(f"Locked winner {name}. SHA256 {sha(LOCK)}")
    print(json.dumps(dict(oof=summary, validation=report), indent=2))


def score_test(expected_hash):
    destination = RESULTS / "test.json"
    if destination.exists():
        raise ValueError("Test set was already scored")
    if sha(LOCK) != expected_hash:
        raise ValueError("Approval does not match the development lock")
    locked = read_json(LOCK)
    for p, key in [(MANIFEST,"manifest_sha256"), (HEAD,"head_sha256"), (Path(__file__),"eval_script_sha256")]:
        if sha(p) != locked[key]:
            raise ValueError(f"Locked file changed: {p}")
    for rel, expected in locked["artifact_sha256"].items():
        if sha(locked_input(rel)) != expected:
            raise ValueError(f"Locked file changed: {rel}")
    ids, y = cases("test")
    winner = locked["winner"]
    X, path = load_npz(winner, "test", "X")
    with np.load(HEAD, allow_pickle=False) as head:
        sources = {winner:(probe_scores(head, X), path)}
    if "rad-dino" in locked["exploratory_heads"]:
        if sha(RAD_HEAD) != locked["exploratory_heads"]["rad-dino"]["head_sha256"]:
            raise ValueError("RAD-DINO head changed after lock")
        Xrad, path = load_npz("rad-dino", "test", "X")
        with np.load(RAD_HEAD, allow_pickle=False) as head:
            sources["rad-dino"] = probe_scores(head, Xrad), path
    sources.update({b:baseline_scores(b, "test") for b in (*YOLO, *EXTRA, "all_positive")})
    report = {}
    scores = {}
    correct = {}
    for model, (score, path) in sources.items():
        t = locked["validation"][model]["threshold"]
        scores[model] = score.tolist()
        report[model] = metrics(y, score, t)
        report[model]["score_input_sha256"] = sha(path) if path else None
        correct[model] = (score >= t) == y
    paired = {}
    for model in YOLO:
        won = int(sum(correct[winner] & ~correct[model]))
        lost = int(sum(~correct[winner] & correct[model]))
        paired[model] = dict(winner_only_right=won, baseline_only_right=lost,
                             mcnemar_exact_p=binomtest(won, won+lost, .5).pvalue if won+lost else 1.)
    write_json(RESULTS / "test_predictions.json", dict(ids=ids, y=y.tolist(), scores=scores))
    write_json(destination, dict(scored_at=datetime.now(timezone.utc).isoformat(),
                                 lock_sha256=expected_hash, winner=winner, models=report, paired=paired))
    print(json.dumps(read_json(destination), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lock_cmd = sub.add_parser("lock")
    lock_cmd.add_argument("--include-rad", action="store_true")
    score = sub.add_parser("score-test")
    score.add_argument("--expected-lock-sha", required=True)
    args = ap.parse_args()
    with threadpool_limits(limits=1):
        if args.cmd == "lock":
            lock(args.include_rad)
        else:
            score_test(args.expected_lock_sha)
