"""Evaluate a fully labeled one-step reward update on the matched 84-film folds."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import fit_probe, probe_scores, unit
from canvas_round2 import make_content
from hf_features import FrozenEncoder, SIZE
from nested import midpoint_threshold
from study_paths import DATA, PUBLISHED


SOURCE = PUBLISHED / "cv84"
PENALTY = 0.05


def train_reward(x, y, original):
    """The known labels specify the reward of each of the two possible actions."""
    scaled = (unit(x.astype(np.float64)) - original["mean"]) / original["scale"]
    design = np.column_stack((scaled, np.ones(len(y))))
    start = np.r_[original["coef"], original["intercept"]].astype(np.float64)
    frequencies = np.bincount(y, minlength=2)
    weights = 0.5 / frequencies[y]

    signs = 1.0 - 2.0 * y

    def evaluate(params):
        error = expit(signs * (design @ params))
        delta = params - start
        expected_error = weights @ error
        gradient = design.T @ (weights * signs * error * (1 - error))
        return expected_error + PENALTY / 2 * (delta @ delta), gradient + PENALTY * delta

    result = minimize(evaluate, start, jac=True, method="L-BFGS-B",
                      options={"maxiter": 600, "maxls": 60, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    params = result.x
    info = dict(initial_loss=float(evaluate(start)[0]), final_loss=float(result.fun),
                weight_change_l2=float(np.linalg.norm(params-start)), iterations=int(result.nit))
    assert info["final_loss"] <= info["initial_loss"] and info["weight_change_l2"] > 0
    return {**original, "coef": params[:-1], "intercept": params[-1:]}, info


def run(output):
    folds = json.loads((SOURCE / "folds.json").read_bytes())
    partitions = json.loads((SOURCE / "detector-report.json").read_bytes())["folds"]
    with (SOURCE / "three-model-oof.csv").open(encoding="utf-8-sig", newline="") as f:
        reference = {r["image"]:r for r in csv.DictReader(f)}
    ids = folds["ids"]
    y = np.asarray(folds["labels"], dtype=int)
    files = {p.stem:p for p in (DATA / "images").glob("*/*") if p.is_file()}
    assert set(ids) == set(files) == set(reference) and len(ids) == 84
    encoder = FrozenEncoder("dinov2")
    chunks = []
    for start in range(0, len(ids), 6):
        batch = []
        for key in ids[start:start+6]:
            content = make_content(files[key])
            frame = Image.new("L", (SIZE, SIZE))
            frame.paste(content, ((SIZE-content.width)//2, (SIZE-content.height)//2))
            batch.append(frame)
        chunks.append(encoder.encode(batch))
    x = np.vstack(chunks)
    rows, weights, training = {}, {}, []
    for part in partitions:
        fold = part["fold"]
        fit, cal, test = (np.asarray(part[k]) for k in ("fit_indices","calibration_indices","test_indices"))
        assert sorted(np.r_[fit,cal,test].tolist()) == list(range(84))
        base = fit_probe(x[fit], y[fit])
        new, info = train_reward(x[fit], y[fit], base)
        cutoff = float(midpoint_threshold(y[cal], probe_scores(new, x[cal])))
        for key, value in new.items():
            weights[f"fold{fold}_{key}"] = value
        weights[f"fold{fold}_threshold"] = cutoff
        training.append({"fold":fold+1, "threshold":cutoff, **info})
        for i, bs, rs in zip(test, probe_scores(base,x[test]), probe_scores(new,x[test]), strict=True):
            ref = reference[ids[i]]
            bt = float(ref["dino_threshold"])
            assert i not in rows and np.isfinite(rs) and np.isfinite(bs)
            assert abs(bs-float(ref["dino_score"]))<0.002 and int(bs>=bt)==int(ref["dino"]), (ids[i], bs, ref["dino_score"], bt)
            rows[i] = dict(image=ids[i], fold=fold+1, truth=int(y[i]), supervised_score=bs,
                           supervised_threshold=bt, supervised=int(bs>=bt), reward_score=rs,
                           reward_threshold=cutoff, reward=int(rs>=cutoff))
    assert sorted(rows) == list(range(84))
    output.mkdir(exist_ok=True, parents=True)
    with (output/"scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[i] for i in range(84))
    np.savez_compressed(output/"reward-fold-heads.npz", **weights)
    metrics = {}
    for name in ("supervised", "reward"):
        p = np.array([rows[i][name] for i in range(84)])
        metrics[name] = dict(correct=int(sum(y==p)), missed=int(sum((y==1)&(p==0))),
                             false_alarms=int(sum((y==0)&(p==1))))
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(films=84, positives=68, negatives=16, models=metrics, fits=training,
                  changed=[rows[i]["image"] for i in range(84) if rows[i]["reward"]!=rows[i]["supervised"]],
                  baseline_max_score_drift=max(abs(rows[i]["supervised_score"]-float(reference[ids[i]]["dino_score"])) for i in rows),
                  device=str(encoder.device),plan_sha256=sha(Path(__file__).with_name("one-step-reward-plan.md")),
                  scores_sha256=sha(output/"scores.csv"), weights_sha256=sha(output/"reward-fold-heads.npz"))
    (output/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n", encoding="utf8", newline="\n")
    print(json.dumps(metrics),flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(ap.parse_args().output)
