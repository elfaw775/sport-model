"""Fit small spatial heads to frozen Qwen3-VL merger tokens on the existing horse split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.special import expit

from compare_detectors import WORK, counts, iou

DATA = WORK / "dataset"
FEATURE_DIM = 128
AXES = ("x1", "y1", "x2", "y2")
CUTOFFS = [.01, .025, .05, .075, .10, .15, .20, .25, .30, .40, .50, .60,
           .70, .80, .90, .95, .99, 1.001]


def paths(split):
    return sorted(path for path in (DATA / "images" / split).iterdir()
                  if path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def extract(model_path, output):
    import mlx.core as mx
    from mlx_vlm import load
    from evaluate_qwen_vl import WEIGHT_SHA256

    with (model_path / "model.safetensors").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHT_SHA256:
            raise ValueError("Frozen probe requires the checkpoint used for this experiment")
    model, processor = load(str(model_path))
    tower = model.vision_tower
    dtype = tower.patch_embed.proj.weight.dtype
    projection = None
    for split in ("train", "val", "test"):
        folder = output / "features" / split
        folder.mkdir(parents=True, exist_ok=True)
        for path in paths(split):
            # Pass pixels only. The examples, filenames and labels never enter the encoder.
            with Image.open(path) as source:
                pixels = processor.image_processor(source.convert("RGB"), min_pixels=320**2,
                                                   max_pixels=640**2, return_tensors="np")
                size = source.size
            grid = np.asarray(pixels["image_grid_thw"])
            tokens, _ = tower(mx.array(pixels["pixel_values"]).astype(dtype), mx.array(grid))
            tokens = np.asarray(tokens.astype(mx.float32))
            _, h, w = grid[0]
            h, w = int(h)//2, int(w)//2
            assert len(tokens) == h*w
            if projection is None:
                rng = np.random.default_rng(42)
                projection = rng.standard_normal((tokens.shape[-1], FEATURE_DIM), dtype=np.float32)
                projection /= np.sqrt(FEATURE_DIM)
            tokens = tokens / np.maximum(np.linalg.norm(tokens, axis=1, keepdims=True), 1e-8)
            features = (tokens @ projection).reshape(h, w, FEATURE_DIM)
            np.savez_compressed(folder / (path.name + ".npz"), features=features.astype("float16"), size=size)
            mx.clear_cache()
        print(f"Extracted {split} ({len(paths(split))} horses)", flush=True)
    np.save(output / "projection.npy", projection)


def sample(output, split, path):
    with np.load(output / "features" / split / (path.name + ".npz")) as stored:
        grid = stored["features"].astype("float64")
        width, height = map(int, stored["size"])
    rows, cols, depth = grid.shape
    y, x = np.mgrid[:rows, :cols]
    anchors = np.stack(((x+.5)/cols, (y+.5)/rows), axis=-1).reshape(-1, 2)
    return grid.reshape(-1, depth), anchors, np.array([width, height, width, height])


def normalized_labels(split, path):
    lines = (DATA / "labels" / split / (path.stem+".txt")).read_text().splitlines()
    boxes = []
    for line in lines:
        cls, x, y, w, h = map(float, line.split())
        assert cls == 0
        boxes.append([x-w/2, y-h/2, x+w/2, y+h/2])
    return np.array(boxes).reshape(-1, 4)


def anchor_targets(anchors, labels):
    # Positive anchors are near a lesion centre. Ignore its ambiguous boundary band.
    target = np.zeros(len(anchors))
    offsets = np.zeros((len(anchors), 4))
    for box in labels:
        inside = np.all((anchors >= box[:2]) & (anchors <= box[2:]), axis=1)
        target[inside] = -1
    for box in sorted(labels, key=lambda b: -np.prod(b[2:]-b[:2])):
        centre = (box[:2]+box[2:])/2
        size = np.maximum(box[2:]-box[:2], 1e-6)
        scaled_delta = (centre-anchors)/size
        near = np.max(np.abs(scaled_delta), axis=1) <= .25
        near[np.argmin((scaled_delta**2).sum(axis=1))] = True
        target[near] = 1
        offsets[near, :2] = centre-anchors[near]
        offsets[near, 2:] = np.log(size)
    return target, offsets


def balanced_logistic(x, y):
    weights = np.where(y == 1, .5 / y.sum(), .5 / (len(y)-y.sum()))

    def objective(beta):
        logits = x @ beta
        loss = weights @ (np.logaddexp(0, logits)-y*logits)
        gradient = x.T @ (weights * (expit(logits)-y))
        gradient[:-1] += .02 * beta[:-1]
        return loss + .01 * beta[:-1] @ beta[:-1], gradient

    opt = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not opt.success:
        raise RuntimeError(opt.message)
    return opt.x


def fit_head(output):
    rows, classes, boxes = [], [], []
    for path in paths("train"):
        features, anchors, _ = sample(output, "train", path)
        target, offsets = anchor_targets(anchors, normalized_labels("train", path))
        keep = target >= 0
        rows.append(features[keep])
        classes.append(target[keep])
        boxes.append(offsets[keep])
    x, y, offsets = np.concatenate(rows), np.concatenate(classes), np.concatenate(boxes)
    mean, std = x.mean(0), x.std(0).clip(1e-5)
    x = np.column_stack(((x-mean)/std, np.ones(len(x))))
    beta = balanced_logistic(x, y)
    z = x[y == 1]
    penalty = np.eye(x.shape[1])*.2*len(z)
    penalty[-1, -1] = 0
    reg = np.linalg.solve(z.T@z + penalty, z.T @ offsets[y == 1])
    np.savez(output/"head.npz", mean=mean, std=std, classifier=beta, box_regressor=reg)
    print(f"Fitted {int(y.sum())} positive and {int((y == 0).sum())} background anchors", flush=True)
    return mean, std, beta, reg


def predict(head, features, anchors, dimensions):
    mean, std, classifier, regressor = head
    design = np.column_stack(((features-mean)/std, np.ones(len(features))))
    scores = expit(design @ classifier)
    offsets = design @ regressor
    centres = anchors + offsets[:, :2]
    half = .5*np.exp(np.clip(offsets[:, 2:], -7, 0))
    boxes = np.clip(np.column_stack((centres-half, centres+half)), 0, 1)*dimensions
    selected = []
    for index in np.argsort(-scores, kind="stable"):
        box = boxes[index]
        if np.all(box[:2] < box[2:]) and all(iou(box, boxes[j]) <= .30 for j in selected):
            selected.append(index)
            if len(selected) >= 8:
                break
    return [dict(zip(AXES, boxes[i]), score=scores[i]) for i in selected]


def screening(output):
    # Predefined, separate image-level experiment: pool frozen tokens, no boxes in the classifier input.
    features, labels, filenames = {}, {}, {}
    for split in ("train", "val", "test"):
        vectors, names = [], []
        for path in paths(split):
            grid, _, _ = sample(output, split, path)
            vectors.append(np.r_[grid.mean(axis=0), grid.std(axis=0)])
            names.append(path.name)
        features[split], filenames[split] = np.array(vectors), names
        if split != "test":
            labels[split] = np.array([bool(len(normalized_labels(split, p))) for p in paths(split)])
    mean = features["train"].mean(axis=0)
    std = features["train"].std(axis=0).clip(1e-5)
    design = {s: np.column_stack(((x-mean)/std, np.ones(len(x)))) for s, x in features.items()}
    beta = balanced_logistic(design["train"], labels["train"].astype(float))
    scores = {s: expit(x@beta) for s, x in design.items()}

    def criterion(cutoff):
        pred = scores["val"] >= cutoff
        truth = labels["val"]
        balanced_accuracy = (np.mean(pred[truth]) + np.mean(~pred[~truth]))/2
        return balanced_accuracy, -abs(.5-cutoff), cutoff

    cutoff = max(CUTOFFS, key=criterion)
    np.savez(output/"screen-head.npz", mean=mean, std=std, classifier=beta, cutoff=cutoff)
    rows = [dict(split=s, filename=name, status="positive" if score >= cutoff else "negative",
                 score=score, boxes=[]) for s in ("val","test")
            for name, score in zip(filenames[s], scores[s])]
    (output/"screen-predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")


def height_control():
    """Exploratory source-format control. Derive the rule from train only and never use it as a predictor input."""
    def records(split):
        for path in paths(split):
            with Image.open(path) as image:
                yield image.height, bool(len(normalized_labels(split, path)))
    training_heights = Counter(h for h, positive in records("train") if not positive)
    height, _ = training_heights.most_common(1)[0]
    results = {}
    for split in ("train", "val", "test"):
        metrics = dict(tp=0, fp=0, tn=0, fn=0)
        for h, positive in records(split):
            predicted = h != height
            key = ("tp" if positive else "fp") if predicted else ("fn" if positive else "tn")
            metrics[key] += 1
        results[split] = metrics
    return {"train_negative_modal_height_px":height, "train_negative_heights":dict(training_heights), "splits":results}


def fit(output):
    from compare_detectors import truth
    screening(output)
    head = fit_head(output)
    label_val = truth("val")
    rows = []
    for split in ("val", "test"):
        predictions, dimensions = {}, {}
        for path in paths(split):
            features, anchors, dims = sample(output, split, path)
            predictions[path.name] = predict(head, features, anchors, dims)
            dimensions[path.name] = dims
        (output/f"{split}_predictions.json").write_text(json.dumps(predictions, ensure_ascii=False))
        if split == "val":
            candidates = [(c, counts(predictions, label_val, c)) for c in CUTOFFS]
            feasible = [(c, m) for c, m in candidates if m["fp"] <= len(label_val)]
            cutoff, _ = max(feasible, key=lambda pair: (pair[1]["tp"], -pair[1]["fp"], -pair[0]))
            print(f"Validation selected cutoff: {cutoff}", flush=True)
            (output/"selected-cutoff.json").write_text(json.dumps({"cutoff":cutoff})+"\n")
        for name, hits in predictions.items():
            boxes = [[float(box[axis]*1000/dim) for axis, dim in zip(AXES, dimensions[name])]
                     for box in hits if box["score"] >= cutoff]
            rows.append(dict(split=split, filename=name, status="positive" if boxes else "negative", boxes=boxes))
    (output/"probe-predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    (output/"height-control.json").write_text(json.dumps(height_control(), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["extract", "fit"])
    parser.add_argument("--model", type=Path, default=WORK/"qwen-vl/model")
    parser.add_argument("--output", type=Path, default=WORK/"qwen-probe")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "extract":
        extract(args.model, args.output)
    else:
        fit(args.output)
