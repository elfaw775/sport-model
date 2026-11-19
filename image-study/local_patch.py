"""Train a lesion-window classifier and a height-only negative control."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

from study_paths import WORK as HERE, DATA as SOURCE, DETECTOR_RUN

CORRECTED = DETECTOR_RUN / "dataset"
MODEL = HERE / "local_models/patch-cnn.pt"
CONTROL = HERE / "local_models/acquisition.json"
SIZES = np.array([64, 112, 176, 240])
CANVAS = 320
PATCH = 64
SEED = 731


def get_rows(split):
    with (SOURCE / "split_manifest.tsv").open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream, delimiter="\t") if row["split"] == split]


def read_images(split, corrected, train=False):
    inputs = []
    by_id = {p.stem: p for p in (SOURCE / "images" / split).iterdir() if p.is_file()}
    for row in get_rows(split):
        with Image.open(by_id[row["image"]]) as image:
            height = image.height
            gray = np.asarray(image.convert("L"))
        image = cv2.resize(gray, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
        boxes = []
        if train:
            assert split == "train"
            label = corrected / "labels" / split / (row["image"] + ".txt")
            for line in label.read_text(encoding="utf-8").splitlines():
                category, cx, cy, w, h = map(float, line.split())
                assert category == 0 and w > 0 and h > 0
                boxes.append(np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])*CANVAS)
            assert len(boxes) == int(row["class0_boxes"])
        inputs.append((row["image"], int(row["category"] == "pos"), height, image, boxes))
    return inputs


class PatchNet(nn.Sequential):
    def __init__(self):
        layers = []
        channels = 1
        for width in (16, 32, 48):
            layers.extend([nn.Conv2d(channels, width, 3, padding=1, bias=False),
                           nn.GroupNorm(4, width), nn.ReLU(), nn.MaxPool2d(2)])
            channels = width
        layers.extend([nn.AvgPool2d(4), nn.Flatten(), nn.Linear(48*4, 1)])
        super().__init__(*layers)


def patch_at(image, x, y, side):
    left = int(np.clip(round(x-side/2), 0, CANVAS-side))
    top = int(np.clip(round(y-side/2), 0, CANVAS-side))
    cut = image[top:top+side, left:left+side]
    return cv2.resize(cut, (PATCH, PATCH), interpolation=cv2.INTER_AREA)


def make_tensor(batch, device):
    inputs = torch.as_tensor(np.stack(batch), device=device).unsqueeze(1).float()
    means = inputs.mean(dim=(2, 3), keepdim=True)
    stds = inputs.std(dim=(2, 3), correction=0, keepdim=True)
    return ((inputs - means) / stds.clamp_min(8)).clamp(-3, 3)


def background_intersects(boxes, x, y, side):
    start = np.array([x-side/2, y-side/2])
    end = start + side
    for box in boxes:
        intersection = np.maximum(0, np.minimum(end, box[2:])-np.maximum(start, box[:2]))
        if np.prod(intersection) > 0.10 * min(side*side, np.prod(box[2:]-box[:2])):
            return True
    return False


def random_background(records, rng):
    for _ in range(100):
        _, _, _, image, boxes = records[rng.integers(len(records))]
        side = int(rng.choice(SIZES))
        x, y = rng.uniform(side/2, CANVAS-side/2, size=2)
        if not background_intersects(boxes, x, y, side):
            return patch_at(image, x, y, side)
    raise RuntimeError("Could not find a training background patch")


def train(examples):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = "cuda"
    model = PatchNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    positives = [e for e in examples if e[1]]
    negatives = [e for e in examples if not e[1]]
    lesions = [(e[3], b) for e in positives for b in e[4]]
    history = []
    for _ in range(18):
        losses = []
        for _ in range(12):
            tiles = []
            for _ in range(32):
                image, box = lesions[rng.integers(len(lesions))]
                side = int(np.clip(np.max(box[2:]-box[:2]) * rng.uniform(1.1, 1.7), 40, CANVAS))
                x, y = (box[:2]+box[2:])/2 + rng.uniform(-0.12, 0.12, size=2)*side
                tiles.append(patch_at(image, x, y, side))
            for group in (negatives, positives):
                tiles.extend(random_background(group, rng) for _ in range(16))
            tiles = [tile[:, ::-1] if rng.random() < 0.5 else tile for tile in tiles]
            target = torch.cat([torch.ones(32), torch.zeros(32)]).to(device)
            logits = model(make_tensor(tiles, device)).flatten()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    return model, history


@torch.inference_mode()
def image_score(model, image):
    tiles = []
    for side in SIZES:
        starts = np.linspace(side/2, CANVAS-side/2, int(np.ceil((CANVAS-side)/(side/2)))+1)
        tiles.extend(patch_at(image, x, y, side) for y in starts for x in starts)
    device = next(model.parameters()).device
    scores = torch.cat([model(make_tensor(tiles[i:i+64], device)).flatten().sigmoid()
                        for i in range(0, len(tiles), 64)])
    return scores.topk(3).values.mean().item()


def record(split, examples, model, median):
    target = HERE / "features"
    target.mkdir(exist_ok=True)
    ids = np.array([e[0] for e in examples])
    ys = np.array([e[1] for e in examples], dtype=np.int64)
    for key, scores in (("patch", [image_score(model, e[3]) for e in examples]),
                        ("acquisition", [abs(e[2] - median) for e in examples])):
        np.savez_compressed(target/f"{key}-{split}.npz", ids=ids, y=ys, scores=np.asarray(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "test", "predict"))
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--corrected", type=Path, default=CORRECTED)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "fit":
        if device != "cuda":
            raise RuntimeError("The prespecified training recipe requires CUDA")
        if MODEL.exists() or CONTROL.exists():
            raise FileExistsError("Existing model. Use a separate experiment directory for a new training run.")
        t0 = perf_counter()
        examples = read_images("train", args.corrected, train=True)
        median = float(np.median([e[2] for e in examples if not e[1]]))
        model, history = train(examples)
        MODEL.parent.mkdir(exist_ok=True, parents=True)
        torch.save(model.cpu().state_dict(), MODEL)
        model.to(device).eval()
        CONTROL.write_text(json.dumps({"negative_train_median_height": median,
            "fit_seconds": round(perf_counter()-t0, 2), "epochs": 18, "train_bce": history,
            "parameters": sum(p.numel() for p in model.parameters()),
            "sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
        record("val", read_images("val", args.corrected), model, median)
        print(CONTROL.read_text())
    else:
        model = PatchNet().to(device)
        control = json.loads(CONTROL.read_text())
        if control["sha256"] != hashlib.sha256(MODEL.read_bytes()).hexdigest():
            raise ValueError("Patch model does not match its fitted weights")
        model.load_state_dict(torch.load(MODEL, map_location=device, weights_only=True))
        model.eval()
        if args.mode == "test":
            if not args.lock or not args.lock.is_file():
                parser.error("Provide the saved development lock before evaluating test.")
            record("test", read_images("test", args.corrected), model, control["negative_train_median_height"])
        else:
            if not args.input:
                parser.error("predict requires --input")
            with Image.open(args.input) as im:
                gray = cv2.resize(np.asarray(im.convert("L")), (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
            print(json.dumps({"patch_score": image_score(model, gray)}))


if __name__ == "__main__":
    main()
