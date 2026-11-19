"""Score a JPG/PNG using the locked research probe and its original image encoder."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from study_paths import PUBLISHED
from evaluate import probe_scores, read_json, sha
from hf_features import FrozenEncoder


HEAD = PUBLISHED / "holdout/winning-head.npz"
LOCK = PUBLISHED / "holdout/decisions.json"


def predict(image_path, device=None):
    decisions = read_json(LOCK)
    if sha(HEAD) != decisions["head_sha256"]:
        raise ValueError("Saved head differs from the locked model")
    model = decisions["winner"]
    encoder = FrozenEncoder(model, device=device)
    with Image.open(image_path) as image, np.load(HEAD, allow_pickle=False) as head:
        score = float(probe_scores(head, encoder.encode([image]))[0])
    t = decisions["validation"][model]["threshold"]
    return {"encoder": model, "score": score, "threshold": t, "predicted_positive": score >= t}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.device), indent=2))
