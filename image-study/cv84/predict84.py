"""Run the research head refitted on all 84 films with its frozen encoder."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from study_paths import PUBLISHED
from evaluate import probe_scores, sha


def encoder_for(name, device):
    if name in {"dinov2", "rad-dino", "resnet18"}:
        from hf_features import FrozenEncoder
        return FrozenEncoder(name, device)
    if name in {"bone-siglip", "biomedclip"}:
        from medical_features import MedicalEncoder
        return MedicalEncoder(name, device)
    from vision_backbones import CHECKPOINTS, Encoder, verified_folder
    if name not in CHECKPOINTS:
        raise ValueError(f"Unsupported encoder: {name}")
    return Encoder(name, verified_folder(name), device)


def predict(path, device, results=PUBLISHED / "cv84"):
    decisions = json.loads((results/"summary.json").read_text(encoding="utf8"))["refit84"]
    checkpoint = results/"refit84-head.npz"
    if sha(checkpoint) != decisions["head_sha256"]:
        raise ValueError("Refit head checksum mismatch")
    encoder = encoder_for(decisions["winner"], device)
    with Image.open(path) as image, np.load(checkpoint, allow_pickle=False) as head:
        score = float(probe_scores(head,encoder.encode([image]))[0])
    t = decisions["threshold"]
    return dict(encoder=decisions["winner"], score=score, threshold=t, predicted_positive=score>=t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image",type=Path)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--results", type=Path, default=PUBLISHED / "cv84", help="Directory containing summary.json and refit84-head.npz")
    args = parser.parse_args()
    print(json.dumps(predict(args.image,args.device,args.results),indent=2))
