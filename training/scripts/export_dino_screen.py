"""Convert the public all-84 DINOv2 Small refit and its frozen encoder to CPU ONNX.

Run from a checkout with image-study dependencies, onnx and the pinned encoder available:
    python training/scripts/export_dino_screen.py --output-dir backend/models
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import torch
from torch.nn import functional as F
from transformers import AutoModel


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "image-study"))
from hf_assets import MODELS, get_model

NAME = "dinov2-fetlock-84"
HEAD_SHA = "71dc95cdf63bd8a9e79899e0d30317596801d0f9e97c71f088b0e6651047296c"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class FilmScore(torch.nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        for key in ("mean", "scale", "coef", "intercept"):
            self.register_buffer(key, torch.from_numpy(head[key].copy()))

    def forward(self, pixel_values):
        hidden = self.encoder(pixel_values=pixel_values, return_dict=False)[0]
        cls = F.normalize(hidden[:, 0], dim=1)
        patches = F.normalize(hidden[:, 1:].mean(dim=1), dim=1)
        feature = F.normalize(torch.cat([cls, patches], dim=1).double(), dim=1)
        scaled = (feature - self.mean) / self.scale
        return torch.sigmoid(scaled @ self.coef + self.intercept)


def export(dest):
    refit = ROOT / "image-study/published/cv84"
    head_path = refit / "refit84-head.npz"
    decision = json.loads((refit / "summary.json").read_bytes())["refit84"]
    if decision["winner"] != "dinov2" or digest(head_path) != HEAD_SHA or decision["head_sha256"] != HEAD_SHA:
        raise ValueError("Unexpected published classifier")
    repository, revision, expected = MODELS["dinov2"]
    # Eager attention uses ONNX operations supported by CPUExecutionProvider.
    encoder = AutoModel.from_pretrained(str(get_model("dinov2")), local_files_only=True,
                                        attn_implementation="eager")
    with np.load(head_path, allow_pickle=False) as head:
        scorer = FilmScore(encoder, head).requires_grad_(False).eval()
    dest.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest) as scratch:
        graph = Path(scratch) / (NAME + ".onnx")
        with torch.inference_mode():
            torch.onnx.export(scorer, (torch.zeros(1, 3, 336, 336),), str(graph),
                              input_names=["pixel_values"], output_names=["film_score"],
                              opset_version=17, dynamo=False, external_data=False)
        onnx.checker.check_model(str(graph))
        sha = digest(graph)
        manifest = dict(model_id="DINOv2-Small-refit84@" + sha[:12], file=graph.name,
                        sha256=sha, bytes=graph.stat().st_size, threshold=decision["threshold"],
                        encoder=repository, encoder_revision=revision,
                        encoder_license="Apache-2.0", encoder_license_file="DINOv2-LICENSE.txt",
                        encoder_weights_sha256=expected["model.safetensors"], head_sha256=HEAD_SHA,
                        training_films=84, task="whole-film lesion screening", opset=17,
                        preprocessing="EXIF transpose; grayscale; bicubic longest edge to 336; center on black "
                                      "336 square; RGB NCHW float32 [0,1]; ImageNet mean and std")
        info = graph.with_suffix(".json")
        info.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
        graph.replace(dest / graph.name)
        info.replace(dest / info.name)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "backend/models")
    torch.set_num_threads(4)
    export(parser.parse_args().output_dir.resolve())
