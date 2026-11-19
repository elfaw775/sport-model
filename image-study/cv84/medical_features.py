"""Encode all 84 films once with fixed published medical-vision weights for group CV."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors import safe_open
from torchvision.transforms import functional as TF

from medical_assets import MODELS, download


from study_paths import CV as HERE, DATA as SOURCE
MANIFEST = SOURCE / "split_manifest.tsv"
OUT = HERE / "features"


class MedicalEncoder:
    """Map a list of PIL films to L2-normalized float32 embeddings, without labels."""
    def __init__(self, name, device="cuda"):
        folder = download(name)
        self.name = name
        self.device = torch.device(device)
        if name == "bone-siglip":
            from transformers import SiglipForImageClassification
            config = json.loads((folder / "preprocessor_config.json").read_text(encoding="utf-8"))
            full, diagnostics = SiglipForImageClassification.from_pretrained(
                folder, local_files_only=True, use_safetensors=True, output_loading_info=True
            )
            if diagnostics.get("missing_keys") or diagnostics.get("mismatched_keys"):
                raise RuntimeError(f"Incomplete bone encoder load: {diagnostics}")
            self.model = full.vision_model
            self.edge = config["size"]["height"]
            self.mean = config["image_mean"]
            self.std = config["image_std"]
            self.resample = Image.Resampling(config["resample"])
            self.pooling = "mean of final vision patch tokens (input of original binary classifier)"
        elif name == "biomedclip":
            from open_clip.model import CLIPVisionCfg, _build_vision_tower
            config = json.loads((folder / "open_clip_config.json").read_text(encoding="utf-8"))
            vision = config["model_cfg"]["vision_cfg"]
            self.model = _build_vision_tower(config["model_cfg"]["embed_dim"], CLIPVisionCfg(**vision))
            with safe_open(folder / "open_clip_pytorch_model.safetensors", framework="pt", device="cpu") as file:
                visual = {key.removeprefix("visual."): file.get_tensor(key)
                          for key in file.keys() if key.startswith("visual.")}
            self.model.load_state_dict(visual, strict=True)
            self.edge = vision["image_size"]
            self.mean = config["preprocess_cfg"]["mean"]
            self.std = config["preprocess_cfg"]["std"]
            self.resample = Image.Resampling.BICUBIC
            self.pooling = "native OpenCLIP visual encoder with learned 512-dimensional projection"
        else:
            raise ValueError(name)
        self.model.eval().requires_grad_(False).to(self.device)

    def transform(self, image):
        gray = ImageOps.exif_transpose(image).convert("L")
        scale = self.edge / max(gray.size)
        size = tuple(max(1, round(d * scale)) for d in gray.size)
        canvas = Image.new("L", (self.edge, self.edge))
        canvas.paste(gray.resize(size, self.resample),
                     ((self.edge - size[0]) // 2, (self.edge - size[1]) // 2))
        return TF.normalize(TF.to_tensor(canvas.convert("RGB")), self.mean, self.std)

    @torch.inference_mode()
    def encode(self, images):
        x = torch.stack([self.transform(im) for im in images]).to(self.device)
        if self.name == "bone-siglip":
            x = self.model(pixel_values=x).last_hidden_state.mean(dim=1)
        else:
            x = self.model(x)
        return torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy()


def extract(name, batch_size):
    repo, revision, hashes = MODELS[name]
    start = time.monotonic()
    model = MedicalEncoder(name)
    load_seconds = time.monotonic() - start
    with MANIFEST.open(encoding="utf-8", newline="") as inp:
        rows = list(csv.DictReader(inp, delimiter="\t"))
    ids = [r["image"] for r in rows]
    paths = {(p.parent.name, p.stem): p for p in (SOURCE / "images").glob("*/*") if p.is_file()}
    if len(set(ids)) != 84 or len(ids) != 84 or set(paths) != {(r["split"],r["image"]) for r in rows}:
        raise ValueError("Inconsistent archive and manifest")
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    values = []
    for offset in range(0, len(rows), batch_size):
        images = []
        for row in rows[offset:offset+batch_size]:
            with Image.open(paths[(row["split"], row["image"])]) as film:
                images.append(film.copy())
        values.append(model.encode(images))
    X = np.vstack(values)
    if not np.isfinite(X).all() or not np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5):
        raise ValueError("Invalid embeddings")
    seconds = time.monotonic() - start
    y = np.asarray([{"neg": 0, "pos": 1}[r["category"]] for r in rows], dtype=np.int64)
    out = OUT / f"{name}.npz"
    partial = out.with_suffix(".partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, ids=np.asarray(ids), X=X, y=y)
    partial.replace(out)
    packages = ["torch","torchvision","Pillow","numpy","transformers","timm","safetensors","open_clip_torch"]
    meta = {
        "tag": name, "repo": repo, "card": f"https://huggingface.co/{repo}",
        "revision": revision, "file_hashes": hashes,
        "license_reported_on_model_card": "Apache-2.0" if name == "bone-siglip" else "MIT",
        "provenance": ("Publisher describes fracture finetuning on Hemg/bone-fracture-detection. "
                       "Patient-level split/provenance of the source model was not independently verified."
                       if name == "bone-siglip" else
                       "Microsoft PMC-15M biomedical figure/caption model; loaded only its image tower. "
                       "Weights are the unmerged safetensors conversion by Hugging Face SFconvertbot in PR 28."),
        "source_data_or_conversion": ("https://huggingface.co/datasets/Hemg/bone-fracture-detection"
                                       if name == "bone-siglip" else f"https://huggingface.co/{repo}/discussions/28"),
        "device": torch.cuda.get_device_name(), "dtype": "float32", "batch_size": batch_size,
        "preprocess": "EXIF orientation, luminance, preserve aspect ratio, long edge resized to target, "
                      "center on black square, RGB, divide by 255, per-channel normalization",
        "edge": model.edge, "mean": model.mean, "std": model.std, "interpolation": model.resample.name,
        "features": model.pooling + ", L2-normalized",
        "shape": list(X.shape), "load_seconds_including_cache_verification": round(load_seconds, 3),
        "encoding_seconds_including_image_io": round(seconds, 3),
        "peak_allocated_pytorch_cuda_mib": round(torch.cuda.max_memory_allocated()/2**20, 1),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "npz_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "packages": {p:importlib.metadata.version(p) for p in packages},
    }
    (OUT / f"{name}.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(name, X.shape, seconds, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.batch_size < 1:
        parser.error("The recorded experiment requires CUDA and a positive batch size.")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    for tag in args.models:
        extract(tag, args.batch_size)
