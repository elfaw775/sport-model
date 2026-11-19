"""Produce whole-film embeddings from frozen, pinned Hugging Face backbones."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from study_paths import DATA as SOURCE, WORK
from hf_assets import MODELS, get_model


HERE = Path(__file__).resolve().parent
SIZE = 336
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PREPROCESS = ("PIL EXIF orientation, luminance then RGB, bicubic resize of long edge to 336, "
              "center on black 336x336 without cropping, [0,1], ImageNet RGB mean/std")


def transform(image, edge=SIZE, mean=MEAN, std=STD):
    gray = ImageOps.exif_transpose(image).convert("L")
    scale = edge / max(gray.size)
    size = tuple(max(1, round(axis * scale)) for axis in gray.size)
    canvas = Image.new("L", (edge, edge))
    canvas.paste(gray.resize(size, Image.Resampling.BICUBIC),
                 ((edge - size[0]) // 2, (edge - size[1]) // 2))
    return TF.normalize(TF.to_tensor(canvas.convert("RGB")), mean, std)


class FrozenEncoder:
    def __init__(self, name, device=None):
        folder = get_model(name)
        self.name = name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.edge, self.mean, self.std = SIZE, MEAN, STD
        if name in ("dinov2", "rad-dino"):
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(str(folder), local_files_only=True)
            self.pooling = "concatenate normalized last-layer CLS and normalized mean of all patch tokens"
            if name == "rad-dino":
                cfg = json.loads((folder / "preprocessor_config.json").read_text(encoding="utf-8"))
                self.edge, self.mean, self.std = cfg["crop_size"]["width"], cfg["image_mean"], cfg["image_std"]
                self.pooling = "normalized last-layer CLS"
        elif name == "resnet18":
            import timm
            from safetensors.torch import load_file
            self.model = timm.create_model("resnet18.a1_in1k", pretrained=False)
            self.model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
            self.model.reset_classifier(0)
            self.pooling = "global average pooled features before classification, L2 normalized"
        else:
            raise ValueError(name)
        self.model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, images):
        inputs = torch.stack([transform(im, self.edge, self.mean, self.std) for im in images]).to(self.device)
        if self.name in ("dinov2", "rad-dino"):
            hidden = self.model(pixel_values=inputs).last_hidden_state
            cls = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
            features = cls
            if self.name == "dinov2":
                patches = torch.nn.functional.normalize(hidden[:, 1:].mean(dim=1), dim=-1)
                features = torch.cat((cls, patches), dim=-1)
        else:
            features = torch.nn.functional.normalize(self.model(inputs), dim=-1)
        values = features.float().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite feature values")
        return values


def extract(model, split, outdir, batch_size):
    with (SOURCE / "split_manifest.tsv").open(encoding="utf-8", newline="") as inp:
        rows = [r for r in csv.DictReader(inp, delimiter="\t") if r["split"] == split]
    image_paths = {p.stem: p for p in (SOURCE / "images" / split).iterdir() if p.is_file()}
    ids = [r["image"] for r in rows]
    if len(set(ids)) != len(rows) or set(ids) != set(image_paths):
        raise ValueError(f"Manifest/image mismatch in {split}")
    labels = [{"pos": 1, "neg": 0}[row["category"]] for row in rows]
    values = []
    start = time.monotonic()
    for i in range(0, len(rows), batch_size):
        images = []
        for image_id in ids[i:i+batch_size]:
            with Image.open(image_paths[image_id]) as im:
                images.append(im.copy())
        values.append(model.encode(images))
    features = np.concatenate(values)
    output = outdir / f"{model.name}-{split}.npz"
    work = output.with_suffix(".tmp")
    with work.open("wb") as dest:
        np.savez_compressed(dest, X=features, ids=np.asarray(ids), y=np.asarray(labels, dtype=np.int64))
    work.replace(output)
    result = {"rows": len(rows), "dimension": features.shape[1], "seconds": round(time.monotonic()-start, 3)}
    print(model.name, split, result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    parser.add_argument("--outdir", type=Path, default=WORK / "features")
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the CUDA runtime.")
    torch.set_num_threads(4)
    for name in args.models:
        repo, rev, hashes = MODELS[name]
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        model = FrozenEncoder(name)
        meta = {"model": repo, "revision": rev, "hashes": hashes, "device": torch.cuda.get_device_name(0),
                "precision": "float32", "preprocess": PREPROCESS.replace("336", str(model.edge)).replace("ImageNet RGB", "model RGB"),
                "mean": model.mean, "std": model.std,
                "pooling": model.pooling, "batch_size": args.batch_size, "torch": str(torch.__version__),
                "load_seconds": round(time.monotonic()-start, 3)}
        for split in args.splits:
            meta[split] = extract(model, split, args.outdir, args.batch_size)
        meta["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        report = args.outdir / f"{name}-{'-'.join(args.splits)}.json"
        report.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
