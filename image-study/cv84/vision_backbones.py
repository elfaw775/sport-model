"""Extract frozen, whole-film features for the exploratory 84-film CV study."""
import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import requests
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from study_paths import CV
from hf_assets import CACHE, checked
from hf_features import MEAN, SOURCE, STD, transform


CHECKPOINTS = {
    "dinov2reg-base": (
        "facebook/dinov2-with-registers-base", "a1d738ccfa7ae170945f210395d99dde8adb1805",
        {
            "config.json": "95b7719e73a5645db4da30fad76b8c9e738caf9b",
            "model.safetensors": "7a6f7b3b9fa4b8732e707476a03cd6cdce210048582f21aafb7991c17d98e362",
        },
    ),
    "convnextv2-tiny": (
        "timm/convnextv2_tiny.fcmae_ft_in22k_in1k", "b1dd46230e80bf4cc3fa0c3c905db2c3ec53a817",
        {
            "config.json": "eb15892e3925e8df76e58e3ef68d569728896905",
            "model.safetensors": "6652fd90fc9c23977659e58515778e16fbbcd43f0b01fc693089cebe6d64c2a9",
        },
    ),
}


def verified_folder(tag):
    repo, rev, hashes = CHECKPOINTS[tag]
    folder = CACHE / repo.replace("/", "--") / rev
    folder.mkdir(parents=True, exist_ok=True)
    for name, digest in hashes.items():
        dest = folder / name
        if checked(dest, digest):
            continue
        part = dest.with_suffix(f".{os.getpid()}.part")
        try:
            for attempt in range(3):
                try:
                    with requests.get(f"https://huggingface.co/{repo}/resolve/{rev}/{name}",
                                      stream=True, timeout=(15, 45)) as response:
                        response.raise_for_status()
                        with part.open("wb") as out:
                            for chunk in response.iter_content(1024 * 1024):
                                out.write(chunk)
                    if not checked(part, digest):
                        raise OSError(f"Checksum failed for {tag}/{name}")
                    part.replace(dest)
                    break
                except (requests.RequestException, OSError) as err:
                    if attempt == 2 or isinstance(err, requests.HTTPError) and err.response.status_code in (401, 403, 404):
                        raise
                    print(f"Retry {attempt + 1}: {tag}/{name}: {type(err).__name__}", flush=True)
                    sleep(2)
        finally:
            part.unlink(missing_ok=True)
    print(f"Verified {tag} at {rev}", flush=True)
    return folder


class Encoder:
    def __init__(self, tag, folder, device="cuda"):
        self.device = torch.device(device)
        self.tag = tag
        if tag == "convnextv2-tiny":
            import timm
            from safetensors.torch import load_file

            self.model = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=False)
            self.model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
            self.model.reset_classifier(0)
            self.pooling = "Classifier removed; normalized, global average pooled representation"
            self.edge = 288
        else:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(str(folder), local_files_only=True)
            self.pooling = "Concatenate separately unit-normalized CLS and mean patch states, excluding register tokens"
            self.edge = 336
        self.model.eval().requires_grad_(False).to(self.device)

    @torch.inference_mode()
    def encode(self, images):
        batch = torch.stack([transform(im, self.edge, MEAN, STD) for im in images]).to(self.device)
        if self.tag == "convnextv2-tiny":
            output = torch.nn.functional.normalize(self.model(batch), dim=1)
        else:
            states = self.model(pixel_values=batch).last_hidden_state
            start = 1 + self.model.config.num_register_tokens
            expected_patches = (self.edge // self.model.config.patch_size) ** 2
            if states.shape[1] - start != expected_patches:
                raise ValueError("Unexpected number of patch tokens")
            cls = torch.nn.functional.normalize(states[:, 0], dim=1)
            patches = torch.nn.functional.normalize(states[:, start:].mean(dim=1), dim=1)
            output = torch.cat((cls, patches), dim=1)
        return output.float().cpu().numpy()


def records():
    manifest = SOURCE / "split_manifest.tsv"
    with manifest.open(encoding="utf-8", newline="") as inp:
        rows = list(csv.DictReader(inp, delimiter="\t"))
    images = {(p.parent.name, p.stem): p for p in (SOURCE / "images").glob("*/*") if p.is_file()}
    ids = [r["image"] for r in rows]
    if len(set(ids)) != len(rows) or len(rows) != 84:
        raise ValueError("Unexpected manifest")
    return ids, np.array([{"pos": 1, "neg": 0}[r["category"]] for r in rows]), [
        images[r["split"], r["image"]] for r in rows
    ]


def extract(tag, folder, outdir, batch_size):
    from importlib.metadata import version

    repo, rev, hashes = CHECKPOINTS[tag]
    ids, labels, paths = records()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    encoder = Encoder(tag, folder)
    torch.cuda.synchronize()
    elapsed_load = perf_counter() - start
    start = perf_counter()
    batches = []
    for i in range(0, len(paths), batch_size):
        images = []
        for path in paths[i:i+batch_size]:
            with Image.open(path) as inp:
                images.append(inp.copy())
        batches.append(encoder.encode(images))
    torch.cuda.synchronize()
    elapsed_features = perf_counter() - start
    X = np.concatenate(batches)
    if not np.isfinite(X).all() or len(ids) != len(X):
        raise ValueError("Invalid output matrix")
    dest = outdir / f"{tag}.npz"
    tmp = dest.with_suffix(".tmp")
    with tmp.open("wb") as out:
        np.savez_compressed(out, X=X, ids=np.array(ids), y=labels)
    tmp.replace(dest)
    data = {
        "name": tag, "card": f"https://huggingface.co/{repo}", "revision": rev, "file_hashes": hashes,
        "file_hash_algorithms": {"config.json": "git-blob-sha1", "model.safetensors": "sha256"},
        "features_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((SOURCE / "split_manifest.tsv").read_bytes()).hexdigest(),
        "shape": list(X.shape), "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "transformers": version("transformers"), "timm": version("timm"), "batch_size": batch_size,
        "precision": "float32", "input_side": encoder.edge, "normalize_mean": MEAN, "normalize_std": STD,
        "preprocess": "EXIF orientation; luminance; bicubic scale of long edge; centered black square padding without cropping; grayscale replicated to RGB; divide by 255, normalize channels",
        "pooling": encoder.pooling, "weights_frozen": True, "images": len(X),
        "load_seconds": round(elapsed_load, 3), "extraction_seconds": round(elapsed_features, 3),
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated()/2**20, 1),
    }
    dest.with_suffix(".json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(tag, X.shape, data["extraction_seconds"], "seconds", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=CHECKPOINTS, default=list(CHECKPOINTS))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--outdir", type=Path, default=CV / "features")
    args = parser.parse_args()
    if args.batch_size < 1 or not torch.cuda.is_available():
        parser.error("Positive batch size and CUDA required")
    args.outdir.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {tag: pool.submit(verified_folder, tag) for tag in args.models}
        for tag, future in pending.items():
            extract(tag, future.result(), args.outdir, args.batch_size)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
