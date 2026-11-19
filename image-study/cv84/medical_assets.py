"""Download immutable Hugging Face files, with pinned Git blob or LFS SHA-256 digests."""
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


MODELS = {
    "bone-siglip": ("prithivMLmods/Bone-Fracture-Detection", "7ccaf775b87236a613265389e05c3f360f1215db", {
        "config.json": "7bf61c1253e904d69899d1b9504fbf26905b604e",
        "preprocessor_config.json": "2e52d8e8492b5c496ae04c37bfa09760469fb18b",
        "model.safetensors": "1fe991a340c25c67413f22305e8a5f3e928a460eba863d5fd21f8f3ae80a89dd",
    }),
    "biomedclip": ("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", "976347208356df345b35d846ae09dd59d5c53668", {
        "open_clip_config.json": "127246c8e2bdbec4fff72874d2814f92ac77a403",
        "open_clip_pytorch_model.safetensors": "254094112d1fcbe0ceaf0632f5a792b41ca4474a9a9f642e7e711d1a9275d01b",
    }),
}
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_paths import CACHE as PROJECT_CACHE

CACHE = PROJECT_CACHE / "hf/cv84"


def checked(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as inp:
        if len(expected) == 64:
            return hashlib.file_digest(inp, "sha256").hexdigest() == expected
        return hashlib.sha1(f"blob {path.stat().st_size}\0".encode() + inp.read()).hexdigest() == expected


def download(name):
    repo, revision, files = MODELS[name]
    folder = CACHE / name / revision
    folder.mkdir(parents=True, exist_ok=True)
    for filename, expected in files.items():
        dest = folder / filename
        if checked(dest, expected):
            continue
        part = folder / f"{filename}.{os.getpid()}.part"
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        print(f"Fetching {name} {filename}", flush=True)
        try:
            for attempt in range(3):
                try:
                    with requests.get(url, stream=True, timeout=(15, 90)) as reply:
                        reply.raise_for_status()
                        with part.open("wb") as out:
                            for chunk in reply.iter_content(1024 * 1024):
                                out.write(chunk)
                    if not checked(part, expected):
                        raise OSError(f"Digest mismatch for {name}/{filename}")
                    part.replace(dest)
                    break
                except (OSError, requests.RequestException) as exc:
                    if isinstance(exc, requests.HTTPError) and exc.response.status_code in (401,403,404):
                        raise
                    if attempt == 2:
                        raise
                    print(f"Retry {name} {type(exc).__name__}", flush=True)
                    time.sleep(2)
        finally:
            part.unlink(missing_ok=True)
    print(f"Verified {name} {revision}", flush=True)
    return folder


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, MODELS))
