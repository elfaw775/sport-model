"""Fetch immutable public Hugging Face checkpoints and verify each file."""
import hashlib
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests


from study_paths import CACHE as PROJECT_CACHE

CACHE = PROJECT_CACHE / "hf"
MODELS = {
    "dinov2": ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056", {
        "config.json": "5664b325e6258d3960fad8c4c1cff958f3cc2272",
        "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    }),
    "resnet18": ("timm/resnet18.a1_in1k", "491b427b45c94c7fb0e78b5474cc919aff584bbf", {
        "config.json": "416530229513d02f9c74c6a11b8e3ea9d0bdaef0",
        "model.safetensors": "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612",
    }),
    "rad-dino": ("microsoft/rad-dino", "110cbc18d5133582e320b43d53bf5c44e410c936", {
        "config.json": "8a941aae08c41bf5fe56f7afe9ff8f90d88f0001",
        "preprocessor_config.json": "cd049e0178ed390676aaa36bcfd90b81679df22b",
        "model.safetensors": "dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae",
    }),
}


def checked(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        if len(expected) == 64:
            return hashlib.file_digest(stream, "sha256").hexdigest() == expected
        return hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0" + stream.read()).hexdigest() == expected


def get_model(key):
    repo, rev, files = MODELS[key]
    folder = CACHE / repo.replace("/", "--") / rev
    folder.mkdir(parents=True, exist_ok=True)
    for name, digest in files.items():
        dest = folder / name
        if checked(dest, digest):
            continue
        part = folder / f"{name}.{os.getpid()}.part"
        try:
            for attempt in range(3):
                try:
                    with requests.get(f"https://huggingface.co/{repo}/resolve/{rev}/{name}",
                                      stream=True, timeout=(12, 45)) as response:
                        response.raise_for_status()
                        with part.open("wb") as output:
                            for chunk in response.iter_content(1024 * 1024):
                                output.write(chunk)
                    if not checked(part, digest):
                        raise OSError(f"Digest mismatch for {key}/{name}")
                    part.replace(dest)
                    break
                except (requests.RequestException, OSError) as err:
                    if attempt == 2 or (isinstance(err, requests.HTTPError) and err.response.status_code in (401, 403, 404)):
                        raise
                    print(f"Retrying {key}/{name} ({type(err).__name__})", flush=True)
                    time.sleep(2)
        finally:
            part.unlink(missing_ok=True)
    print(f"Verified {key} {rev}", flush=True)
    return folder


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(get_model, MODELS))
