"""Run a classifier experiment with a temporary, verified copy of the public film dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "elfaw775/sport-model"
COMMIT = "18a133f038eca0ea67c40effd71fb7b62c014f79"
PREFIX = "training/data/sporthorse_v2/"


def get(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={"User-Agent":"equine-film-study"}),timeout=45) as reply:
                return reply.read()
        except OSError:
            if attempt == 2:
                raise


def run(command, splits, *, legacy_labels=False):
    tree = json.loads(get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    if tree["truncated"]:
        raise ValueError("Incomplete repository tree")
    objects = {r["path"][len(PREFIX):]:r for r in tree["tree"] if r["type"]=="blob" and r["path"].startswith(PREFIX)}
    with tempfile.TemporaryDirectory(prefix="sport-horse-xray-films-") as tmp:
        root=Path(tmp).resolve()
        def fetch(rel):
            dest=(root/rel).resolve()
            if not dest.is_relative_to(root):
                raise ValueError("Unexpected repository path")
            b=get(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"+quote(PREFIX+rel))
            if hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()!=objects[rel]["sha"]:
                raise ValueError("Input checksum changed: "+rel)
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(b)
            return dest
        manifest=fetch("split_manifest.tsv")
        with manifest.open(encoding="utf-8",newline="") as fh:
            expected={(r["split"],r["image"]) for r in csv.DictReader(fh,delimiter="\t") if r["split"] in splits}
        images=[rel for rel in objects if len(Path(rel).parts)==3 and Path(rel).parts[0]=="images"
                and Path(rel).parts[1] in splits and Path(rel).suffix.lower() in {".jpg",".jpeg",".png"}]
        if {(Path(rel).parts[1],Path(rel).stem) for rel in images}!=expected or len(images)!=len(expected):
            raise ValueError("Published filenames differ from the pinned manifest")
        files = list(images)
        if legacy_labels:
            labels = [Path("labels", Path(rel).parts[1], Path(rel).stem + ".txt").as_posix() for rel in images]
            if not all(rel in objects for rel in labels):
                raise ValueError("A pinned legacy label is missing")
            files.extend(labels)
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(fetch,files))
        print(f"{len(images)} verified films available to this process",flush=True)
        return subprocess.run(command,env={**os.environ,"SPORT_XRAY_DATA":str(root)}).returncode


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits",choices=("train","val","test"),nargs="+",default=["train","val","test"])
    ap.add_argument("--legacy-labels",action="store_true",help="Also verify and stage the original YOLO box labels")
    ap.add_argument("command",nargs=argparse.REMAINDER)
    args=ap.parse_args()
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    if not command:
        ap.error("Supply a training command after --")
    raise SystemExit(run(command,args.splits,legacy_labels=args.legacy_labels))
