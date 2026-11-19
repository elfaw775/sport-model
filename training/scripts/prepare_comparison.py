"""Stage the archived split with corrected CVAT geometry, without editing the source data."""
import argparse
import csv
import filecmp
import json
from pathlib import Path
import shutil
import subprocess
import sys
from collections import Counter


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "training/data/sporthorse_v2"
WORK = ROOT / "work/model-comparison"


def prepare(work=WORK):
    work = Path(work).resolve()
    converted = work / "rotation-aware"
    subprocess.run([sys.executable, str(Path(__file__).with_name("coco2yolo.py")),
                    str(ROOT / "training/source/coco/instances_default.json"),
                    "--out", str(converted), "--keep", "病变"], check=True)
    dataset = work / "dataset"
    with (SOURCE / "split_manifest.tsv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len({row["image"] for row in rows}) == len(rows) == 84
    assert Counter(row["split"] for row in rows) == dict(train=59, val=13, test=12)
    counts = Counter()
    for row in rows:
        stem, split = row["image"], row["split"]
        matches = [file for file in (SOURCE / "images" / split).iterdir() if file.stem == stem]
        source, = matches
        label = (converted / (stem + ".txt")).read_text(encoding="utf-8") if row["category"] == "pos" else ""
        assert len(label.splitlines()) == int(row["class0_boxes"])
        counts[split] += len(label.splitlines())
        path = dataset / "images" / split / source.name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            # Some loaders can repair a malformed JPEG in-place. Do not hard-link the archive.
            shutil.copyfile(source, path)
        elif not filecmp.cmp(source, path, shallow=False):
            raise RuntimeError(f"Unexpected image already present: {path}")
        output = dataset / "labels" / split / (stem + ".txt")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(label, encoding="utf-8")
    assert counts == dict(train=57, val=12, test=12)
    for split in counts:
        expected = {row["image"] for row in rows if row["split"] == split}
        for folder, allowed in (("images", {".jpg", ".jpeg", ".png"}), ("labels", {".txt"})):
            found = {p.stem for p in (dataset / folder / split).iterdir() if p.suffix.lower() in allowed}
            if found != expected:
                raise ValueError(f"Unexpected {folder}/{split} files")
    (dataset / "data.yaml").write_text("path: " + json.dumps(str(dataset), ensure_ascii=False)
        + "\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {0: lesion}\n", encoding="utf-8")
    print(f"Ready: {dataset}; boxes per split: {dict(counts)}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    prepare(parser.parse_args().work)


if __name__ == "__main__":
    main()
