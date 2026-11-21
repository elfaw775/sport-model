"""Project radiograph catalog shared by the local service and Pages export."""
import csv
import re


def demo_library(dataset):
    images = dataset / "images"
    files = {name: images / "test" / name for name in ("16-侧位.jpg", "23-LF-正位.jpg")}
    catalog = []
    manifest = dataset / "split_manifest.tsv"
    if not manifest.is_file():
        return catalog, files

    with manifest.open(encoding="utf-8", newline="") as table:
        rows = sorted(csv.DictReader(table, delimiter="\t"), key=lambda row: row["image"])
    installed = {(path.parent.name, path.stem): path for split in ("train", "val", "test")
                 for path in (images / split).glob("*") if path.suffix in {".jpg", ".png"} and path.is_file()}
    # Number the full manifest so sparse checkouts keep the same public image IDs.
    for index, row in enumerate(rows, 1):
        source = installed.get((row["split"], row["image"]))
        if source is None:
            continue
        filename = f"XR-{index:03d}{source.suffix}"
        view = re.search(r"-(正位|侧位|斜位)(?:[\s，,]|$)", source.stem)
        files[filename] = source
        catalog.append({"filename": filename, "projection": view[1] if view else "",
                        "horse_name": f"演示马 {index:03d}", "url": f"/api/demo-images/{filename}"})
    return catalog, files
