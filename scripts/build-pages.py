"""Assemble the public workbench and the previous YOLO viewer into the Pages artifact."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.demo_library import demo_library


def build():
    output = ROOT / "work/pages-build/sport-model"
    legacy = ROOT / "out"
    if not (legacy / "index.html").is_file():
        raise SystemExit("Build the legacy Next.js export first (see README.md).")
    if output.exists():
        if not (output / ".nojekyll").is_file():
            raise SystemExit(f"Previous output directory cannot be identified: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / ".nojekyll").touch()
    shutil.copytree(legacy, output / "legacy")
    static = ROOT / "backend/static"
    shutil.copytree(static, output / "static", ignore=shutil.ignore_patterns("index.html"))
    for name in ("boot.js", "service-worker.js", "cases.js", "review.js", "inference.js"):
        shutil.copy2(ROOT / "backend/pages" / name, output)
    version = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, text=True).strip()
    index = (static / "index.html").read_text().replace('./static/app.js', f'./boot.js?v={version}')
    (output / "index.html").write_text(index.replace('<meta charset="utf-8">', f'<meta charset="utf-8">\n  <meta name="workbench-version" content="{version}">'))
    shutil.copytree(ROOT / "backend/models", output / "models")
    runtime = ROOT / "node_modules/onnxruntime-web/dist"
    ort = output / "ort"
    ort.mkdir()
    for name in ("ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"):
        shutil.copy2(runtime / name, ort)
    shutil.copy2(ROOT / "backend/pages/ONNXRuntime-LICENSE.txt", ort / "LICENSE.txt")
    catalog, files = demo_library(ROOT / "training/data/sporthorse_v2")
    if len(catalog) != 84:
        raise SystemExit(f"Pages requires 84 study films, found {len(catalog)}")
    samples = output / "samples"
    samples.mkdir()
    names = {}
    for item in catalog:
        file = item["filename"]
        names[files[file]] = file
        shutil.copy2(files[file], samples / file)
        del item["url"]
    aliases = {alias: names[source] for alias, source in files.items() if alias != names[source]}
    (samples / "catalog.json").write_text(json.dumps({"catalog":catalog,"aliases":aliases}, ensure_ascii=False))
    print(f"Built {output} with {len(catalog)} radiographs ({version})")


if __name__ == "__main__":
    build()
