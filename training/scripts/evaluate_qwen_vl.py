"""Run a pinned MLX Qwen3-VL checkpoint on unannotated, full-frame split images."""
import argparse
import hashlib
import json
import math
import platform
import re
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
REVISION = "9c4f5209e57b31f4b9dfba735de3fb983739c9cc"
WEIGHT_SHA256 = "4750d95a2162829e127a94e83ac350d498d02070aab216c4687da48804a06ffb"
PROMPT = """Examine this equine fetlock radiograph for visible evidence of a fracture, including fracture lines or detached fracture fragments. Some images contain multiple views. Return JSON only, with fields "status" and "boxes". Use "positive" if you see a suspected fracture, "negative" if you see none, or "uncertain" if you cannot judge. For each suspected fracture, put one tight bounding box around the visible lesion, not the whole joint. Coordinates must be [xmin,ymin,xmax,ymax], integers on a 0 to 1000 scale relative to the full image, origin at the top left. With no localized fracture return an empty list for "boxes"."""
MAX_PIXELS = 1024 * 1024
MAX_TOKENS = 256


def parse_answer(raw):
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    try:
        answer = json.loads(text)
    except ValueError:
        return {"status": "parse_error", "boxes": [], "error": "invalid JSON"}
    if not isinstance(answer, dict) or answer.get("status") not in ("positive", "negative", "uncertain"):
        return {"status": "parse_error", "boxes": [], "error": "invalid status"}
    status = answer["status"]
    boxes = answer.get("boxes")
    if not isinstance(boxes, list) or (status != "positive" and boxes):
        return {"status": "parse_error", "boxes": [], "error": "invalid box list or conflicting status"}
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4
                or not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            return {"status": "parse_error", "boxes": [], "error": "invalid box coordinates"}
    return {"status": status, "boxes": boxes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Local snapshot of the pinned checkpoint")
    parser.add_argument("--adapter", type=Path, help="Optional MLX LoRA checkpoint directory")
    parser.add_argument("--images", type=Path, default=ROOT / "training/data/sporthorse_v2/images")
    parser.add_argument("--output", required=True, type=Path, help="New output directory or directory without predictions")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = args.output / "predictions.jsonl"
    if predictions.exists():
        raise FileExistsError(f"Predictions already exist at {predictions}")
    with (args.model / "model.safetensors").open("rb") as weights:
        if hashlib.file_digest(weights, "sha256").hexdigest() != WEIGHT_SHA256:
            raise ValueError("Checkpoint SHA-256 mismatch")
    mx.random.seed(42)
    model, processor = load(str(args.model), adapter_path=str(args.adapter) if args.adapter else None)
    processor.image_processor.max_pixels = MAX_PIXELS
    chat = apply_chat_template(processor, load_config(args.model), PROMPT, num_images=1)
    settings = {
        "model": MODEL_ID, "revision": REVISION, "model_sha256": WEIGHT_SHA256, "prompt": PROMPT,
        "method": "zero-shot; independent greedy generation for each full-frame input",
        "temperature": 0.0, "seed": 42, "max_tokens": MAX_TOKENS,
        "processor_min_pixels": processor.image_processor.min_pixels, "processor_max_pixels": MAX_PIXELS,
        "versions": {p: version(p) for p in ["mlx-vlm", "mlx", "mlx-metal", "transformers", "Pillow"]},
        "python": platform.python_version(), "platform": platform.platform(), "chip": mx.device_info()["device_name"],
    }
    if args.adapter:
        settings["method"] = "LoRA-adapted; independent greedy generation for each full-frame input"
        settings["adapter_files"] = {}
        for file in sorted(args.adapter.iterdir()):
            if file.is_file() and file.suffix in (".json", ".safetensors"):
                with file.open("rb") as data:
                    settings["adapter_files"][file.name] = hashlib.file_digest(data, "sha256").hexdigest()
    (args.output / "runtime.json").write_text(json.dumps(settings, indent=2) + "\n")
    with predictions.open("x", encoding="utf8") as output:
        for split in ["val", "test"]:
            paths = sorted(p for p in (args.images / split).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not paths:
                raise ValueError(f"No images found in {split}")
            for path in paths:
                with Image.open(path) as original:
                    pixels = original.convert("RGB")
                start = time.perf_counter()
                result = generate(model, processor, chat, image=[pixels], max_tokens=MAX_TOKENS, temperature=0.0)
                row = {
                    "split": split, "filename": path.name, "size": list(pixels.size), **parse_answer(result.text),
                    "raw": result.text, "seconds": round(time.perf_counter() - start, 4),
                    "prompt_tokens": result.prompt_tokens, "generated_tokens": result.generation_tokens,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(split, path.name, row["status"], row["seconds"], flush=True)
                mx.clear_cache()


if __name__ == "__main__":
    main()
