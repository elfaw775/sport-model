"""Run a pinned MLX Qwen-VL checkpoint on unannotated, full-frame split images."""
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

from qwen_checkpoints import CHECKPOINTS


ROOT = Path(__file__).resolve().parents[2]
PROMPT = """Examine this equine fetlock radiograph for visible evidence of a fracture, including fracture lines or detached fracture fragments. Some images contain multiple views. Return JSON only, with fields "status" and "boxes". Use "positive" if you see a suspected fracture, "negative" if you see none, or "uncertain" if you cannot judge. For each suspected fracture, put one tight bounding box around the visible lesion, not the whole joint. Coordinates must be [xmin,ymin,xmax,ymax], integers on a 0 to 1000 scale relative to the full image, origin at the top left. With no localized fracture return an empty list for "boxes"."""
MAX_PIXELS = 1024 * 1024
MAX_TOKENS = 256
PIXEL_PROMPT = PROMPT.replace("integers on a 0 to 1000 scale relative to the full image",
                              "absolute pixel coordinates in the resized image (width {width}, height {height})")


def parse_answer(raw, frame=None):
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
    limits = (*frame, *frame) if frame else (1000,) * 4
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4
                or not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= bound
                           for v, bound in zip(box, limits))
                or box[0] >= box[2] or box[1] >= box[3]):
            return {"status": "parse_error", "boxes": [], "error": "invalid box coordinates"}
    if frame:
        boxes = [[v * 1000 / bound for v, bound in zip(box, limits)] for box in boxes]
    return {"status": status, "boxes": boxes}


def mimo_final_text(raw):
    text = raw.strip()
    thought = re.fullmatch(r"<think>.*?</think>\s*(.*)", text, re.DOTALL)
    if thought:
        text = thought.group(1).strip()
    answer = re.fullmatch(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return answer.group(1) if answer else text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="2b")
    parser.add_argument("--bf16-compute", action="store_true", help="Cast FP16 base tensors to BF16 to match training")
    parser.add_argument("--coordinates", choices=["normalized", "resized-pixels"], default="normalized")
    parser.add_argument("--mimo-final-answer", action="store_true",
                        help="Allow up to 1024 tokens and read a final answer after closed reasoning tags")
    parser.add_argument("--model", required=True, type=Path, help="Local snapshot of the pinned checkpoint")
    parser.add_argument("--adapter", type=Path, help="Optional MLX LoRA checkpoint directory")
    parser.add_argument("--images", type=Path, default=ROOT / "training/data/sporthorse_v2/images")
    parser.add_argument("--output", required=True, type=Path, help="New output directory or directory without predictions")
    args = parser.parse_args()
    if args.mimo_final_answer and args.checkpoint != "mimo-7b":
        parser.error("--mimo-final-answer requires --checkpoint mimo-7b")
    max_tokens = 1024 if args.mimo_final_answer else MAX_TOKENS
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = args.output / "predictions.jsonl"
    if predictions.exists():
        raise FileExistsError(f"Predictions already exist at {predictions}")
    spec = CHECKPOINTS[args.checkpoint]
    spec.verify(args.model)
    mx.random.seed(42)
    model, processor = load(str(args.model), adapter_path=str(args.adapter) if args.adapter else None)
    if args.bf16_compute:
        model.set_dtype(mx.bfloat16, predicate=lambda dtype: dtype == mx.float16)
    processor.image_processor.max_pixels = MAX_PIXELS
    config = load_config(args.model)
    chat = apply_chat_template(processor, config, PROMPT, num_images=1)
    native = args.coordinates == "resized-pixels"
    settings = {
        "model": spec.model_id, "revision": spec.revision, "weight_files_sha256": spec.weights,
        "prompt": PIXEL_PROMPT if native else PROMPT, "coordinates": args.coordinates, "bf16_compute": args.bf16_compute,
        "method": "zero-shot; independent greedy generation for each full-frame input",
        "temperature": 0.0, "seed": 42, "max_tokens": max_tokens,
        "processor_min_pixels": processor.image_processor.min_pixels, "processor_max_pixels": MAX_PIXELS,
        "versions": {p: version(p) for p in ["mlx-vlm", "mlx", "mlx-metal", "transformers", "Pillow"]},
        "python": platform.python_version(), "platform": platform.platform(), "chip": mx.device_info()["device_name"],
    }
    if args.mimo_final_answer:
        settings["answer_protocol"] = "JSON final answer after optional completed <think> and <answer> tags"
    if "model.safetensors" in spec.weights:
        settings["model_sha256"] = spec.weights["model.safetensors"]
    if args.adapter:
        settings["method"] = "LoRA-adapted; independent greedy generation for each full-frame input"
        settings["adapter_files"] = {}
        for file in sorted(args.adapter.iterdir()):
            if file.is_file() and file.suffix in (".json", ".safetensors"):
                with file.open("rb") as data:
                    settings["adapter_files"][file.name] = hashlib.file_digest(data, "sha256").hexdigest()
    (args.output / "runtime.json").write_text(json.dumps(settings, indent=2) + "\n")
    inference_started = time.perf_counter()
    with predictions.open("x", encoding="utf8") as output:
        for split in ["val", "test"]:
            paths = sorted(p for p in (args.images / split).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not paths:
                raise ValueError(f"No images found in {split}")
            for path in paths:
                with Image.open(path) as original:
                    pixels = original.convert("RGB")
                frame = None
                if native:
                    image_processor = processor.image_processor
                    _, rows, cols = image_processor(pixels, return_tensors="np")["image_grid_thw"][0]
                    frame = (int(cols) * image_processor.patch_size, int(rows) * image_processor.patch_size)
                    prompt = PIXEL_PROMPT.format(width=frame[0], height=frame[1])
                    chat = apply_chat_template(processor, config, prompt, num_images=1)
                start = time.perf_counter()
                result = generate(model, processor, chat, image=[pixels], max_tokens=max_tokens, temperature=0.0)
                final = mimo_final_text(result.text) if args.mimo_final_answer else result.text
                row = {
                    "split": split, "filename": path.name, "size": list(pixels.size), **parse_answer(final, frame),
                    "raw": result.text, "seconds": round(time.perf_counter() - start, 4),
                    "prompt_tokens": result.prompt_tokens, "generated_tokens": result.generation_tokens,
                }
                if frame:
                    row["resized_size"] = frame
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(split, path.name, row["status"], row["seconds"], flush=True)
                mx.clear_cache()
    settings["inference_seconds"] = round(time.perf_counter() - inference_started, 4)
    settings["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 4)
    (args.output / "runtime.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
