"""Run DeepSeek-VL2-Tiny on the same held-out films and JSON task used to evaluate Qwen."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
from time import perf_counter

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

from evaluate_qwen_vl import MAX_TOKENS, PROMPT, ROOT, parse_answer


CHECKPOINT = "mlx-community/deepseek-vl2-tiny-4bit"
REVISION = "4772de7c997d0b115d14593051621bcf0fdcce84"
WEIGHT_DIGEST = "b7afb62609eae2cfa0d1b79cbf9e72c2c80fd376ba783e730d24d89e47ce5a35"


GROUNDING_PROMPT = (
    "Locate each visible lesion in this equine fetlock radiograph, considering all views. "
    "Use native detection boxes for <|ref|>suspected fracture line or detached fracture fragment<|/ref|>. "
    "If there is no visible fracture, return NO_FRACTURE. If you cannot judge, return UNCERTAIN."
)


def parse_grounding(raw):
    text = raw.strip().removesuffix("<｜end▁of▁sentence｜>").strip()
    if text.rstrip(".").upper() in ("UNCERTAIN", "NO_FRACTURE"):
        return {"status": "uncertain" if text.rstrip(".").upper() == "UNCERTAIN" else "negative", "boxes": []}
    # The upstream visual grounding decoder expresses XYXY on the [0,999] scale.
    segments = re.findall(r"<\|det\|>(.*?)<\|/det\|>", text, re.S)
    if (not segments or len(segments) != text.count("<|det|>") or len(segments) != text.count("<|/det|>")
            or "NO_FRACTURE" in text.upper() or "UNCERTAIN" in text.upper()):
        return {"status": "parse_error", "boxes": [], "error": "missing or contradictory grounding response"}
    boxes = []
    try:
        for segment in segments:
            group = json.loads(segment)
            if not isinstance(group, list) or not group:
                raise ValueError
            for box in group:
                if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int or not 0 <= v <= 999 for v in box):
                    raise ValueError
                boxes.append([v * 1000 / 999 for v in box])
    except ValueError:
        return {"status": "parse_error", "boxes": [], "error": "invalid native detection coordinates"}
    return parse_answer(json.dumps({"status": "positive", "boxes": boxes}))


def generate_for_image(model, processor, chat, pixels):
    batch = processor(text=chat, images=[pixels])
    tiles = batch["images"]
    crop_grid = batch["images_spatial_crop"]
    # MLX-VLM 0.8.0 indexes crop metadata up to the *tile* count. Zero rows
    # mark the end of images; this adds metadata only and leaves all pixels intact.
    assert crop_grid.shape == (1, 1, 2)
    assert tiles.shape[1] == 1 + int(crop_grid[0, 0].prod().item())
    grid = mx.pad(crop_grid, ((0, 0), (0, tiles.shape[1] - 1), (0, 0)))
    return generate(model, processor, chat, input_ids=batch["input_ids"],
                    pixel_values=tiles, mask=batch["attention_mask"],
                    images_spatial_crop=grid, images_seq_mask=batch["images_seq_mask"],
                    max_tokens=MAX_TOKENS, temperature=0.0, verbose=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", choices=("json", "grounding"), default="json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "predictions.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    with (args.model / "model.safetensors").open("rb") as weights:
        if hashlib.file_digest(weights, "sha256").hexdigest() != WEIGHT_DIGEST:
            raise ValueError("Unexpected DeepSeek checkpoint")
    mx.random.seed(42)
    model, processor = load(str(args.model))
    prompt = GROUNDING_PROMPT if args.protocol == "grounding" else PROMPT
    parse = parse_grounding if args.protocol == "grounding" else parse_answer
    chat = apply_chat_template(processor, load_config(args.model), prompt, num_images=1)
    assert chat.count("<image>") == 1 and prompt in chat
    params = json.loads((args.model / "processor_config.json").read_text())
    runtime = {
        "model": CHECKPOINT, "upstream": "deepseek-ai/deepseek-vl2-tiny", "revision": REVISION,
        "model_sha256": WEIGHT_DIGEST, "protocol": args.protocol, "prompt": prompt, "formatted_prompt": chat,
        "temperature": 0.0, "max_tokens": MAX_TOKENS, "seed": 42,
        "processor_config": params, "crop_metadata": "zero-padded to tile count for MLX-VLM 0.8.0",
        "chip": mx.device_info()["device_name"],
        "versions": {p: version(p) for p in ("mlx-vlm", "mlx", "transformers", "Pillow")},
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    inference_started = perf_counter()
    with destination.open("x", encoding="utf-8") as stream:
        for split in ("val", "test"):
            paths = sorted(p for p in (ROOT / "training/data/sporthorse_v2/images" / split).iterdir()
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not paths:
                raise ValueError(f"No images found in {split}")
            for path in paths:
                with Image.open(path) as image:
                    pixels = image.convert("RGB")
                started = perf_counter()
                answer = generate_for_image(model, processor, chat, pixels)
                row = dict(split=split, filename=path.name, size=list(pixels.size),
                           seconds=round(perf_counter() - started, 4), raw=answer.text,
                           prompt_tokens=answer.prompt_tokens, generated_tokens=answer.generation_tokens,
                           **parse(answer.text))
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(split, path.name, row["status"], row["seconds"], flush=True)
                mx.clear_cache()
    runtime["inference_seconds"] = round(perf_counter() - inference_started, 4)
    runtime["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 4)
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")


if __name__ == "__main__":
    main()
