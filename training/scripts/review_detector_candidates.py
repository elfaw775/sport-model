"""Test whether a frozen Qwen second read removes false boxes from the corrected YOLO."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image, ImageDraw

from compare_detectors import WORK
from qwen_checkpoints import CHECKPOINTS
from score_qwen_vl import COMPARISON, load_yolo


PROMPT = """This is a crop from an equine fetlock radiograph. A detector flagged the area inside the red outline. Inspect that area for a visible fracture line or a detached fracture fragment. Reply with exactly one word: positive if you see supporting evidence, negative if you confidently see no such evidence, or uncertain if you cannot judge."""


def candidate_crop(image, corners):
    x1, y1, x2, y2 = corners
    width, height = x2 - x1, y2 - y1
    left, top = max(0, int(x1 - width / 2)), max(0, int(y1 - height / 2))
    right, bottom = min(image.width, int(x2 + width / 2 + 1)), min(image.height, int(y2 + height / 2 + 1))
    crop = image.crop((left, top, right, bottom))
    ImageDraw.Draw(crop).rectangle((x1-left, y1-top, x2-left, y2-top), outline="red", width=1)
    return crop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    CHECKPOINTS["2b"].verify(args.model)
    model, processor = load(str(args.model))
    processor.image_processor.max_pixels = 1024**2
    chat = apply_chat_template(processor, load_config(args.model), PROMPT, num_images=1)
    mx.random.seed(42)
    baseline = json.loads(COMPARISON.with_name("metrics.json").read_text())["yolo"]
    cutoff = baseline["val_selected_cutoff"]
    with ZipFile(COMPARISON) as archive, args.output.open("x", encoding="utf8") as output:
        for split in ("val", "test"):
            proposals = load_yolo(archive, "yolo", split, cutoff)
            for name, (_, boxes) in sorted(proposals.items()):
                reviewed = []
                kept = []
                with Image.open(WORK / "dataset/images" / split / name) as source:
                    image = source.convert("RGB")
                for box in boxes:
                    corners = [box[key] for key in ("x1", "y1", "x2", "y2")]
                    start = perf_counter()
                    result = generate(model, processor, chat, image=[candidate_crop(image, corners)],
                                      temperature=0.0, max_tokens=12)
                    answer = result.text.strip().lower().rstrip(".")
                    reviewed.append({"source_box": box, "raw": result.text,
                                     "seconds": round(perf_counter()-start, 3)})
                    if answer != "negative":
                        dims = (image.width, image.height) * 2
                        kept.append([value * 1000 / dim for value, dim in zip(corners, dims)])
                    mx.clear_cache()
                row = {"split": split, "filename": name, "boxes": kept, "reviewed": reviewed,
                       "status": "positive" if kept else "negative"}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()


if __name__ == "__main__":
    main()
