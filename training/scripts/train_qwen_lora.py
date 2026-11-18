"""One fixed epoch of completion-only QLoRA on the existing 59-horse training split."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer.common.model import prepare_model_for_training
from mlx_vlm.trainer.common.utils import save_trainable_weights
from mlx_vlm.utils import load_config, prepare_inputs
from PIL import Image

from compare_detectors import WORK
from evaluate_qwen_vl import MAX_PIXELS, PROMPT
from qwen_checkpoints import CHECKPOINTS


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def training_rows():
    rows = []
    for image in sorted((WORK / "dataset/images/train").iterdir()):
        label = WORK / "dataset/labels/train" / (image.stem + ".txt")
        boxes = []
        for line in label.read_text().splitlines():
            cls, cx, cy, w, h = map(float, line.split())
            assert cls == 0
            coords = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
            boxes.append([min(1000, max(0, round(v*1000))) for v in coords])
        answer = {"status": "positive" if boxes else "negative", "boxes": boxes}
        rows.append((image, label, answer))
    assert len(rows) == 59 and sum(len(ans["boxes"]) for _, _, ans in rows) == 57
    random.Random(42).shuffle(rows)
    return rows


def completion_loss(model, prefix_ids, pixels, grid, target_ids):
    sequence = mx.concatenate((prefix_ids, target_ids[:, :-1]), axis=1)
    logits = model(sequence, pixel_values=pixels, image_grid_thw=grid).logits
    # At the last prefix position predict the first answer token. End the target with im_end.
    answer_logits = logits[:, prefix_ids.shape[1]-1:, :].astype(mx.float32)
    return nn.losses.cross_entropy(answer_logits, target_ids, reduction="mean")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=("2b", "7b", "8b"), default="2b")
    parser.add_argument("--bf16-compute", action="store_true", help="Cast FP16 base tensors to BF16 for numerical stability")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "adapter").exists():
        raise FileExistsError("Choose an output directory without an existing adapter")
    spec = CHECKPOINTS[args.checkpoint]
    spec.verify(args.model)
    mx.random.seed(42)
    model, processor = load(str(args.model))
    if args.bf16_compute:
        model.set_dtype(mx.bfloat16, predicate=lambda dtype: dtype == mx.float16)
    processor.image_processor.max_pixels = MAX_PIXELS
    prompt = apply_chat_template(processor, load_config(args.model), PROMPT, num_images=1)
    depth = len(model.language_model.model.layers)
    targets = [f"language_model.model.layers.{layer}.self_attn.{proj}_proj"
               for layer in range(depth - 4, depth) for proj in ("q", "v")]
    model = prepare_model_for_training(model, target_modules=targets, train_type="lora",
                                       lora_rank=8, lora_alpha=16, lora_dropout=0)
    model.train()
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert set(trainable) == {module + "." + name for module in targets for name in ("lora_a", "lora_b")}
    count = sum(array.size for array in trainable.values())
    optimizer = optim.AdamW(learning_rate=0.00015, weight_decay=0)
    calc_grad = nn.value_and_grad(model, completion_loss)
    rows = training_rows()
    metadata = {"base_model": spec.model_id, "base_revision": spec.revision, "weight_files_sha256": spec.weights, "rank": 8, "alpha": 16,
                "bf16_compute": args.bf16_compute, "modules": targets, "trainable_parameters": count, "epochs": 1, "batch_size": 1,
                "optimizer": "AdamW", "learning_rate": 0.00015, "weight_decay": 0, "seed": 42,
                "loss": "mean cross-entropy over assistant JSON and end token only",
                "max_pixels": MAX_PIXELS, "prompt": PROMPT,
                "inputs": [{"filename": im.name, "image_sha256": digest(im), "label_sha256": digest(lb)}
                           for im, lb, _ in rows]}
    if "model.safetensors" in spec.weights:
        metadata["base_sha256"] = spec.weights["model.safetensors"]
    started = time.perf_counter()
    with (args.output / "loss.jsonl").open("w") as log:
        for step, (image, _, answer) in enumerate(rows, 1):
            with Image.open(image) as film:
                batch = prepare_inputs(processor, images=[film.convert("RGB")], prompts=prompt)
            answer_text = json.dumps(answer, separators=(",", ":"))
            tokens = processor.tokenizer.encode(answer_text, add_special_tokens=False) + [processor.tokenizer.eos_token_id]
            loss, grads = calc_grad(model, batch["input_ids"], batch["pixel_values"],
                                    batch["image_grid_thw"], mx.array([tokens]))
            optimizer.update(model, grads)
            mx.eval(loss, model.trainable_parameters(), optimizer.state)
            value = loss.item()
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite training loss at step {step}")
            event = {"step": step, "loss": value, "target_tokens": len(tokens), "prefix_tokens": batch["input_ids"].shape[1]}
            log.write(json.dumps(event) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print(f"{step}/{len(rows)} loss={value:.4f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
            mx.clear_cache()
    updated = dict(tree_flatten(model.trainable_parameters()))
    # lora_b starts at zero; verify all eight modules received an update.
    delta = {key: mx.abs(arr).max().item() for key, arr in updated.items() if key.endswith(".lora_b")}
    assert len(delta) == 8 and all(v > 0 for v in delta.values())
    adapter = args.output / "adapter/adapters.safetensors"
    save_trainable_weights(model, adapter)
    metadata.update(lora_b_max_abs_after=delta, training_seconds=time.perf_counter()-started,
                    adapter_sha256=digest(adapter), peak_memory_gb=mx.get_peak_memory()/1e9)
    (args.output / "training.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
