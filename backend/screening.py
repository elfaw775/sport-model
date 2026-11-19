"""Whole-film fetlock lesion screening with the published 84-film DINOv2 head."""
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .onnx_models import verified_session


WEIGHTS = Path(__file__).resolve().parent / "models/dinov2-fetlock-84.onnx"
EDGE = 336
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@lru_cache(maxsize=1)
def _model():
    manifest = json.loads(WEIGHTS.with_suffix(".json").read_text(encoding="utf-8"))
    model_id, digest, threshold = manifest["model_id"], manifest["sha256"], float(manifest["threshold"])
    if (manifest["file"] != WEIGHTS.name or model_id != "DINOv2-Small-refit84@" + digest[:12]
            or not 0 < threshold < 1):
        raise ValueError("整片筛查模型登记信息无效")
    return verified_session(WEIGHTS, digest), model_id, threshold


def _prepare(image):
    # Match image-study/hf_features.py, including the original aspect ratio and black padding.
    gray = ImageOps.exif_transpose(image).convert("L")
    scale = EDGE / max(gray.size)
    width, height = (max(1, round(value * scale)) for value in gray.size)
    canvas = Image.new("L", (EDGE, EDGE))
    canvas.paste(gray.resize((width, height), Image.Resampling.BICUBIC),
                 ((EDGE - width) // 2, (EDGE - height) // 2))
    values = np.asarray(canvas.convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(((values - MEAN) / STD).transpose(2, 0, 1)[None])


def classify(image):
    session, model_id, threshold = _model()
    output, = session.run(["film_score"], {"pixel_values": _prepare(image)})
    score = float(output.item())
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("整片筛查模型输出无效")
    return {"model": model_id, "score": score, "threshold": threshold, "predicted_positive": score >= threshold}
