from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


WEIGHTS = Path(__file__).resolve().parents[1] / "public/models/model-accurate.onnx"
WEIGHTS_SHA256 = "2174b21e3a81b740a07d7369ef5677a91fc097ffd4beb92b0d832a421d0825cb"
MODEL_ID = "YOLO11s-fresh-v2@" + WEIGHTS_SHA256[:12]
THRESHOLD = 0.15
INPUT_SIZE = 640


@lru_cache(maxsize=1)
def _session():
    weights = WEIGHTS.read_bytes()
    if sha256(weights).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("模型文件与登记版本不一致")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(weights, sess_options=options, providers=["CPUExecutionProvider"])


def _prepare(image):
    scale = min(INPUT_SIZE / image.width, INPUT_SIZE / image.height)
    width = max(1, int(image.width * scale + 0.5))
    height = max(1, int(image.height * scale + 0.5))
    left, top = (INPUT_SIZE - width) // 2, (INPUT_SIZE - height) // 2
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    canvas.paste(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR), (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(tensor), scale, left, top


def _decode(output, scale, left, top, width, height):
    if output.ndim != 3 or output.shape[:2] != (1, 5):
        raise ValueError("模型输出与单类别检测格式不一致")
    predictions = output[0].T
    candidates = predictions[np.isfinite(predictions).all(axis=1) & (predictions[:, 4] >= THRESHOLD)]
    centers, sizes = candidates[:, :2], candidates[:, 2:4]
    boxes = np.column_stack((centers - sizes / 2, centers + sizes / 2))
    areas = np.prod(np.maximum(0, sizes), axis=1)
    order = np.argsort(-candidates[:, 4], kind="stable")
    detections = []
    while order.size:
        current, remaining = order[0], order[1:]
        box = boxes[current]
        mapped = np.clip((box - [left, top, left, top]) / scale, 0, [width, height, width, height])
        if mapped[2] > mapped[0] and mapped[3] > mapped[1]:
            detections.append(dict(zip(("x1", "y1", "x2", "y2"), map(float, mapped))) | {"score": float(candidates[current, 4])})
        overlap = np.maximum(0, np.minimum(box[2:], boxes[remaining, 2:]) - np.maximum(box[:2], boxes[remaining, :2]))
        intersection = np.prod(overlap, axis=1)
        union = areas[current] + areas[remaining] - intersection
        order = remaining[intersection <= 0.7 * union]
    return detections


def detect(image: Image.Image) -> list[dict[str, float]]:
    tensor, scale, left, top = _prepare(image)
    session = _session()
    output, = session.run(None, {session.get_inputs()[0].name: tensor})
    return _decode(output, scale, left, top, image.width, image.height)
