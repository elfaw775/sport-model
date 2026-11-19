from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .onnx_models import verified_session


WEIGHTS = Path(__file__).resolve().parent / "models/yolo11s-fetlock-640.onnx"
WEIGHTS_SHA256 = "61286f939c6a6c4a591d5034e33794ba0af99b5f1f29aac279bb15974c025b0a"
MODEL_ID = "YOLO11s-fetlock-640@" + WEIGHTS_SHA256[:12]
THRESHOLD = 0.20
INPUT_SIZE = 640


def _prepare(image):
    scale = min(INPUT_SIZE / image.width, INPUT_SIZE / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    left, top = (INPUT_SIZE - width) // 2, (INPUT_SIZE - height) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    # Match Ultralytics' square letterbox used to select the operating threshold.
    canvas[top:top + height, left:left + width] = cv2.resize(
        np.asarray(image.convert("RGB")), (width, height), interpolation=cv2.INTER_LINEAR,
    )
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    box_scale = (width / image.width, height / image.height) * 2
    return np.ascontiguousarray(tensor), box_scale, left, top


def _decode(output, scale, left, top, width, height, threshold=THRESHOLD):
    if output.ndim != 3 or output.shape[:2] != (1, 5):
        raise ValueError("模型输出与单类别检测格式不一致")
    predictions = output[0].T
    candidates = predictions[np.isfinite(predictions).all(axis=1) & (predictions[:, 4] >= threshold)]
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
    session = verified_session(WEIGHTS, WEIGHTS_SHA256)
    output, = session.run(None, {session.get_inputs()[0].name: tensor})
    return _decode(output, scale, left, top, image.width, image.height)
