from pathlib import Path

import numpy as np
from PIL import Image

from backend import detector


def test_decode_suppresses_duplicate_and_restores_non_square_coordinates():
    output = np.array([[[320, 320, 200], [320, 320, 320], [160, 160, 30],
                        [80, 80, 30], [0.9, 0.8, 0.10]]], dtype=np.float32)
    result = detector._decode(output, 2, 0, 160, 320, 160)
    assert len(result) == 1
    assert [result[0][axis] for axis in ("x1", "y1", "x2", "y2")] == [120, 60, 200, 100]


def test_shipped_weights_on_two_repository_radiographs():
    examples = Path(__file__).resolve().parents[2] / "training/data/sporthorse_v2/images/test"
    with Image.open(examples / "16-侧位.jpg") as image:
        boxes = detector.detect(image)
        assert len(boxes) == 1
        assert 0.20 < boxes[0]["score"] < 0.23
        assert 0 <= boxes[0]["x1"] < boxes[0]["x2"] <= image.width
        assert 0 <= boxes[0]["y1"] < boxes[0]["y2"] <= image.height
    with Image.open(examples / "23-LF-正位.jpg") as image:
        assert detector.detect(image) == []
