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


def test_released_onnx_matches_reference_pt_predictions_on_all_evaluation_films():
    import json
    from zipfile import ZipFile

    root = Path(__file__).resolve().parents[2] / "training"
    assert detector.THRESHOLD == .20
    with ZipFile(root / "results/comparison-2026-11-18/predictions.zip") as reference:
        for split, count in (("val", 13), ("test", 12)):
            predictions = json.loads(reference.read(f"yolo/{split}_predictions.json"))
            assert len(predictions) == count
            for name, all_boxes in predictions.items():
                expected = [box for box in all_boxes if box["score"] >= .20]
                with Image.open(root / "data/sporthorse_v2/images" / split / name) as image:
                    actual = detector.detect(image)
                assert len(actual) == len(expected), name
                for box, target in zip(actual, expected):
                    assert abs(box["score"] - target["score"]) < 1e-4, name
                    for key in ("x1", "y1", "x2", "y2"):
                        assert abs(box[key] - target[key]) < .1, name
