import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_rfdetr import audit_images, coco_score, to_records
from compare_detectors import counts, select_operating_point, truth
from prepare_comparison import ROOT, SOURCE, prepare


class ComparisonTests(unittest.TestCase):
    def test_snapshot_preserves_split_and_uses_private_copies(self):
        coco = json.loads((ROOT / "training/source/coco/instances_default.json").read_text(encoding="utf-8"))
        expected_sizes = {Path(record["file_name"]).stem: (record["width"], record["height"])
                          for record in coco["images"]}
        with tempfile.TemporaryDirectory() as root:
            dataset = prepare(Path(root))
            audit_images(dataset)
            for split, films, n_boxes in (("train", 59, 57), ("val", 13, 12), ("test", 12, 12)):
                labels = truth(split, dataset)
                self.assertEqual(len(labels), films)
                self.assertEqual(sum(map(len, labels.values())), n_boxes)
                for name, boxes in labels.items():
                    copied = dataset / "images" / split / name
                    self.assertFalse(copied.samefile(SOURCE / "images" / split / name))
                    if boxes:
                        with Image.open(copied) as image:
                            self.assertEqual(image.size, expected_sizes[Path(name).stem])

    def test_validation_operating_point_respects_false_positives(self):
        labels = {"positive": [(0, 0, 10, 10)], "normal": []}
        predictions = {
            "positive": to_records([(0, 0, 10, 10), (0, 0, 10, 10)], [.6, .15], [0, 0], 80, 60),
            "normal": to_records([(0, 0, 10, 10), (10, 10, 20, 20)], [.15, .15], [0, 0], 80, 60),
        }
        self.assertEqual(counts(predictions, labels, .15), {"tp": 1, "fp": 3, "fn": 0})
        threshold = select_operating_point(predictions, labels)
        self.assertEqual(threshold, .2)
        self.assertEqual(counts(predictions, labels, threshold), {"tp": 1, "fp": 0, "fn": 0})

    def test_shared_coco_metric_and_coordinate_clipping(self):
        labels = {"img": [(0, 0, 10, 10)], "normal": []}
        sizes = {"img": (10, 10), "normal": (10, 10)}
        predictions = {"img": to_records([(-1, -1, 11, 11), (0, 0, 10, 10)], [.8, .99], [0, 1], 10, 10),
                       "normal": []}
        with contextlib.redirect_stdout(io.StringIO()):
            perfect = coco_score(labels, predictions, sizes)
            empty = coco_score(labels, {"img": [], "normal": []}, sizes)
        self.assertAlmostEqual(perfect["AP50_95"], 1)
        self.assertEqual(empty["AP50_95"], 0)


if __name__ == "__main__":
    unittest.main()
