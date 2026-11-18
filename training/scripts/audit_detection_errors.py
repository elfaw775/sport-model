"""Index each annotated lesion against archived YOLO candidates and reproduce the reported counts."""
import csv
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from compare_detectors import ROOT, counts, iou, truth


RESULTS = ROOT / "training/results/comparison-2026-11-18"
IMAGES = Path("training/data/sporthorse_v2/images")
OPERATING_SCORE = 0.15
ARCHIVED_SCORE = 0.01
MATCH_IOU = 0.5


def audit_image(split, model, name, targets, predictions):
    image_path = IMAGES / split / name
    with Image.open(ROOT / image_path) as image:
        long_edge = max(image.size)
    candidates = sorted((p for p in predictions if p['score'] >= ARCHIVED_SCORE),
                        key=lambda p: -p['score'])
    remaining = list(range(len(targets)))
    matched_scores = {}
    best_iou = [0.0] * len(targets)
    for candidate in candidates:
        box = [candidate[key] for key in ('x1', 'y1', 'x2', 'y2')]
        overlaps = [iou(box, target) for target in targets]
        best_iou = [max(previous, current) for previous, current in zip(best_iou, overlaps)]
        if remaining:
            index = max(remaining, key=overlaps.__getitem__)
            if overlaps[index] >= MATCH_IOU:
                matched_scores[index] = candidate['score']
                remaining.remove(index)

    for index, target in enumerate(targets):
        score = matched_scores.get(index)
        if score is not None:
            status = 'detected' if score >= OPERATING_SCORE else 'low_score'
        elif best_iou[index] >= MATCH_IOU:
            status = 'assigned_elsewhere'
        elif best_iou[index] > 0:
            status = 'partial_overlap'
        else:
            status = 'no_overlap'
        yield {
            'split': split, 'model': model, 'image': image_path.as_posix(),
            'lesion_index': index + 1, 'status': status,
            'x1': round(target[0], 2), 'y1': round(target[1], 2),
            'x2': round(target[2], 2), 'y2': round(target[3], 2),
            'short_edge_at_640_px': round(640 * min(target[2]-target[0], target[3]-target[1]) / long_edge, 1),
            'max_iou': round(best_iou[index], 5),
            'matched_score': '' if score is None else score,
            'image_candidates_0.01': len(candidates),
            'image_candidates_0.15': sum(p['score'] >= OPERATING_SCORE for p in candidates),
        }


def main():
    rows = []
    metrics = json.loads((RESULTS / 'metrics.json').read_text())
    with ZipFile(RESULTS / 'predictions.zip') as archive:
        for split in ('val', 'test'):
            labels = truth(split)
            for model in ('shipped', 'yolo'):
                predictions = json.loads(archive.read(f'{model}/{split}_predictions.json'))
                observed = Counter()
                for name in sorted(labels):
                    for row in audit_image(split, model, name, labels[name], predictions[name]):
                        observed[row['status']] += 1
                        rows.append(row)
                fixed = counts(predictions, labels, OPERATING_SCORE)
                floor = counts(predictions, labels, ARCHIVED_SCORE)
                assert fixed == metrics[model]['fixed_0.15'][split]
                assert fixed['tp'] == observed['detected']
                assert fixed['fn'] == observed.total() - observed['detected']
                assert floor['tp'] == observed['detected'] + observed['low_score']
                print(split, model, dict(observed), 'FP at 0.15/0.01:', fixed['fp'], floor['fp'])
    with (RESULTS / 'lesion-error-audit.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
