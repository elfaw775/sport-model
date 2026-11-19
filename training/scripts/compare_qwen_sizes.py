"""Compare two sets of predictions on identical films and individual reference lesions."""
import argparse
import hashlib
import json
from pathlib import Path

from compare_detectors import iou, truth
from score_qwen_vl import AXES, load_qwen, score


def matched_lesions(labels, rows):
    matches = set()
    for filename, targets in labels.items():
        # The archived scorer uses stable score order and the first target in a tie.
        remaining = dict(enumerate(targets, start=1))
        for prediction in sorted(rows[filename][1], key=lambda box: -box["score"]):
            xyxy = tuple(prediction[axis] for axis in AXES)
            overlaps = {idx: iou(xyxy, target) for idx, target in remaining.items()}
            if overlaps:
                best = max(overlaps, key=overlaps.get)
                if overlaps[best] >= 0.5:
                    matches.add(f"{filename}#{best}")
                    del remaining[best]
    return matches


def paired(previous, current, population):
    return {
        "retained": len(previous & current), "both_missed": population - len(previous | current),
        "gained": sorted(current - previous), "lost": sorted(previous - current),
    }


def compare(labels, reference, candidate):
    results = {}
    for model, rows in (("reference", reference), ("candidate", candidate)):
        metrics = score(labels, rows)  # Checks every filename is present exactly once.
        lesions = matched_lesions(labels, rows)
        films = {name for name, targets in labels.items()
                 if rows[name][0] == ("positive" if targets else "negative")}
        assert len(lesions) == metrics["lesion_level_iou50"]["tp"]
        assert len(films) == metrics["image_level"]["tp"] + metrics["image_level"]["tn"]
        results[model] = (metrics, lesions, films)
    previous, current = results["reference"], results["candidate"]
    return {
        "reference": previous[0], "candidate": current[0],
        "paired_lesions": paired(previous[1], current[1], sum(map(len, labels.values()))),
        "paired_correct_films": paired(previous[2], current[2], len(labels)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    ref, cand = load_qwen(args.reference), load_qwen(args.candidate)
    report = {"prediction_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest()
                                     for k, v in vars(args).items()}}
    for split in ("val", "test"):
        report[split] = compare(truth(split), ref[split], cand[split])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
