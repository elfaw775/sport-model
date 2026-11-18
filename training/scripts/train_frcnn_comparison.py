"""Train a two-class Faster R-CNN head on the fixed rotation-corrected split."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import clip_boxes_to_image
from torchvision.transforms.functional import to_tensor


WORK = Path(__file__).resolve().parents[2] / "work/model-comparison"
CHECKSUM = "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf"
CONFIG = dict(min_size=640, max_size=640, rpn_post_nms_top_n_test=200,
              box_batch_size_per_image=64, box_score_thresh=0.01, box_nms_thresh=0.7)


def files(root, split):
    return sorted(path for path in (root / "images" / split).iterdir()
                  if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})


def image_tensor(path):
    with Image.open(path) as image:
        return to_tensor(image.convert("RGB"))


def targets(label_file, width, height):
    boxes = []
    for line in label_file.read_text().splitlines():
        label, cx, cy, bw, bh = map(float, line.split())
        if label != 0:
            raise ValueError(f"Unexpected class in {label_file}")
        boxes.append([(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height])
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64)}


@torch.no_grad()
def frozen_features(model, path, label_root):
    image = image_tensor(path)
    target = targets(label_root / path.with_suffix(".txt").name, image.shape[2], image.shape[1])
    images, truth = model.transform([image], [target])
    features = model.backbone(images.tensors)
    proposals, _ = model.rpn(images, features)
    boxes = truth[0]["boxes"]
    if boxes.numel():
        expanded = boxes.repeat(8, 1)
        lengths = (expanded[:, 2:] - expanded[:, :2]).repeat(1, 2)
        nearby = expanded + (torch.rand_like(expanded) - 0.5) * 0.32 * lengths
        proposals[0] = torch.cat([proposals[0], clip_boxes_to_image(nearby, images.image_sizes[0])])
    roi = model.roi_heads
    samples, _, labels, regression = roi.select_training_samples(proposals, truth)
    pooled = roi.box_roi_pool(features, samples, images.image_sizes)
    return roi.box_head(pooled), labels[0], regression[0]


@torch.inference_mode()
def export(model, paths, output):
    predictions = {}
    for path in paths:
        result, = model([image_tensor(path)])
        predictions[path.name] = [
            dict(zip(("x1", "y1", "x2", "y2"), box)) | {"score": score}
            for box, score in zip(result["boxes"].tolist(), result["scores"].tolist())]
    output.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(predictions)} images to {output.name}", flush=True)


def build_model(weights, trained_head=None):
    with weights.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKSUM:
            raise ValueError("Unexpected Faster R-CNN COCO checkpoint checksum")
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **CONFIG)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.requires_grad_(False)
    heads = model.roi_heads
    heads.box_predictor = FastRCNNPredictor(heads.box_predictor.cls_score.in_features, 2)
    if trained_head is not None:
        heads.box_predictor.load_state_dict(torch.load(trained_head, map_location="cpu", weights_only=True))
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=WORK / "dataset")
    parser.add_argument("--weights", type=Path, default=WORK / "pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth")
    parser.add_argument("--output", type=Path, default=WORK / "frcnn")
    args = parser.parse_args()
    start = time.monotonic()
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(args.weights)
    heads = model.roi_heads
    paths = files(args.dataset, "train")
    cache = []
    for index, path in enumerate(paths, 1):
        cache.append(frozen_features(model, path, args.dataset / "labels/train"))
        if index % 10 == 0 or index == len(paths):
            print(f"Cached {index}/{len(paths)} train cases ({time.monotonic()-start:.0f}s)", flush=True)
    optimizer = torch.optim.AdamW(heads.box_predictor.parameters(), lr=0.001, weight_decay=0.0001)
    history = []
    for epoch in range(1, 13):
        order = list(range(len(cache)))
        random.shuffle(order)
        classifier_total = bbox_total = 0.0
        for offset in range(0, len(order), 4):
            indices = order[offset:offset+4]
            features, labels, reg_targets = zip(*(cache[index] for index in indices))
            logits, deltas = heads.box_predictor(torch.cat(features))
            classifier, bbox = fastrcnn_loss(logits, deltas, labels, reg_targets)
            if not torch.isfinite(classifier + bbox):
                raise ValueError("Nonfinite loss")
            optimizer.zero_grad()
            (classifier + bbox).backward()
            optimizer.step()
            classifier_total += classifier.item() * len(indices)
            bbox_total += bbox.item() * len(indices)
        history.append({"epoch": epoch, "classifier_loss": classifier_total/len(cache), "box_loss": bbox_total/len(cache)})
        print(history[-1], flush=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": 2, "config": CONFIG,
                "epochs": 12, "seed": 42}, args.output / "epoch12.pth")
    torch.save(heads.box_predictor.state_dict(), args.output / "frcnn-head.pth")
    export(model, files(args.dataset, "val"), args.output / "predictions.json")
    export(model, files(args.dataset, "test"), args.output / "test_predictions.json")
    metadata = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                "pretrained_sha256": CHECKSUM, "epochs": 12, "seed": 42, "config": CONFIG,
                "trainable": "roi_heads.box_predictor only",
                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                "hist": history, "elapsed_seconds": time.monotonic()-start}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
