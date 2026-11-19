"""Local CUDA comparison of RF-DETR Small and YOLO11s on the corrected v2 split."""

import argparse
import gc
import hashlib
import json
import math
import os
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

from PIL import Image

from compare_detectors import counts, select_operating_point, truth
from prepare_comparison import ROOT, prepare

RESOLUTION = 640
SCORE_FLOOR = 0.001
MAX_DETECTIONS = 100
YOLO_SHA256 = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
RF_MD5 = "fb37061c1af7bace359c91b723a8d5c1"
ARCHIVED = ROOT / "training/models/yolo11s_fresh_v2_best.pt"


def file_hash(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit_images(dataset):
    """Refuse identical decoded pixels across splits and ambiguous image orientation."""
    seen = {}
    for split in ("train", "val", "test"):
        for path in sorted((dataset / "images" / split).iterdir()):
            with Image.open(path) as image:
                if image.getexif().get(274, 1) != 1 or getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"Normalize image orientation against the labels first: {path}")
                check = (image.size, hashlib.sha256(image.convert("RGB").tobytes()).digest())
            other_split, original = seen.setdefault(check, (split, path.name))
            if other_split != split:
                raise ValueError(f"Identical image across {other_split} and {split}: {original} / {path.name}")


def fit_yolo(official, dataset, args):
    from ultralytics import YOLO, settings

    settings.update({key: False for key in ("sync", "hub", "wandb", "mlflow", "comet_ml", "clearml") if key in settings})
    model = YOLO(str(official), task="detect")
    model.train(
        data=str(dataset / "data.yaml"), epochs=args.epochs, imgsz=RESOLUTION, device=0,
        project=str(args.work), name="yolo11s", seed=args.seed, workers=0, amp="bf16",
        batch=args.yolo_batch, nbs=8, freeze=10, patience=0, cache=False, plots=False,
        optimizer="AdamW", lr0=0.001, lrf=0.1, warmup_epochs=1, cos_lr=True,
        mosaic=0, close_mosaic=0, mixup=0, cutmix=0, copy_paste=0,
        fliplr=0, flipud=0, hsv_h=0, hsv_s=0, hsv_v=0, translate=0.02, scale=0.05,
        max_det=MAX_DETECTIONS, rect=False,
    )
    return Path(model.trainer.best)


def fit_rf(official, dataset, args):
    from rfdetr import RFDETRSmall

    model = RFDETRSmall(pretrain_weights=str(official), device="cuda", amp=True, compile=False,
                        trust_checkpoint=True)  # Official weight checksum was verified before fitting.
    work = args.work / "rf-detr-small"
    model.train(
        dataset_dir=str(dataset), dataset_file="yolo", output_dir=str(work),
        resolution=RESOLUTION, epochs=args.epochs, num_workers=0, batch_size=args.rf_batch,
        grad_accum_steps=math.ceil(8 / args.rf_batch), devices=1, seed=args.seed, amp_dtype="bf16",
        lr=1.5e-4, lr_encoder=3e-5, lr_scheduler="cosine", warmup_epochs=1,
        square_resize_div_64=True, multi_scale=False, scale_jitter=False, aug_config={},
        run_test=False, eval_max_dets=MAX_DETECTIONS, eval_ema_only=True,
        skip_best_epochs=min(2, args.epochs - 1), checkpoint_interval=args.epochs,
        tensorboard=False, wandb=False, mlflow=False, early_stopping=False, progress_bar=None,
    )
    return work / "checkpoint_best_total.pth"


def to_records(boxes, scores, classes, width, height):
    result = []
    for xyxy, confidence, category in sorted(zip(boxes, scores, classes), key=lambda row: -float(row[1])):
        score = float(confidence)
        a, b, c, d = (float(p) for p in xyxy)
        if int(category) != 0 or not all(math.isfinite(x) for x in (score, a, b, c, d)) or score < SCORE_FLOOR:
            continue
        x1, y1, x2, y2 = max(0, a), max(0, b), min(width, c), min(height, d)
        if x1 < x2 and y1 < y2:
            result.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score})
        if len(result) == MAX_DETECTIONS:
            break
    return result


def predictor(weights, family):
    if family == "rf_detr":
        from rfdetr import RFDETRSmall

        # Produced by this run, not externally supplied.
        # RF-DETR 1.9.4 may strip model_config from the best-total checkpoint. Reconstruct
        # the training grid explicitly so loading does not interpolate it back down to 512.
        model = RFDETRSmall.from_checkpoint(str(weights), device="cuda", trust_checkpoint=True,
                                            resolution=RESOLUTION, positional_encoding_size=RESOLUTION // 16)
        if model.model_config.resolution != RESOLUTION:
            raise RuntimeError(f"Unexpected RF-DETR checkpoint resolution: {model.model_config.resolution}")
        model.inference(compile=False)

        def predict(image):
            result = model.predict(image, threshold=SCORE_FLOOR, include_source_image=False)
            return result.xyxy, result.confidence, result.class_id
    else:
        from ultralytics import YOLO

        model = YOLO(str(weights), task="detect")
        if model.names != {0: "lesion"}:
            raise ValueError(f"Expected one lesion class, got {model.names}")

        def predict(image):
            boxes = model.predict(image, imgsz=RESOLUTION, conf=SCORE_FLOOR, iou=0.7,
                                  max_det=MAX_DETECTIONS, device=0, rect=False, verbose=False)[0].boxes.cpu()
            return boxes.xyxy.numpy(), boxes.conf.numpy(), boxes.cls.numpy()
    return predict


def run_predictions(dataset, split, predict, torch):
    records, timing, sizes = {}, [], {}
    with torch.inference_mode():
        for path in sorted((dataset / "images" / split).iterdir()):
            with Image.open(path) as read:
                image = read.convert("RGB")
            if not timing:
                predict(image)  # Warm-up is not included in the timing.
            torch.cuda.synchronize()
            start = perf_counter()
            raw = predict(image)
            torch.cuda.synchronize()
            timing.append((perf_counter() - start) * 1000)
            sizes[path.name] = image.size
            records[path.name] = to_records(*raw, *image.size)
    return records, sizes, median(timing)


def coco_score(labels, predictions, sizes):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if labels.keys() != predictions.keys() or labels.keys() != sizes.keys():
        raise ValueError("Ground truth, filenames and predictions must match exactly")
    images, annotations, proposals = [], [], []
    for img_id, file in enumerate(sorted(labels), 1):
        width, height = sizes[file]
        images.append({"id": img_id, "file_name": file, "width": width, "height": height})
        for x1, y1, x2, y2 in labels[file]:
            w, h = x2 - x1, y2 - y1
            annotations.append({"id": len(annotations) + 1, "image_id": img_id, "category_id": 1,
                                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
        for box in predictions[file]:
            proposals.append({"image_id": img_id, "category_id": 1, "score": box["score"],
                              "bbox": [box["x1"], box["y1"], box["x2"] - box["x1"], box["y2"] - box["y1"]]})
    ref = COCO()
    ref.dataset = {"info": {}, "images": images, "categories": [{"id": 1, "name": "lesion"}],
                   "annotations": annotations}
    ref.createIndex()
    if proposals:
        detected = ref.loadRes(proposals)
    else:
        detected = COCO()
        detected.dataset = {**ref.dataset, "annotations": []}
        detected.createIndex()
    score = COCOeval(ref, detected, "bbox")
    score.params.maxDets = [1, 10, MAX_DETECTIONS]
    score.evaluate()
    score.accumulate()
    score.summarize()
    return {"AP50_95": float(score.stats[0]), "AP50": float(score.stats[1]), "AP75": float(score.stats[2])}


def release_cuda(torch):
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-init", type=Path, required=True, help="Official COCO yolo11s.pt")
    parser.add_argument("--rf-init", type=Path, required=True, help="Official COCO RF-DETR Small pth")
    default_work = ROOT / "work/rfdetr-comparison" / datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument("--work", type=Path, default=default_work, help="New directory; never overwrites runs")
    parser.add_argument("--score-existing", action="store_true",
                        help="Finish scoring the trusted checkpoints from an unscored local --work directory")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--yolo-batch", type=int, default=4)
    parser.add_argument("--rf-batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.yolo_batch, args.rf_batch) < 1:
        parser.error("Epoch and batch sizes must be positive")
    yolo, rf = args.yolo_init.resolve(strict=True), args.rf_init.resolve(strict=True)
    if file_hash(yolo) != YOLO_SHA256 or file_hash(rf, "md5") != RF_MD5:
        parser.error("Initial weights must match the checksums of the documented official downloads")
    args.work = args.work.resolve()
    if args.score_existing:
        if not (args.work / "dataset/data.yaml").is_file():
            parser.error("--score-existing needs a previously prepared --work directory")
        if (args.work / "comparison.json").exists() or any(args.work.glob("*-test-predictions.json")):
            parser.error("The held-out split has already been scored or started scoring")
    else:
        args.work.mkdir(parents=True, exist_ok=False)
    os.environ.update(YOLO_CONFIG_DIR=str(ROOT / "work/rfdetr-comparison/ultralytics"),
                      MPLCONFIGDIR=str(ROOT / "work/rfdetr-comparison/matplotlib"), YOLO_OFFLINE="true",
                      YOLO_AUTOINSTALL="false", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_MODE="disabled")
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("A CUDA GPU with BF16 is required; will not silently train on CPU")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    dataset = args.work / "dataset" if args.score_existing else prepare(args.work)
    audit_images(dataset)
    report = {"device": torch.cuda.get_device_name(0), "epochs": args.epochs,
              "resolution": RESOLUTION, "seed": args.seed, "yolo_batch": args.yolo_batch, "rf_batch": args.rf_batch,
              "software": {p: version(p) for p in ("torch", "torchvision", "ultralytics", "rfdetr", "pycocotools")},
              "split_manifest_sha256": file_hash(ROOT / "training/data/sporthorse_v2/split_manifest.tsv"),
              "cvat_annotations_sha256": file_hash(ROOT / "training/source/coco/instances_default.json"),
              "models": {"archived_yolo11s": {"checkpoint_sha256": file_hash(ARCHIVED)}}}

    fit_report = args.work / "fit.json"
    if args.score_existing and fit_report.is_file():
        report = json.loads(fit_report.read_text(encoding="utf-8"))
    weights = {"archived_yolo11s": ARCHIVED}
    for name, fitted, init in (("retrained_yolo11s", fit_yolo, yolo), ("rf_detr", fit_rf, rf)):
        train_stats = {}
        if args.score_existing:
            relative = ("yolo11s/weights/best.pt" if name == "retrained_yolo11s"
                        else "rf-detr-small/checkpoint_best_total.pth")
            path = args.work / relative
        else:
            print(f"Training {name} on {report['device']}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            path = fitted(init, dataset, args)
            train_stats = {"train_seconds": round(perf_counter() - start, 2),
                           "peak_training_mib": round(torch.cuda.max_memory_allocated() / 1024**2)}
        if not path.is_file():
            raise RuntimeError(f"No validation-selected checkpoint: {path}")
        checkpoint_sha = file_hash(path)
        previous = report["models"].get(name, {})
        if previous and previous.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError(f"Checkpoint no longer matches the completed fit: {path}")
        weights[name] = path
        report["models"][name] = {
            **previous, **train_stats, "checkpoint_sha256": checkpoint_sha,
            "initial_sha256": file_hash(init), "checkpoint": path.relative_to(args.work).as_posix(),
        }
        release_cuda(torch)
    write_json(fit_report, report)

    for split in ("val", "test"):
        labels = truth(split, dataset)
        for name, path in weights.items():
            predict = predictor(path, name)
            pred, sizes, elapsed = run_predictions(dataset, split, predict, torch)
            del predict
            release_cuda(torch)
            write_json(args.work / f"{name}-{split}-predictions.json", pred)
            model = report["models"][name]
            if split == "val":
                model["validation_selected_threshold"] = select_operating_point(pred, labels)
            threshold = model["validation_selected_threshold"]
            model[split] = {**coco_score(labels, pred, sizes), "images": len(labels),
                            "fixed_0.15": counts(pred, labels, 0.15),
                            "at_validation_threshold": counts(pred, labels, threshold),
                            "median_inference_ms": round(elapsed, 2)}
        # Freeze every model's validation choice before generating any test predictions.
        if split == "val":
            write_json(args.work / "validation.json", report)
    write_json(args.work / "comparison.json", report)
    print(f"Report: {args.work / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
