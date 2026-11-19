"""Train a COCO-initialized YOLO on corrected labels and export per-image predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import monotonic


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work/model-comparison"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=WORK / "dataset/data.yaml")
    parser.add_argument("--weights", type=Path, default=WORK / "pretrained/yolo11s.pt")
    parser.add_argument("--output", type=Path, default=WORK / "yolo")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--imgsz", default=640, type=int, help="Square training and inference size, in pixels")
    parser.add_argument("--optimizer-profile", choices=("baseline", "fetlock-box"), default="baseline")
    args = parser.parse_args()
    output = args.output.resolve()
    data = args.data.resolve(strict=True)
    weights = args.weights.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "fit").exists():
        parser.error("训练结果目录已存在，请指定新的 --output，避免覆盖")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output / "config"))
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import YAML

    config = YAML.load(str(data))
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = data.parent / dataset_root
    torch.set_num_threads(2)
    model = YOLO(str(weights))
    if len(model.names) != 80:
        raise ValueError("本实验要求 COCO 80 类检测初始权重；不能使用训练过的马片单类权重")
    trainer = None
    if args.optimizer_profile == "fetlock-box":
        from fetlock_optimizer import FetlockBoxTrainer
        trainer = FetlockBoxTrainer
    started = monotonic()
    model.train(
        trainer=trainer, data=str(data), device=args.device, project=str(output), name="fit",
        epochs=args.epochs, patience=0, batch=args.batch, imgsz=args.imgsz, seed=42,
        workers=0, amp=False, cache=False, plots=False, freeze=10,
        optimizer="AdamW", lr0=0.001, lrf=0.1, warmup_epochs=1, nbs=8,
        mosaic=0.0, close_mosaic=0, fliplr=0.0, flipud=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.02, scale=0.05,
    )
    elapsed = monotonic() - started
    checkpoint = output / "fit/weights/last.pt"
    detector = YOLO(str(checkpoint))
    counts = {}
    for split in ("val", "test"):
        folder = dataset_root / config[split]
        files = sorted(path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        predictions = {}
        for file in files:
            result, = detector.predict(
                source=str(file), imgsz=args.imgsz, conf=0.01, iou=0.7, device=args.device,
                rect=False, max_det=300, verbose=False, save=False,
            )
            predictions[file.name] = [
                dict(zip(("x1", "y1", "x2", "y2"), box)) | {"score": score}
                for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())
            ]
        (output / f"{split}_predictions.json").write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
        counts[split] = len(predictions)
    metadata = {"initial_weights": weights.name, "initial_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "checkpoint": "fit/weights/last.pt", "epochs": args.epochs, "batch": args.batch, "seed": 42,
                "imgsz": args.imgsz, "optimizer_profile": args.optimizer_profile,
                "training_seconds": elapsed, "image_counts": counts,
                "torch": str(torch.__version__), "ultralytics": ultralytics.__version__, "device": args.device}
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
