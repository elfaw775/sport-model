"""Time the deployed ONNX detector on the same films and one requested ORT provider."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter

import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend import detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("CPUExecutionProvider", "CUDAExecutionProvider"),
                        required=True)
    args = parser.parse_args()
    if args.provider not in ort.get_available_providers():
        parser.error(f"{args.provider} unavailable. Installed providers: {ort.get_available_providers()}")
    weights = detector.WEIGHTS.read_bytes()
    if sha256(weights).hexdigest() != detector.WEIGHTS_SHA256:
        raise ValueError("Unexpected ONNX checkpoint")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    start = perf_counter()
    session = ort.InferenceSession(weights, sess_options=options, providers=[args.provider])
    if session.get_providers()[0] != args.provider:
        raise RuntimeError(f"Execution provider failed to initialize: {session.get_providers()}")
    init_ms = 1000 * (perf_counter() - start)
    input_name = session.get_inputs()[0].name

    def infer(image):
        tensor, scale, left, top = detector._prepare(image)
        output, = session.run(None, {input_name: tensor})
        return detector._decode(output, scale, left, top, image.width, image.height)

    images = ROOT / "training/data/sporthorse_v2/images"
    with Image.open(sorted((images / "train").glob("*.jpg"))[0]) as warmup:
        infer(warmup.convert("RGB"))
    samples = []
    for path in sorted((images / "test").iterdir()):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            with Image.open(path) as original:
                samples.append((path.name, original.convert("RGB")))
    if not samples:
        raise ValueError("The historical test split contains no images")
    measurements = []
    for _ in range(3):
        for name, image in samples:
            start = perf_counter()
            boxes = infer(image)
            measurements.append({"file": name, "ms": round(1000 * (perf_counter() - start), 2),
                                 "boxes": len(boxes)})
    print(json.dumps({"model": detector.MODEL_ID, "provider": args.provider,
                      "provider_order": session.get_providers(), "onnxruntime": ort.__version__,
                      "platform": platform.platform(), "machine": platform.machine(),
                      "films": len(samples), "rounds": 3, "session_init_ms": round(init_ms, 2),
                      "median_ms": round(median(row["ms"] for row in measurements), 2),
                      "timing": "Warm batch=1, resize, inference and box decode. JPEG read excluded.",
                      "samples": measurements}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
