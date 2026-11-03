#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把导出的 YOLO ONNX 做 **静态 int8 量化**（带校准图片），用于网页端减小体积。

为什么不用动态量化：动态量化对小模型损伤很大（实测检测数从 4 掉到 1）；
静态量化会用真实图片校准激活范围，通常能保住大部分精度。

用法：
  python quantize_onnx.py \
      --input runs/horse/xxx/weights/best.onnx \
      --output runs/horse/xxx/weights/best.int8.onnx \
      --images datasets/sporthorse_v2/images/train \
      --size 640 --limit 64
"""
import argparse
import glob
import os

import cv2
import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)


def letterbox(img, size=640, color=114):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas


class ImageReader(CalibrationDataReader):
    def __init__(self, paths, size=640, limit=64):
        self.items = []
        for p in paths[:limit]:
            img = cv2.imread(p)
            if img is None:
                continue
            lb = letterbox(img, size)
            x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            self.items.append(x)
        self.i = 0
        print(f"校准图片数量: {len(self.items)}")

    def get_next(self):
        if self.i >= len(self.items):
            return None
        d = {"images": self.items[self.i]}
        self.i += 1
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--images", required=True, help="校准图片目录")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--limit", type=int, default=64)
    args = ap.parse_args()

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(args.images, e))
    paths = sorted(paths)
    if not paths:
        raise SystemExit(f"校准目录里没有图片: {args.images}")

    reader = ImageReader(paths, args.size, args.limit)
    quantize_static(
        args.input, args.output, reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    print(f"fp32: {os.path.getsize(args.input)/1e6:.1f} MB")
    print(f"int8: {os.path.getsize(args.output)/1e6:.1f} MB  -> {args.output}")


if __name__ == "__main__":
    main()
