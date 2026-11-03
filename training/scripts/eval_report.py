#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测模型的可读评估：在指定 split 上统计 TP / FP / FN（IoU>=0.5）+ 图片级命中率。

比 mAP 更直观：
  - 命中(TP)：模型给的框和真值框 IoU>=0.5 的个数
  - 误报(FP)：模型给了框但和任何真值都对不上的个数
  - 漏检(FN)：真值有框但模型没找出来的个数

用法：
  YOLO_CONFIG_DIR="$PWD/.ultralytics" MPLCONFIGDIR="$PWD/.mplconfig" \
  python3 eval_report.py --model runs/horse/lesion_yolo11s_freeze10/weights/best.pt \
      --data datasets/sporthorse --splits test val train
"""
import argparse
import glob
import os

import numpy as np


def xywhn2xyxy(b, w, h):
    x, y, bw, bh = b
    return np.array([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(lbl_path):
    boxes = []
    if os.path.exists(lbl_path):
        for line in open(lbl_path, encoding="utf-8"):
            p = line.split()
            if len(p) >= 5:
                boxes.append([float(v) for v in p[1:5]])
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="datasets/sporthorse")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--confs", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)

    for split in args.splits:
        img_dir = os.path.join(args.data, "images", split)
        lbl_dir = os.path.join(args.data, "labels", split)
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        print("=" * 78)
        print(f"split = {split}   图片 = {len(imgs)}")
        print("=" * 78)

        # 不依赖 conf 统计 GT 总量
        n_gt_total = sum(len(load_gt(os.path.join(lbl_dir, os.path.splitext(os.path.basename(p))[0] + ".txt"))) for p in imgs)
        print(f"真值病变框总数: {n_gt_total}")

        for conf in args.confs:
            TP = FP = FN = 0
            img_hit = 0
            img_fp = 0
            n_pred_total = 0
            for p in imgs:
                stem = os.path.splitext(os.path.basename(p))[0]
                gt = load_gt(os.path.join(lbl_dir, stem + ".txt"))
                r = model.predict(p, conf=conf, verbose=False, device=args.device)[0]
                H, W = r.orig_shape
                preds = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
                n_pred_total += len(preds)

                used = [False] * len(gt)
                m = 0
                fp = 0
                for pb in preds:
                    best_iou, best_j = 0.0, -1
                    for j, g in enumerate(gt):
                        if used[j]:
                            continue
                        v = iou(pb, xywhn2xyxy(g, W, H))
                        if v > best_iou:
                            best_iou, best_j = v, j
                    if best_iou >= args.iou_thr:
                        used[best_j] = True
                        m += 1
                    else:
                        fp += 1
                TP += m
                FP += fp
                FN += len(gt) - m
                if m > 0:
                    img_hit += 1
                if fp > 0:
                    img_fp += 1

            prec = TP / (TP + FP) if TP + FP else 0.0
            rec = TP / (TP + FN) if TP + FN else 0.0
            print(f"\n  conf>={conf}:")
            print(f"    预测框总数={n_pred_total}  命中TP={TP}  误报FP={FP}  漏检FN={FN}")
            print(f"    精度(P 报出来的里有多少对) = {prec:.3f}")
            print(f"    召回(R 真病变找到多少)     = {rec:.3f}")
            print(f"    图片级：{img_hit}/{len(imgs)} 张至少命中 1 个；{img_fp}/{len(imgs)} 张有误报")


if __name__ == "__main__":
    main()
