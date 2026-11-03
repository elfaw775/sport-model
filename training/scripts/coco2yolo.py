#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 CVAT 导出的 COCO 1.0 (instances_*.json) 转成 YOLO 的 label txt。

COCO 是 JSON，会把所有帧的所有标注都带出来（不像 YOLO 导出可能丢帧），
所以当 CVAT 的 "Ultralytics YOLO Detection 1.0" 导出丢标注时，用这个兜底。

用法：
  python3 coco2yolo.py instances_default.json --out labels_converted \
      --keep 病变            # 只保留类别名含"病变"的框，重编号为 class 0

  # 想保留全部类别（0=病变, 1=annotation mark 一起留）：
  python3 coco2yolo.py instances_default.json --out labels_converted --keep-all

转完后，把 labels_converted 里的 txt 覆盖到 sporthorse1103/labels/train/，
再跑 split_dataset.py 即可。
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="COCO 1.0 -> YOLO labels")
    ap.add_argument("json", help="CVAT COCO 1.0 导出的 instances_*.json")
    ap.add_argument("--out", required=True, help="输出 label 目录")
    ap.add_argument("--keep", nargs="+", default=["病变"],
                    help="要保留的类别名（子串匹配），默认 ['病变']，重编号为 0,1,...")
    ap.add_argument("--keep-all", action="store_true",
                    help="保留所有类别，不按名称过滤")
    ap.add_argument("--img-dir", default=None,
                    help="可选：原始图片目录，用于校验数量是否对得上")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    if not images:
        raise SystemExit(f"错误：{args.json} 里没有 images 字段，确认导的是 COCO 1.0（instances_*.json）")

    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    # 决定保留哪些类别，并重编号
    if args.keep_all:
        keep_names = list(cat_id_to_name.values())
        remap = {c["id"]: i for i, c in enumerate(categories)}
    else:
        keep_names = [c["name"] for c in categories if any(k in c["name"] for k in args.keep)]
        remap = {}
        for i, name in enumerate(keep_names):
            for c in categories:
                if c["name"] == name:
                    remap[c["id"]] = i

    print("保留类别:", keep_names)

    # image_id -> (file_name, w, h)
    img_info = {im["id"]: (im["file_name"], im["width"], im["height"]) for im in images}

    # 按 image_id 归集框
    from collections import defaultdict
    boxes_by_img = defaultdict(list)
    skipped = 0
    for a in annotations:
        cid = a.get("category_id")
        if cid not in remap:
            skipped += 1
            continue
        bbox = a.get("bbox")  # [x, y, w, h] 像素，左上角原点
        if not bbox:
            skipped += 1
            continue
        boxes_by_img[a["image_id"]].append((cid, bbox))

    os.makedirs(args.out, exist_ok=True)
    written = 0
    clipped = 0
    degenerate = 0
    for img_id, (fname, W, H) in img_info.items():
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(args.out, stem + ".txt")
        lines = []
        for cid, (x, y, w, h) in boxes_by_img.get(img_id, []):
            # 裁剪到图片边界（旋转框的 AABB 可能溢出）
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(W), x + w), min(float(H), y + h)
            cw, ch = x1 - x0, y1 - y0
            if cw <= 0.5 or ch <= 0.5:
                degenerate += 1
                continue
            if (x0, y0, x1, y1) != (x, y, x + w, y + h):
                clipped += 1
            cx = (x0 + cw / 2) / W
            cy = (y0 + ch / 2) / H
            nw = cw / W
            nh = ch / H
            cls = remap[cid]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        written += 1

    print(f"图片总数(COCO): {len(images)}")
    print(f"标注总数(COCO): {len(annotations)}")
    print(f"被过滤的框(非保留类别/无bbox): {skipped}")
    print(f"越界并已裁剪的框: {clipped}")
    print(f"裁剪后退化被丢弃的框: {degenerate}")
    print(f"已写出 label txt: {written} 个 -> {args.out}")

    if args.img_dir:
        imgs = {os.path.splitext(f)[0] for f in os.listdir(args.img_dir)}
        labs = {os.path.splitext(f)[0] for f in os.listdir(args.out)}
        print(f"图片目录文件数: {len(imgs)}")
        print(f"缺 label 的图片: {len(imgs - labs)}")
        print(f"图片目录里没有的 label: {len(labs - imgs)}")


if __name__ == "__main__":
    main()
