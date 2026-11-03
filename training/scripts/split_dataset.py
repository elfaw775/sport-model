#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 CVAT 导出的 "Ultralytics YOLO Detection 1.0" 标注集拆分成 train/val/test，
并生成可直接交给 ultralytics 训练的目录结构与 data.yaml。

背景约定（来自用户确认）：
  * 每张图片 = 一匹不同的马  => 按【图片】随机分层拆分，而不是按框拆分，
    避免同一匹马同时出现在训练集和测试集里（数据泄漏）。
  * class 0 = 病变（唯一要检测的目标类别）
  * class 1 = 标注的部位标记（可忽略，仅用于文件名记录）
  * 图片文件名里带“骨折”的，是骨折病例；不带的大概率是正常马。

用法示例（默认配置）：
  python3 split_dataset.py \
      --images "骨折x线片（总）" \
      --labels "sporthorse1103/labels/train" \
      --out datasets/sporthorse \
      --ratio 0.7 0.15 0.15 \
      --seed 42 \
      --mode filename-heuristic

输出结构：
  datasets/sporthorse/
    data.yaml
    split_manifest.tsv          # 每张图归到哪一组、类别、原因，方便人工复核
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt
"""

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict

# 图片扩展名白名单
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="CVAT YOLO 标注集 -> train/val/test 拆分")
    p.add_argument("--images", required=True, help="原始图片目录")
    p.add_argument("--negative-images", nargs="*", default=[],
                   help="额外的负样本（健康/无病变）图片目录，可多个。这些图会生成空 label（背景）")
    p.add_argument("--labels", required=True, help="原始 label 目录（labels/train）")
    p.add_argument("--out", required=True, help="输出数据集根目录")
    p.add_argument("--ratio", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"), help="train/val/test 比例，默认 0.7 0.15 0.15")
    p.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    p.add_argument("--keep-classes", nargs="+", type=int, default=[0],
                   help="保留的类别 id，默认只有 0（病变）。多个时按顺序重编号为 0,1,...")
    # 默认用 ASCII 名，避免 ultralytics 因中文类别名去下载 22MB 的 Arial.Unicode.ttf
    p.add_argument("--class-names", nargs="+", default=["lesion"],
                   help="与 --keep-classes 一一对应的类别名")
    p.add_argument("--mode", choices=["filename-heuristic", "all-negatives", "positives-only"],
                   default="filename-heuristic",
                   help="无 class0 框的图片如何处理（见文档）")
    p.add_argument("--exclude-stems", nargs="*", default=[],
                   help="要跳过的图片名（不含扩展名），如重复副本")
    p.add_argument("--fracture-keywords", nargs="+", default=["骨折"],
                   help="文件名包含这些关键词的图片视为骨折病例")
    p.add_argument("--link", action="store_true",
                   help="用软链接而不是拷贝图片（省磁盘，但移动原图会失效）")
    p.add_argument("--dry-run", action="store_true", help="只打印拆分结果，不写文件")
    return p.parse_args()


def stem_of(path):
    """返回不含扩展名的文件名（保留扩展名前可能的空格，保证与 label 精确同名）。"""
    base = os.path.basename(path)
    root, _ = os.path.splitext(base)
    return root


def read_boxes(label_path):
    """读取 label 文件，返回 [(cls, x, y, w, h), ...]，忽略空行与非法行。"""
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            boxes.append((cls, x, y, w, h))
    return boxes


def allocate(n, ratios, rng):
    """把 n 个样本按 ratios 尽量均匀地分到 len(ratios) 个桶，返回每桶数量列表。"""
    floors = [int(n * r) for r in ratios]
    rem = n - sum(floors)
    # 按小数部分从大到小补足余数
    order = sorted(range(len(ratios)),
                   key=lambda i: (n * ratios[i] - floors[i], -i), reverse=True)
    for k in range(rem):
        floors[order[k % len(order)]] += 1
    return floors


def main():
    args = parse_args()
    images_dir = args.images
    labels_dir = args.labels
    out_root = args.out

    if not os.path.isdir(images_dir):
        sys.exit(f"错误：图片目录不存在 -> {images_dir}")
    if not os.path.isdir(labels_dir):
        sys.exit(f"错误：label 目录不存在 -> {labels_dir}")

    ratio = args.ratio
    if abs(sum(ratio) - 1.0) > 1e-6:
        sys.exit("错误：--ratio 三者之和必须等于 1")
    if len(args.keep_classes) != len(args.class_names):
        sys.exit("错误：--keep-classes 与 --class-names 数量必须一致")

    keep = set(args.keep_classes)
    remap = {c: i for i, c in enumerate(args.keep_classes)}  # 源类别 -> 新连续编号
    fracture_kw = args.fracture_keywords
    rng = random.Random(args.seed)

    # 1. 收集所有图片（主目录 + 负样本目录）
    sources = [(images_dir, "main")] + [(d, "neg") for d in args.negative_images]
    all_images = []  # [(path, kind)]
    seen = {}
    for d, kind in sources:
        if not os.path.isdir(d):
            sys.exit(f"错误：图片目录不存在 -> {d}")
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() not in IMG_EXTS:
                continue
            p = os.path.join(d, f)
            st = stem_of(p)
            if st in set(args.exclude_stems):
                continue
            if st in seen:
                sys.exit(f"错误：文件名重复 -> {st}（{kind} 目录）；请先重命名")
            seen[st] = p
            all_images.append((p, kind))
    if not all_images:
        sys.exit(f"错误：{images_dir} 里没有图片")

    # 2. 建立 label 索引：stem -> (label_path, boxes)
    label_index = {}
    for f in os.listdir(labels_dir):
        if f.endswith(".txt"):
            stem = stem_of(f)
            label_index[stem] = os.path.join(labels_dir, f)

    # 3. 逐图归类
    positives, negatives, excluded = [], [], []
    meta = {}  # stem -> dict
    for img, kind in all_images:
        stem = stem_of(img)
        label_path = label_index.get(stem)
        boxes = read_boxes(label_path) if label_path else []
        has_pos = any(c in keep for c, *_ in boxes)
        name_has_fracture = any(kw in stem for kw in fracture_kw)

        meta[stem] = {
            "image": img,
            "label": label_path,
            "boxes": boxes,
            "has_pos": has_pos,
            "fracture_name": name_has_fracture,
            "source": kind,
        }

        if kind == "neg":
            # 负样本目录：默认当背景（空 label）；若意外带了 class0 框则按正样本处理
            if has_pos:
                print(f"  ⚠️  负样本目录里的 {stem} 含 class0 框，已按正样本处理")
                positives.append(stem)
            else:
                negatives.append(stem)
            continue

        if has_pos:
            positives.append(stem)
        else:
            if args.mode == "filename-heuristic":
                # 无 class0 框：若文件名带骨折关键词 -> 排除（缺标注，需补标）；
                # 否则当正常（负样本）。
                if name_has_fracture:
                    excluded.append(stem)
                else:
                    negatives.append(stem)
            elif args.mode == "all-negatives":
                negatives.append(stem)
            else:  # positives-only
                excluded.append(stem)

    # 4. 分层拆分（正负样本各自 shuffle 后按比例切）
    def split_bucket(stems):
        r = rng.sample(stems, len(stems))
        counts = allocate(len(r), ratio, rng)
        a, b = counts[0], counts[0] + counts[1]
        return r[:a], r[a:b], r[b:]

    pos_splits = split_bucket(positives)
    neg_splits = split_bucket(negatives)

    split_names = ["train", "val", "test"]
    assign = defaultdict(list)  # split -> [stem, ...]
    for s, name in enumerate(split_names):
        assign[name] = pos_splits[s] + neg_splits[s]

    # 5. 打印汇总
    print("=" * 72)
    print("拆分汇总  (mode=%s, seed=%d)" % (args.mode, args.seed))
    print("=" * 72)
    print(f"图片总数        : {len(all_images)}")
    print(f"正样本(class0框): {len(positives)}")
    print(f"负样本          : {len(negatives)}")
    print(f"排除(缺标注)     : {len(excluded)}")
    print("-" * 72)
    for name in split_names:
        n_pos = sum(1 for s in assign[name] if meta[s]["has_pos"])
        print(f"  {name:5s}: 共 {len(assign[name]):3d} 张 (正 {n_pos} / 负 {len(assign[name])-n_pos})")
    if excluded:
        print("-" * 72)
        print(f"被排除的 {len(excluded)} 张（建议补标 class0 后再纳入）：")
        for s in excluded:
            print("   -", s)

    # 6. 写文件
    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。")
        return

    def ensure(d):
        os.makedirs(d, exist_ok=True)

    def copy_or_link(src, dst):
        if os.path.exists(dst):
            os.remove(dst)
        if args.link:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)

    for name in split_names:
        ensure(os.path.join(out_root, "images", name))
        ensure(os.path.join(out_root, "labels", name))

    manifest_path = os.path.join(out_root, "split_manifest.tsv")
    manifest_lines = ["image\tcategory\tsplit\tclass0_boxes\tclass1_boxes\treason"]

    for name in split_names:
        for stem in assign[name]:
            m = meta[stem]
            copy_or_link(m["image"], os.path.join(out_root, "images", name, os.path.basename(m["image"])))

            # 只保留要的类别，并重编号
            kept = [b for b in m["boxes"] if int(b[0]) in keep]
            n0 = sum(1 for b in m["boxes"] if int(b[0]) == 0)
            n1 = sum(1 for b in m["boxes"] if int(b[0]) == 1)
            out_label = os.path.join(out_root, "labels", name, stem + ".txt")
            with open(out_label, "w", encoding="utf-8") as f:
                for cls, x, y, w, h in kept:
                    f.write(f"{remap[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            cat = "pos" if m["has_pos"] else "neg"
            manifest_lines.append(
                f"{stem}\t{cat}\t{name}\t{n0}\t{n1}\t"
            )

    for stem in excluded:
        m = meta[stem]
        n0 = sum(1 for b in m["boxes"] if int(b[0]) == 0)
        n1 = sum(1 for b in m["boxes"] if int(b[0]) == 1)
        manifest_lines.append(
            f"{stem}\texcluded\t-\t{n0}\t{n1}\t缺class0标注(文件名含骨折关键词)"
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    # 7. data.yaml
    names_lines = "\n".join(f"  {i}: {nm}" for i, nm in enumerate(args.class_names))
    data_yaml = f"""# 自动生成，勿手改（可手改 epochs/imgsz 等训练超参，见训练指南）
path: {os.path.abspath(out_root)}   # 数据集根目录（绝对路径）
train: images/train
val: images/val
test: images/test

# 类别（与 --keep-classes / --class-names 对应）
names:
{names_lines}
"""
    with open(os.path.join(out_root, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(data_yaml)

    print("\n完成。输出目录：", os.path.abspath(out_root))
    print("  data.yaml :", os.path.join(out_root, "data.yaml"))
    print("  清单      :", manifest_path)


if __name__ == "__main__":
    main()
