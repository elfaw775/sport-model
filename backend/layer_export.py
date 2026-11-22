"""Per-image overlays and CVAT COCO 1.0 training archives."""
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from PIL import Image, ImageDraw, ImageFont, ImageOps


def regions(case, image, layer):
    marks = []
    if image["id"] == "primary" and layer in {"ai", "combined"}:
        for detection in case["detections"]:
            points = [(detection["x1"], detection["y1"]), (detection["x2"], detection["y2"])]
            marks.append(("rectangle", points, {"source": "ai", "病变类型": "疑似病变",
                         "score": detection["score"], "model": case["model"]}))
    if layer in {"manual", "combined"}:
        for mark in image["annotations"]:
            points = [(point["x"], point["y"]) for point in mark["points"]]
            marks.append((mark["kind"], points, {"source": "manual", "病变类型": mark["label"],
                         "annotation_id": mark["id"], "reviewer": mark["updated_by"]}))
    return marks


def overlay_png(raw, marks):
    with Image.open(BytesIO(raw)) as original:
        pixels = ImageOps.exif_transpose(original).convert("RGB")
    pixels.info.clear()
    draw = ImageDraw.Draw(pixels)
    width, height = pixels.size
    line = max(1, round(min(width, height) / 240))
    font = ImageFont.load_default(size=max(10, min(32, line * 5)))
    numbers = {"ai": 0, "manual": 0}
    for kind, points, attrs in marks:
        source = attrs["source"]
        numbers[source] += 1
        color, prefix = ("#ffbd58", "A") if source == "ai" else ("#54efac", "M")
        coords = [(max(0, min(width - 1, round(x))), max(0, min(height - 1, round(y)))) for x, y in points]
        if kind == "rectangle":
            draw.rectangle(coords, outline=color, width=line)
        else:
            draw.line(coords + [coords[0]], fill=color, width=line, joint="curve")
        label = prefix + str(numbers[source])
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        w, h = right - left + 4, bottom - top + 4
        x = max(0, min(coords[0][0], width - w))
        y = max(0, min(coords[0][1] - h, height - h))
        draw.rectangle((x, y, x + w - 1, y + h - 1), fill=color)
        draw.text((x + 2 - left, y + 2 - top), label, fill="#14211c", font=font)
    output = BytesIO()
    pixels.save(output, format="PNG")
    return output.getvalue()


def coco_region(number, kind, points, attrs):
    xs, ys = zip(*points)
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    segmentation, area = [], width * height
    if kind == "polygon":
        segmentation = [[coord for point in points for coord in point]]
        area = abs(sum(x * points[i-1][1] - y * points[i-1][0] for i, (x, y) in enumerate(points))) / 2
    return {"id": number, "image_id": 1, "category_id": 1, "segmentation": segmentation,
            "area": area, "bbox": [min(xs), min(ys), width, height], "iscrowd": 0,
            "attributes": {"occluded": False, "rotation": 0, **attrs}}


def training_zip(case, image, raw, mime, marks, layer):
    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}[mime]
    with Image.open(BytesIO(raw)) as source:
        if source.getexif().get(274, 1) in range(2, 9):
            oriented = ImageOps.exif_transpose(source)
            if oriented.mode == "CMYK":
                oriented = oriented.convert("RGB")
            oriented.info.clear()
            output = BytesIO()
            oriented.save(output, format="PNG")
            raw, ext = output.getvalue(), "png"
    filename = f"{case['id']}-{image['id']}.{ext}"
    document = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"case_id": case["id"], "image_id": image["id"], "original_filename": image["filename"], "layer": layer},
        "categories": [{"id": 1, "name": "病变", "supercategory": ""},
                       {"id": 2, "name": "annotation mark", "supercategory": ""}],
        "images": [{"id": 1, "width": image["width"], "height": image["height"], "file_name": filename,
                    "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}],
        "annotations": [coco_region(i, *mark) for i, mark in enumerate(marks, 1)]
    }
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("annotations/instances_default.json", json.dumps(document, ensure_ascii=False), ZIP_DEFLATED)
        archive.writestr("images/default/" + filename, raw, ZIP_STORED)
    return output.getvalue()
