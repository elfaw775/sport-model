"""Render saved manual regions on a copy of the oriented source radiograph."""
from io import BytesIO
import json

from PIL import Image, ImageDraw, ImageFont, ImageOps, PngImagePlugin


def annotated_png(original, annotations):
    with Image.open(BytesIO(original)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.info.clear()
    width, height = image.size
    thickness = max(1, round(min(width, height) / 240))
    font = ImageFont.load_default(size=max(10, min(32, thickness * 5)))
    draw = ImageDraw.Draw(image)
    for number, mark in enumerate(annotations, 1):
        points = [(min(width - 1, round(p["x"])), min(height - 1, round(p["y"]))) for p in mark["points"]]
        if mark["kind"] == "rectangle":
            draw.rectangle(points, outline="#07392b", width=thickness + 1)
            draw.rectangle(points, outline="#54efac", width=thickness)
        elif mark["kind"] == "polygon":
            draw.line([*points, points[0]], fill="#07392b", width=thickness + 2)
            draw.line([*points, points[0]], fill="#54efac", width=thickness)
        else:
            raise ValueError("Unsupported manual annotation shape")
        text = str(number)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        x = max(0, min(points[0][0], width - (right - left) - 4))
        y = max(0, min(points[0][1] - (bottom - top) - 4, height - (bottom - top) - 4))
        draw.rectangle((x, y, x + right-left + 3, y + bottom-top + 3), fill="#07392b")
        draw.text((x + 2-left, y + 2-top), text, font=font, fill="white")
    metadata = PngImagePlugin.PngInfo()
    legend = [{"number": i, "id": m["id"], "label": m["label"], "kind": m["kind"],
               "points": m["points"], "revision": m["revision"]} for i, m in enumerate(annotations, 1)]
    metadata.add_itxt("EQUIVISION.manual_annotations", json.dumps(legend, ensure_ascii=False))
    stream = BytesIO()
    image.save(stream, format="PNG", pnginfo=metadata)
    return stream.getvalue()
