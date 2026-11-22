"""Signed-case image annotations, in oriented original-image pixel coordinates."""
import json
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field


class ImagePoint(BaseModel):
    x: float = Field(ge=0, allow_inf_nan=False, strict=True)
    y: float = Field(ge=0, allow_inf_nan=False, strict=True)


class AnnotationEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["rectangle", "polygon"]
    points: list[ImagePoint] = Field(min_length=2, max_length=32)
    label: str = Field(min_length=1, max_length=80)
    revision: int = Field(strict=True, ge=0, le=2**63-2)

    def geometry(self, width, height, original=None):
        pts = [(p.x, p.y) for p in self.points]
        if not self.label.strip() or any(x >= width or y >= height for x, y in pts):
            raise HTTPException(422, "请填写名称，标注顶点须位于原图内")
        if self.kind == "rectangle":
            if len(pts) != 2 or pts[0][0] == pts[1][0] or pts[0][1] == pts[1][1]:
                raise HTTPException(422, "矩形需要两个对角点，宽高须大于零")
            xs, ys = zip(*pts)
            pts = [(min(xs), min(ys)), (max(xs), max(ys))]
        elif original is not None and pts == [(p["x"], p["y"]) for p in json.loads(original)]:
            return original  # A rename retains the saved boundary, including legacy polygons.
        elif len(pts) == 4:
            pts = largest_quadrilateral(pts)
            if not pts:
                raise HTTPException(422, "四点须互不重合，并能围成有面积的区域")
        else:
            raise HTTPException(422, "画四边形需要恰好四个顶点")
        return json.dumps([{"x": x, "y": y} for x, y in pts], allow_nan=False)


def cross(a, b, c):
    return (b[0]-a[0]) * (c[1]-a[1]) - (b[1]-a[1]) * (c[0]-a[0])


def intersects(a, b, c, d):
    if any(max(a[k], b[k]) < min(c[k], d[k]) or max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
        return False
    return cross(a, b, c) * cross(a, b, d) <= 0 and cross(c, d, a) * cross(c, d, b) <= 0


def simple_polygon(pts):
    n = len(pts)
    if n < 3 or len(set(pts)) != n:
        return False
    # Adjacent edges may share their endpoint but may not double back on each other.
    for i, curr in enumerate(pts):
        prev, after = pts[i-1], pts[(i+1) % n]
        if cross(prev, curr, after) == 0 and (
            (prev[0]-curr[0])*(after[0]-curr[0]) + (prev[1]-curr[1])*(after[1]-curr[1]) >= 0
        ):
            return False
    edges = list(zip(pts, pts[1:] + pts[:1]))
    for i, (a, b) in enumerate(edges):
        for j in range(i + 2, n):
            if (i, j) != (0, n-1) and intersects(a, b, *edges[j]):
                return False
    return twice_area(pts) != 0


def twice_area(pts):
    return sum(cross(pts[0], pts[i-1], pts[i]) for i in range(2, len(pts)))


def largest_quadrilateral(pts):
    # Starting from a fixed vertex, exactly three cycles remain up to reversal.
    a, b, c, d = sorted(pts)
    best, max_area = None, 0
    for ring in ([a,b,c,d], [a,b,d,c], [a,c,b,d]):
        if not simple_polygon(ring):
            continue
        area = twice_area(ring)
        if abs(area) > max_area:
            max_area = abs(area)
            best = ring if area > 0 else [ring[0], *ring[:0:-1]]
    return best


def prepare(database):
    database.execute("""CREATE TABLE IF NOT EXISTS image_annotations (
        case_number INTEGER NOT NULL, position INTEGER NOT NULL, id TEXT NOT NULL,
        kind TEXT NOT NULL, label TEXT NOT NULL, points TEXT NOT NULL, revision INTEGER NOT NULL,
        deleted INTEGER NOT NULL DEFAULT 0, created_by TEXT NOT NULL, created_at TEXT NOT NULL,
        updated_by TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(case_number,position,id),
        FOREIGN KEY(case_number,position) REFERENCES case_images(case_number,position) ON DELETE CASCADE)""")


def save(database, key, edit, image, actor, now):
    """Caller holds the write lock and has checked image access and signature status."""
    row = database.execute("""SELECT * FROM image_annotations
        WHERE case_number=? AND position=? AND id=?""", key).fetchone()
    original = row["points"] if row is not None and row["kind"] == "polygon" else None
    shape = edit.geometry(image["width"], image["height"], original)
    fields = (edit.kind, edit.label.strip(), shape)
    if row is None and edit.revision == 0:
        database.execute("""INSERT INTO image_annotations(case_number,position,id,kind,label,points,revision,
            created_by,created_at,updated_by,updated_at) VALUES (?,?,?,?,?,?,1,?,?,?,?)""",
            (*key, *fields, actor, now, actor, now))
    elif row is None or row["deleted"]:
        raise HTTPException(409, "标注已移除或不存在，请刷新影像")
    elif (row["revision"] - 1 == edit.revision and row["updated_by"] == actor
          and fields == (row["kind"], row["label"], row["points"])):
        return
    elif row["revision"] != edit.revision:
        raise HTTPException(409, "标注已被更新，请刷新影像后核对")
    elif fields == (row["kind"], row["label"], row["points"]):
        return
    else:
        database.execute("""UPDATE image_annotations SET kind=?, label=?, points=?, revision=revision+1,
            updated_by=?, updated_at=? WHERE case_number=? AND position=? AND id=?""", (*fields, actor, now, *key))
    bump_revision(database, key[0])


def remove(database, key, revision, actor, now):
    row = database.execute("""SELECT revision, deleted, updated_by FROM image_annotations
        WHERE case_number=? AND position=? AND id=?""", key).fetchone()
    if row is None:
        raise HTTPException(404, "未找到此影像的标注")
    if row["deleted"] and row["revision"] == revision + 1 and row["updated_by"] == actor:
        return
    if row["deleted"] or row["revision"] != revision:
        raise HTTPException(409, "标注已变化，请刷新影像后核对")
    database.execute("""UPDATE image_annotations SET deleted=1, revision=revision+1, updated_by=?, updated_at=?
        WHERE case_number=? AND position=? AND id=?""", (actor, now, *key))
    bump_revision(database, key[0])


def bump_revision(database, number):
    database.execute("UPDATE cases SET annotation_revision=annotation_revision+1 WHERE number=?", (number,))
