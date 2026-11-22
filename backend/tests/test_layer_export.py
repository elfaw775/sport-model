from io import BytesIO
import json
from uuid import uuid4
from zipfile import ZipFile

from PIL import Image

from backend import detector, screening
from backend.tests.test_api import append_image, client, login, picture, upload


def unpack(response):
    assert response.status_code == 200, response.text[:400]
    assert response.headers["content-type"] == "application/zip"
    with ZipFile(BytesIO(response.content)) as archive:
        data = json.loads(archive.read("annotations/instances_default.json"))
        filename = data["images"][0]["file_name"]
        assert sorted(archive.namelist()) == ["annotations/instances_default.json", "images/default/" + filename]
        return data, archive.read("images/default/" + filename)


def test_archived_layer_exports(client, monkeypatch):
    raw = picture(240, 220)
    login(client)
    case_id = upload(client, raw).json()["id"]
    case = "/api/cases/" + case_id
    primary = case + "/images/primary"
    annotation = str(uuid4())
    points = [{"x": 130, "y": 110}, {"x": 170, "y": 110}, {"x": 190, "y": 170}, {"x": 110, "y": 170}]
    saved = client.put(primary + "/annotations/" + annotation, json={
        "kind": "polygon", "label": "人工可疑区域", "points": points, "revision": 0})
    assert saved.is_success, saved.text
    assert append_image(client, case, picture(80, 60)).is_success
    monkeypatch.setattr(detector, "detect", lambda _: [{"x1": 20, "y1": 50, "x2": 40, "y2": 90, "score": 0.7}])
    monkeypatch.setattr(screening, "classify", lambda _: {"model": "demo", "score": .4, "threshold": .2, "predicted_positive": True})
    assert client.post(case + "/analyze").is_success
    context = client.get(case + "?include_signed_evidence=true").json()
    signed = client.post(case + "/review", json={"findings": "已复核", "conclusion": "", "revision": 0,
                         "review_context": context["review_context"], "annotation_revision": context["annotation_revision"]})
    assert signed.is_success, signed.text
    assert client.post(case + "/archive").is_success
    for layer, expected_sources in [("combined", ["ai", "manual"]), ("ai", ["ai"]), ("manual", ["manual"])]:
        base = primary + "/layers/" + layer
        document, original = unpack(client.get(base + ".zip"))
        assert original == raw
        assert document["info"]["layer"] == layer
        assert [c["name"] for c in document["categories"]] == ["病变", "annotation mark"]
        assert [entry["attributes"]["source"] for entry in document["annotations"]] == expected_sources
        for entry in document["annotations"]:
            if entry["attributes"]["source"] == "ai":
                assert entry["area"] == 800 and entry["bbox"] == [20, 50, 20, 40] and entry["segmentation"] == []
                assert entry["attributes"]["score"] == 0.7
            else:
                assert entry["area"] == 3600 and entry["bbox"] == [110, 110, 80, 60]
                assert entry["segmentation"] == [[110, 170, 130, 110, 170, 110, 190, 170]]
                assert entry["attributes"]["annotation_id"] == annotation
        png = client.get(base + ".png")
        assert png.is_success and png.headers["content-type"] == "image/png"
        with Image.open(BytesIO(png.content)) as pixels:
            assert pixels.size == (240, 220)
            assert (pixels.getpixel((40, 70)) == (255, 189, 88)) == ("ai" in expected_sources)
            assert (pixels.getpixel((150, 170)) == (84, 239, 172)) == ("manual" in expected_sources)
    companion, _ = unpack(client.get(case + "/images/1/layers/combined.zip"))
    assert companion["annotations"] == []
    assert append_image(client, case, picture(70, 55)).is_success
    assert client.get(case + "/images/2/layers/manual.zip").status_code == 404
    assert client.get(primary + "/layers/unknown.zip").status_code == 404
    assert client.get(primary).content == raw
    client.post("/api/logout")
    assert client.get(primary + "/layers/combined.zip").status_code == 401


def test_training_export_aligns_exif_image_with_annotation_coordinates(client):
    login(client)
    source = Image.new("RGB", (35, 18), "red")
    source.paste((0, 0, 255), (0, 9, 35, 18))
    orientation = Image.Exif()
    orientation[274] = 6
    buf = BytesIO()
    source.save(buf, "JPEG", exif=orientation)
    case_id = upload(client, buf.getvalue()).json()["id"]
    data, raw = unpack(client.get("/api/cases/" + case_id + "/images/primary/layers/manual.zip"))
    assert data["annotations"] == []
    image_info = data["images"][0]
    assert (image_info["width"], image_info["height"]) == (18, 35)
    assert image_info["file_name"].endswith(".png")
    with Image.open(BytesIO(raw)) as image:
        assert image.size == (18, 35) and not image.getexif().get(274)
        assert image.getpixel((1, 8))[2] > 225
        assert image.getpixel((16, 8))[0] > 225
