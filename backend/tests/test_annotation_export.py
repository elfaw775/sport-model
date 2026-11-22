import io
import json
from uuid import uuid4

from PIL import Image

from backend.tests.test_api import append_image, client, login, picture, upload


def test_export_saved_regions_from_one_image_and_preserve_original(client):
    login(client)
    data = picture(width=140, height=110)
    case = upload(client, data).json()
    case_url = "/api/cases/" + case["id"]
    append_image(client, case_url, picture(60, 80))
    image = case_url + "/images/primary"
    url = image + "/annotations.png"
    assert client.get(url).status_code == 409
    for kind, xy in [("rectangle", [(30, 30), (50, 60)]), ("polygon", [(65, 20), (85, 20), (90, 40), (60, 40)])]:
        created = client.put(image + "/annotations/" + str(uuid4()), json={
            "kind": kind, "label": "复核区域", "revision": 0, "points": [{"x": x, "y": y} for x, y in xy]
        })
        assert created.status_code == 200, created.text
    result = client.get(url)
    assert result.status_code == 200 and result.headers["content-type"] == "image/png"
    assert result.headers["content-disposition"] == f'attachment; filename="{case["id"]}-primary-annotations.png"'
    with Image.open(io.BytesIO(result.content)) as rendered:
        assert rendered.size == (140, 110)
        assert rendered.getpixel((50, 50)) != (17, 17, 17)  # Right edge of rectangle
        assert rendered.getpixel((75, 40)) != (17, 17, 17)  # Bottom edge of quadrilateral
        assert rendered.getpixel((100, 85)) == (17, 17, 17)
        legend = json.loads(rendered.info["EQUIVISION.manual_annotations"])
        assert sorted(item["kind"] for item in legend) == ["polygon", "rectangle"]
        assert [item["number"] for item in legend] == [1, 2]
        assert all(item["label"] == "复核区域" for item in legend)
    assert client.get(image).content == data
    assert client.get(case_url + "/images/1/annotations.png").status_code == 409
    assert client.get(case_url + "/images/987/annotations.png").status_code == 404
    client.post("/api/logout")
    assert client.get(url).status_code == 401
