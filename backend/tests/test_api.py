import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import detector
from backend.app import MAX_BYTES, create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SPORT_MODEL_USERNAME", "demo")
    monkeypatch.setenv("SPORT_MODEL_PASSWORD", "test-password")
    monkeypatch.setenv("SPORT_MODEL_DISPLAY_NAME", "王一伊")
    monkeypatch.delenv("SPORT_MODEL_SECURE_COOKIE", raising=False)
    with TestClient(create_app(tmp_path)) as session:
        yield session


def login(client, username="demo"):
    result = client.post("/api/login", json={"username": username, "password": "test-password"})
    assert result.status_code == 200
    return result


def picture(width=32, height=48):
    stream = io.BytesIO()
    Image.new("L", (width, height), 17).save(stream, "PNG")
    return stream.getvalue()


def upload(client, content=None, **metadata):
    data = {"horse_name": "追风", "projection": "侧位", "quality_confirmed": "true"} | metadata
    return client.post("/api/cases", data=data,
                       files={"image": ("../xray.png", picture() if content is None else content, "image/png")})


def test_auth_validation_and_revocation(client):
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/cases").status_code == 401
    assert upload(client).status_code == 401
    assert client.post("/api/login", json={"username": "demo", "password": "no"}).status_code == 401
    cookie = login(client).headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert client.get("/api/me").json()["display_name"] == "王一伊"
    token = client.cookies["equivision_session"]
    assert client.post("/api/logout", headers={"origin": "https://other.example"}).status_code == 403
    assert client.post("/api/logout").status_code == 204
    assert client.get("/api/cases", headers={"cookie": f"equivision_session={token}"}).status_code == 401


def test_rejects_invalid_images_without_archiving(client):
    login(client)
    assert upload(client, b"not an image").status_code == 422
    assert upload(client, picture(8001, 1)).status_code == 413
    assert upload(client, bytes(MAX_BYTES + 1)).status_code == 413
    assert upload(client, quality_confirmed="false").status_code == 422
    assert upload(client, projection="invalid").status_code == 422
    assert upload(client, horse_name="  ").status_code == 422
    assert client.get("/api/cases").json() == []


def test_workflow_persistence_and_ownership(client, tmp_path, monkeypatch):
    login(client)
    payload = picture()
    response = upload(client, payload)
    assert response.status_code == 201, response.text
    case = response.json()
    assert (case["id"], case["filename"], case["quality_confirmed"], case["projection"]) == (
        "EV-000001", "xray.png", True, "侧位")
    assert (case["status"], case["width"], case["height"], case["model"]) == ("uploaded", 32, 48, None)
    resource = f"/api/cases/{case['id']}"
    assert client.get(resource + "/image").content == payload
    assert client.post(resource + "/review", json={"note": "太早"}).status_code == 409

    def unavailable(_image):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(detector, "detect", unavailable)
    assert client.post(resource + "/analyze").status_code == 503
    assert client.get(resource).json()["status"] == "uploaded"
    predicted = [{"x1": 3.0, "y1": 4.0, "x2": 8.0, "y2": 12.0, "score": 0.82}]
    monkeypatch.setattr(detector, "detect", lambda image: predicted if image.size == (32, 48) else [])
    analyzed = client.post(resource + "/analyze").json()
    assert analyzed["detections"] == predicted and analyzed["model"] == detector.MODEL_ID
    assert analyzed["status"] == "analyzed" and analyzed["analyzed_at"]
    assert client.post(resource + "/analyze").status_code == 409
    assert client.post(resource + "/review", json={"note": "   "}).status_code == 422
    reviewed = client.post(resource + "/review", json={"note": "  已核对原片  "}).json()
    assert reviewed["status"] == "reviewed" and reviewed["reviewed_at"]
    assert (reviewed["reviewer"], reviewed["review_note"], reviewed["detections"]) == ("王一伊", "已核对原片", predicted)
    assert client.post(resource + "/review", json={"note": "改写"}).status_code == 409
    with TestClient(create_app(tmp_path)) as restarted:
        login(restarted)
        assert restarted.get("/api/cases").json() == [reviewed]
        assert restarted.get(resource + "/image").content == payload
    monkeypatch.setenv("SPORT_MODEL_USERNAME", "another")
    with TestClient(create_app(tmp_path)) as another:
        login(another, "another")
        assert another.get("/api/cases").json() == []
        assert another.get(resource).status_code == 404
        assert another.get(resource + "/image").status_code == 404
        assert another.post(resource + "/analyze").status_code == 404
        assert another.post(resource + "/review", json={"note": "test"}).status_code == 404
