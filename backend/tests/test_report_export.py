import sqlite3
from uuid import uuid4

from fastapi.testclient import TestClient

from backend import detector, screening
from backend.app import create_app
from backend.tests.test_api import append_image, login, picture, rectangle, sign_in_member, team, upload


def test_signed_report_freezes_evidence_and_downloads_offline(team, tmp_path, monkeypatch):
    sign_in_member(team, "wangyiyi")
    case = upload(team, horse_name="追风 <HF>").json()
    path = "/api/cases/" + case["id"]
    download = path + "/report/download"
    assert team.get(download).status_code == 409
    append_image(team, path, picture(61, 80))
    mark = team.put(path + "/images/1/annotations/" + str(uuid4()), json=rectangle("疑似区域 <1>"))
    assert mark.status_code == 200
    measure = team.post(path + "/images/1/measurements",
                        json={"kind": "distance", "points": [{"x": 1, "y": 2}, {"x": 4, "y": 6}]})
    assert measure.status_code == 201
    monkeypatch.setattr(screening, "classify", lambda _: {
        "model": "DINO-ref", "score": .63, "threshold": .1788, "predicted_positive": True})
    monkeypatch.setattr(detector, "detect", lambda _: [{"x1": 3, "y1": 6, "x2": 8, "y2": 12, "score": .34}])
    assert team.post(path + "/analyze").status_code == 200
    sign_in_member(team, "zhushijia")
    signed = team.post(path + "/review", json={
        "findings": "主图可见 <候选>", "conclusion": "建议复查", "revision": 0, "annotation_revision": 1})
    assert signed.status_code == 200
    result = team.get(download)
    assert result.status_code == 200 and result.headers["content-type"].startswith("text/html")
    assert result.headers["content-disposition"] == f'attachment; filename="{case["id"]}-report.html"'
    for value in ("追风 &lt;HF&gt;", "DINO-ref", ".1788", detector.MODEL_ID, "0.34", "朱诗嘉",
                  "附图 1", "5.00 px", "疑似区域 &lt;1&gt;", "主图可见 &lt;候选&gt;", "建议复查"):
        assert value in result.text
    assert "签署时保存" in result.text and "<候选>" not in result.text
    assert team.patch(path, json={"horse_name": "补录名"}).status_code == 200
    assert append_image(team, path, picture(75, 90)).status_code == 201
    assert team.post(path + "/archive").status_code == 200
    assert "signed_evidence" not in team.get("/api/cases").json()[0]
    with TestClient(create_app(tmp_path)) as other:
        assert other.get(download).status_code == 401
        sign_in_member(other, "liqihang")
        detail = other.get(path + "?include_signed_evidence=true").json()
        assert detail["horse_name"] == "补录名"
        assert len(detail["images"]) == 3 and len(detail["signed_evidence"]["images"]) == 2
        assert detail["signed_evidence"]["horse_name"] == "追风 <HF>"
        assert detail["signed_evidence"]["images"] == signed.json()["images"]
        assert other.get(download).content == result.content
    monkeypatch.setenv("SPORT_MODEL_USERNAME", "other")
    monkeypatch.setenv("SPORT_MODEL_PASSWORD", "test-password")
    with TestClient(create_app(tmp_path)) as private:
        login(private, "other")
        assert private.get(download).status_code == 404


def test_blank_and_legacy_reports(team, tmp_path, monkeypatch):
    sign_in_member(team, "wangyiyi")
    case = upload(team).json()
    path = "/api/cases/" + case["id"]
    monkeypatch.setattr(screening, "classify", lambda _: {
        "model": "DINO-ref", "score": .2, "threshold": .1, "predicted_positive": True})
    monkeypatch.setattr(detector, "detect", lambda _: [])
    assert team.post(path + "/analyze").status_code == 200
    assert team.post(path + "/review", json={"note": ""}).status_code == 422
    assert team.get(path + "/report/download").status_code == 409
    assert team.post(path + "/review", json={"note": "", "empty_review_confirmed": True}).status_code == 200
    assert "已二次确认两项留空" in team.get(path + "/report/download").text

    # Older workbenches kept the review and score without a signature snapshot or box coordinates.
    with sqlite3.connect(tmp_path / "cases.sqlite3") as db:
        db.execute("DELETE FROM signed_reports")
        db.execute("""UPDATE cases SET screening=NULL, detections='[{"score":0.45}]'""")
    assert team.get(path + "?include_signed_evidence=true").json()["signed_evidence"] is None
    legacy = team.get(path + "/report/download")
    assert legacy.status_code == 200
    assert "历史报告 · 按现存病例资料整理" in legacy.text and "未保存整片筛查结果" in legacy.text
