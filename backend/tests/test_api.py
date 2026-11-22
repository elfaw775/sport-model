import io
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import detector, screening
from backend.app import MAX_BYTES, create_app, demo_library


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


@pytest.fixture
def team(tmp_path, monkeypatch):
    for key in ("SPORT_MODEL_USERNAME", "SPORT_MODEL_PASSWORD", "SPORT_MODEL_DISPLAY_NAME", "SPORT_MODEL_SECURE_COOKIE"):
        monkeypatch.delenv(key, raising=False)
    with TestClient(create_app(tmp_path)) as session:
        yield session


def sign_in_member(client, username):
    result = client.post("/api/login", json={"username": username, "password": username + "sportsHorse"})
    assert result.status_code == 200
    return result.json()


def picture(width=32, height=48):
    stream = io.BytesIO()
    Image.new("L", (width, height), 17).save(stream, "PNG")
    return stream.getvalue()


def upload(client, content=None, **metadata):
    data = {"horse_name": "追风", "projection": "侧位", "quality_confirmed": "true"} | metadata
    return client.post("/api/cases", data=data,
                       files={"image": ("../xray.png", picture() if content is None else content, "image/png")})


@pytest.mark.parametrize("name,ref_score,positive", [
    ("16-侧位.jpg", 0.9996267822680424, True),
    ("7-RF-正位.jpg", 0.002013474966457373, False),
])
def test_real_screening_and_localization_on_main_image(client, name, ref_score, positive):
    # CPU PyTorch reference from the frozen DINO encoder and the published all-84-film head.
    source = Path(__file__).resolve().parents[2] / "training/data/sporthorse_v2/images/test" / name
    login(client)
    case = upload(client, source.read_bytes()).json()
    path = "/api/cases/" + case["id"]
    assert case["screening"] is None
    response = client.post(path + "/analyze")
    assert response.status_code == 200, response.text
    analyzed = response.json()
    film = analyzed["screening"]
    assert film["model"].startswith("DINOv2-Small-refit84@")
    assert film["threshold"] == 0.17883443932013382
    assert film["score"] == pytest.approx(ref_score, abs=1e-6)
    assert film["predicted_positive"] is positive
    assert analyzed["model"] == detector.MODEL_ID and isinstance(analyzed["detections"], list)
    assert client.get(path).json() == analyzed


def test_demo_images_are_authenticated_and_read_only(client):
    names = ("16-侧位.jpg", "23-LF-正位.jpg")
    assert client.get("/api/demo-images").status_code == 401
    assert client.get("/api/demo-images/XR-001.jpg").status_code == 401
    assert client.get("/api/demo-images/" + names[0]).status_code == 401
    login(client)
    catalog = client.get("/api/demo-images").json()
    assert len(catalog) == len({item["horse_name"] for item in catalog}) == 84
    assert len({item["url"] for item in catalog}) == 84
    images = Path(__file__).resolve().parents[2] / "training/data/sporthorse_v2/images"
    installed = {path.read_bytes(): path for path in images.glob("*/*") if path.suffix in {".jpg", ".png"}}
    for item in catalog:
        assert set(item) == {"filename", "projection", "horse_name", "url"}
        assert re.fullmatch(r"XR-\d{3}\.(jpg|png)", item["filename"])
        assert item["url"] == "/api/demo-images/" + item["filename"]
        response = client.get(item["url"])
        assert response.status_code == 200
        source = installed.pop(response.content)
        assert item["filename"].endswith(source.suffix)
        assert response.headers["content-type"] == ("image/png" if source.suffix == ".png" else "image/jpeg")
        if "+" in source.stem:
            assert item["projection"] == ""
        else:
            assert item["projection"] in ("正位", "侧位", "斜位") and item["projection"] in source.stem
    assert not installed
    for name in names:
        response = client.get("/api/demo-images/" + name)
        source = Path(__file__).resolve().parents[2] / "training/data/sporthorse_v2/images/test" / name
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
        assert response.content == source.read_bytes()
        canonical = response.headers["Content-Location"]
        assert canonical in {item["url"] for item in catalog}
        assert client.get(canonical).content == response.content
    assert client.get("/api/demo-images/7-RF-正位.jpg").status_code == 404
    assert client.get("/api/cases").json() == []


def test_demo_image_numbers_stable_with_partial_dataset(tmp_path):
    (tmp_path / "split_manifest.tsv").write_text("image\tsplit\nA-正位\tval\nB-侧位\ttest\n", encoding="utf-8")
    test_dir = tmp_path / "images/test"
    test_dir.mkdir(parents=True)
    (test_dir / "B-侧位.jpg").touch()
    catalog, _ = demo_library(tmp_path)
    assert [entry["filename"] for entry in catalog] == ["XR-002.jpg"]


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


def test_members_share_cases_but_keep_distinct_creator_and_reviewer(team, monkeypatch):
    members = {"wangyiyi": "王一伊", "zhushijia": "朱诗嘉", "alimire": "阿里米热", "liqihang": "李其杭"}
    assert team.get("/api/login-options").json() == members
    for username, password in (("demo", "EquiDemo2026!"), ("wangyiyi", "wangyiyisportshorse"),
                               ("Wangyiyi", "wangyiyisportsHorse"), ("liqihang", "zhushijiasportsHorse")):
        assert team.post("/api/login", json={"username": username, "password": password}).status_code == 401
    monkeypatch.setattr(detector, "detect", lambda _image: [])
    for username, name in members.items():
        identity = sign_in_member(team, username)
        assert identity == team.get("/api/me").json() == {"username": username, "display_name": name}

    sign_in_member(team, "wangyiyi")
    case = team.post("/api/visits", json={"horse_name": "追风"}).json()
    path = "/api/cases/" + case["id"]
    assert case["created_by"] == "wangyiyi"
    for username in members:
        sign_in_member(team, username)
        assert team.get("/api/cases").json() == [case]
        assert team.get(path).json() == case
    sign_in_member(team, "zhushijia")
    assert team.patch(path, json={"comment": "他人补录"}).json()["comment"] == "他人补录"
    primary, follow_up = picture(), picture(73, 40)
    assert append_image(team, path, primary).status_code == 201
    images = append_image(team, path, follow_up).json()["images"]
    sign_in_member(team, "alimire")
    assert team.get(path + "/image").content == team.get(images[0]["url"]).content == primary
    assert team.get(images[1]["url"]).content == follow_up
    assert team.post(path + "/archive").status_code == 409
    assert team.post(path + "/analyze").json()["status"] == "analyzed"
    sign_in_member(team, "liqihang")
    reviewed = team.post(path + "/review", json={"note": "已核对"}).json()
    assert reviewed["reviewer"] == "李其杭" and reviewed["created_by"] == "wangyiyi"
    team.post("/api/logout")
    assert team.get("/api/cases").status_code == team.get(path).status_code == 401
    assert team.get(images[1]["url"]).status_code == team.delete(path).status_code == 401
    sign_in_member(team, "zhushijia")
    assert team.get("/api/cases").json() == [reviewed]
    archived = team.post(path + "/archive").json()
    assert archived["archived_at"] and archived == reviewed | {"archived_at": archived["archived_at"]}
    assert team.delete(path).status_code == 204
    sign_in_member(team, "wangyiyi")
    assert not team.get("/api/cases").json() and team.get(images[1]["url"]).status_code == 404


def test_default_team_migrates_only_legacy_owners_once_and_revokes_old_sessions(client, tmp_path, monkeypatch):
    assert client.get("/api/login-options").json() == {}
    legacy_token = login(client).cookies["equivision_session"]
    payload = picture()
    legacy = upload(client, payload).json()
    path = "/api/cases/" + legacy["id"]
    monkeypatch.setattr(detector, "detect", lambda _image: [])
    assert client.post(path + "/analyze").status_code == 200
    prior = client.post(path + "/review", json={"note": "原复核意见"}).json()
    with sqlite3.connect(tmp_path / "cases.sqlite3") as database:
        database.execute("INSERT INTO cases(owner, horse_name, created_at) VALUES ('阿里米热','另一匹马','2026-10-01')")
        database.execute("INSERT INTO cases(owner, horse_name, created_at) VALUES ('other','不明归属','2026-10-01')")
    for key in ("SPORT_MODEL_USERNAME", "SPORT_MODEL_PASSWORD", "SPORT_MODEL_DISPLAY_NAME"):
        monkeypatch.delenv(key)
    with TestClient(create_app(tmp_path)) as upgraded:
        assert upgraded.get("/api/me", headers={"cookie": f"equivision_session={legacy_token}"}).status_code == 401
        sign_in_member(upgraded, "alimire")
        migrated = prior | {"created_by": "wangyiyi"}
        shared = upgraded.get("/api/cases").json()
        assert [(case["horse_name"],case["created_by"]) for case in shared] == [("另一匹马","alimire"),("追风","wangyiyi")]
        assert upgraded.get(path).json() == migrated
        sign_in_member(upgraded, "wangyiyi")
        assert upgraded.get("/api/cases").json() == shared
        assert upgraded.get(path + "/image").content == payload
    with sqlite3.connect(tmp_path / "cases.sqlite3") as database:
        database.execute("INSERT INTO cases(owner, horse_name, created_at) VALUES ('demo','后来新增','2026-11-18')")
    with TestClient(create_app(tmp_path)) as restarted:
        sign_in_member(restarted, "wangyiyi")
        assert restarted.get("/api/cases").json() == shared


def test_team_and_custom_accounts_using_one_database_cannot_access_each_others_cases(team, tmp_path, monkeypatch):
    sign_in_member(team, "wangyiyi")
    team_case = upload(team).json()
    monkeypatch.setenv("SPORT_MODEL_USERNAME", "external")
    monkeypatch.setenv("SPORT_MODEL_PASSWORD", "test-password")
    with TestClient(create_app(tmp_path)) as private:
        login(private, "external")
        private_case = upload(private).json()
        for client, own, hidden in [(team, team_case, private_case), (private, private_case, team_case)]:
            path = "/api/cases/" + hidden["id"]
            assert client.get("/api/cases").json() == [own]
            assert client.get(path).status_code == client.get(path + "/image").status_code == 404
            assert client.get(hidden["images"][0]["url"]).status_code == 404
            assert client.patch(path, json={"comment": "不能修改"}).status_code == 404
            assert append_image(client, path, picture()).status_code == 404
            assert client.post(path + "/archive").status_code == client.post(path + "/analyze").status_code == 404
            assert client.post(path + "/review", json={"note": "不能复核"}).status_code == client.delete(path).status_code == 404


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

    monkeypatch.setattr(screening, "classify", unavailable)
    failure = client.post(resource + "/analyze")
    assert failure.status_code == 503 and "DINOv2" in failure.json()["detail"]
    assert client.get(resource).json() == case
    # A negative film screen and a positive YOLO box are two distinct findings.
    film = {"model": "DINOv2-test", "score": 0.10, "threshold": 0.18, "predicted_positive": False}
    monkeypatch.setattr(screening, "classify", lambda image: film)
    monkeypatch.setattr(detector, "detect", unavailable)
    failure = client.post(resource + "/analyze")
    assert failure.status_code == 503 and "YOLO11s" in failure.json()["detail"]
    assert client.get(resource).json() == case
    predicted = [{"x1": 3.0, "y1": 4.0, "x2": 8.0, "y2": 12.0, "score": 0.82}]
    monkeypatch.setattr(detector, "detect", lambda image: predicted if image.size == (32, 48) else [])
    analyzed = client.post(resource + "/analyze").json()
    assert analyzed["detections"] == predicted and analyzed["model"] == detector.MODEL_ID
    assert analyzed["screening"] == film
    assert analyzed["status"] == "analyzed" and analyzed["analyzed_at"]
    assert client.post(resource + "/analyze").status_code == 409
    assert client.post(resource + "/archive").status_code == 409
    assert client.post(resource + "/review", json={"note": "   "}).status_code == 422
    assert client.get(resource).json() == analyzed
    reviewed = client.post(resource + "/review", json={"note": "  已核对原片  "}).json()
    assert reviewed["status"] == "reviewed" and reviewed["reviewed_at"] and reviewed["archived_at"] is None
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


def append_image(client, path, payload, filename="附片.png", projection="正位"):
    return client.post(path + "/images", data={"projection": projection, "quality_confirmed": "true"},
                       files={"image": (filename, payload, "image/png")})


def test_manual_visits_can_be_archived_edited_and_deleted_without_inference(client, tmp_path):
    login(client)
    assert client.post("/api/visits", json={"horse_name": "  "}).status_code == 422
    case = client.post("/api/visits", json={"horse_name": "追风", "comment": "初诊"}).json()
    path = "/api/cases/" + case["id"]
    assert not case["images"] and not case["has_image"] and case["status"] == "registered"
    assert client.get(path + "/image").status_code == 404
    assert client.post(path + "/analyze").status_code == 409
    archived = client.post(path + "/archive").json()
    assert archived["archived_at"] and archived["reviewed_at"] is None and archived["model"] is None
    assert client.post(path + "/archive").json() == archived
    assert client.patch(path, json={"horse_name": "  "}).status_code == 422
    assert client.patch(path, json={}).status_code == 422
    changed = client.patch(path, json={"horse_name": "追风 II", "comment": "病史已补充"}).json()
    assert (changed["horse_name"], changed["comment"], changed["archived_at"]) == ("追风 II", "病史已补充", archived["archived_at"])
    assert client.patch(path, json={"comment": ""}).json()["comment"] == ""
    first, second = picture(), picture(70, 30)
    assert append_image(client, path, b"invalid").status_code == 422
    assert client.get(path).json()["archived_at"] == archived["archived_at"]
    attached = append_image(client, path, first).json()
    assert attached["has_image"] and attached["status"] == "uploaded" and attached["archived_at"] is None
    assert client.post(path + "/archive").status_code == 409
    assert attached["images"][0]["id"] == "primary"
    response = append_image(client, path, second, "复查.png", "侧位")
    assert response.status_code == 201
    images = response.json()["images"]
    assert [(item["id"], item["projection"]) for item in images] == [("primary", "正位"), ("1", "侧位")]
    assert client.get(path + "/image").content == client.get(images[0]["url"]).content == first
    assert client.get(images[1]["url"]).content == second
    other = client.post("/api/visits", json={"horse_name": "独立病例"}).json()
    second_path = "/api/cases/" + other["id"]
    only_image = append_image(client, second_path, second).json()["images"]
    assert len(only_image) == 1 and only_image[0]["id"] == "primary"
    assert client.get(second_path + "/images/1").status_code == 404
    assert client.delete(path).status_code == 204
    assert client.delete(path).status_code == 404
    assert all(client.get(item["url"]).status_code == 404 for item in images)
    assert [row["id"] for row in client.get("/api/cases").json()] == [other["id"]]
    with sqlite3.connect(tmp_path / "cases.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM case_images WHERE case_number=1").fetchone() == (0,)


def test_edit_and_upload_do_not_change_reviewed_primary_image(client, monkeypatch):
    login(client)
    primary = picture()
    path = "/api/cases/" + upload(client, primary).json()["id"]
    monkeypatch.setattr(detector, "detect", lambda _: [{"x1": 1, "y1": 2, "x2": 5, "y2": 6, "score": 0.4}])
    assert client.post(path + "/analyze").status_code == 200
    reviewed = client.post(path + "/review", json={"note": "兽医已复核"}).json()
    assert reviewed["reviewed_at"] and reviewed["archived_at"] is None
    archived = client.post(path + "/archive").json()
    assert archived["archived_at"] and client.post(path + "/archive").json() == archived
    edited = client.patch(path, json={"comment": "复查预约"}).json()
    assert edited["horse_name"] == reviewed["horse_name"]
    update = append_image(client, path, picture(65, 30), "后续复查.png").json()
    for key in ("status", "detections", "model", "reviewer", "reviewed_at", "review_note", "archived_at", "width", "height"):
        assert update[key] == archived[key]
    assert client.get(path + "/image").content == primary and len(update["images"]) == 2


def test_legacy_unreviewed_image_archive_keeps_its_original_history(client, tmp_path, monkeypatch):
    login(client)
    path = "/api/cases/" + upload(client).json()["id"]
    previous_time = "2026-10-01T01:02:03+00:00"
    with sqlite3.connect(tmp_path / "cases.sqlite3") as database:
        database.execute("UPDATE cases SET archived_at=?", (previous_time,))
    case = client.get(path).json()
    assert case["status"] == "uploaded" and case["reviewed_at"] is None
    assert client.post(path + "/archive").json() == case
    attached = append_image(client, path, picture()).json()
    assert attached["archived_at"] == previous_time and attached["reviewer"] is None
    monkeypatch.setattr(detector, "detect", lambda _: [])
    analyzed = client.post(path + "/analyze").json()
    assert analyzed["status"] == "analyzed" and client.post(path + "/archive").json() == analyzed
    reviewed = client.post(path + "/review", json={"note": "补充复核"}).json()
    assert reviewed["archived_at"] == previous_time and reviewed["reviewed_at"] and reviewed["reviewer"] == "王一伊"


def test_case_edits_and_attachments_require_owner_and_same_origin(client, tmp_path, monkeypatch):
    login(client)
    case = upload(client).json()
    path = "/api/cases/" + case["id"]
    cross_site = {"origin": "https://other.example"}
    assert client.patch(path, json={"comment": "变更"}, headers=cross_site).status_code == 403
    assert client.delete(path, headers=cross_site).status_code == 403
    monkeypatch.setenv("SPORT_MODEL_USERNAME", "other")
    with TestClient(create_app(tmp_path)) as stranger:
        assert stranger.post("/api/visits", json={"horse_name": "anon"}).status_code == 401
        login(stranger, "other")
        assert stranger.patch(path, json={"comment": "test"}).status_code == 404
        assert append_image(stranger, path, picture()).status_code == 404
        assert stranger.get(case["images"][0]["url"]).status_code == 404
        assert stranger.post(path + "/archive").status_code == 404
        assert stranger.delete(path).status_code == 404
    assert client.get(path).json() == case


def test_startup_migrates_legacy_database_and_preserves_history(tmp_path, monkeypatch):
    from contextlib import closing

    monkeypatch.setenv("SPORT_MODEL_USERNAME", "demo")
    monkeypatch.setenv("SPORT_MODEL_PASSWORD", "test-password")
    payload = picture()
    previous_time = "2026-10-01T01:02:03+00:00"
    with closing(sqlite3.connect(tmp_path / "cases.sqlite3")) as database, database:
        database.execute("""CREATE TABLE cases(number INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT, horse_name TEXT,
            filename TEXT, width INTEGER, height INTEGER, projection TEXT, quality_confirmed INTEGER, mime TEXT,
            original BLOB, created_at TEXT, status TEXT, detections TEXT, model TEXT, threshold REAL,
            analyzed_at TEXT, reviewer TEXT, review_note TEXT, reviewed_at TEXT)""")
        database.execute("""INSERT INTO cases VALUES (7,'demo','旧病例','old.png',32,48,'侧位',1,'image/png',?,
            ?,'reviewed','[{"score":0.45}]','original-model',0.15,?,'旧复核人','旧意见',?)""",
                         (payload, previous_time, previous_time, previous_time))
        database.execute("UPDATE sqlite_sequence SET seq=11 WHERE name='cases'")
    for _ in range(2):
        with TestClient(create_app(tmp_path)) as migrated:
            login(migrated)
            case = migrated.get("/api/cases/EV-000007").json()
            assert case["images"][0]["filename"] == "old.png" and migrated.get(case["images"][0]["url"]).content == payload
            assert case["created_at"] == case["reviewed_at"] == case["archived_at"] == previous_time
            assert (case["status"], case["reviewer"], case["review_note"], case["model"], case["detections"]) == (
                "reviewed", "旧复核人", "旧意见", "original-model", [{"score": 0.45}])
            assert case["report"] == {"findings": "", "conclusion": "旧意见", "author": "旧复核人",
                                      "updated_at": previous_time, "revision": 0}
            assert case["screening"] is None
            assert case["images"][0]["quality"] == "未确认" and case["images"][0]["measurements"] == []
    with TestClient(create_app(tmp_path)) as migrated:
        login(migrated)
        assert migrated.post("/api/visits", json={"horse_name": "新病例"}).json()["id"] == "EV-000012"


def test_image_measurements_use_each_image_coordinates_and_server_computed_values(team, tmp_path):
    sign_in_member(team, "wangyiyi")
    path = "/api/cases/" + upload(team).json()["id"]
    images = append_image(team, path, picture(100, 80)).json()["images"]
    primary, secondary = [image["url"] for image in images]
    other = upload(team).json()["images"][0]["url"]
    assert images[0]["quality"] == "未确认" and images[0]["metadata_reviewer"] is None
    sign_in_member(team, "zhushijia")
    meta = team.patch(secondary, json={"quality": "一般", "projection": "背内侧-掌外侧斜位"}).json()["images"]
    assert meta[0] == images[0]
    assert (meta[1]["quality"], meta[1]["projection"], meta[1]["metadata_reviewer"]) == ("一般", "背内侧-掌外侧斜位", "朱诗嘉")
    assert meta[1]["metadata_reviewed_at"] and team.patch(secondary, json={}).status_code == 422
    distance = {"kind": "distance", "points": [{"x": 60, "y": 40}, {"x": 63, "y": 44}]}
    angle = {"kind": "angle", "points": [{"x": 6, "y": 7}, {"x": 6, "y": 4}, {"x": 2, "y": 4}]}
    for body in [distance, {"kind": "distance", "points": angle["points"][:1]}, {"kind": "angle", "points": angle["points"][:2]},
                 {"kind": "angle", "points": [*angle["points"][:2], angle["points"][1]]},
                 {"kind": "distance", "points": [angle["points"][0], {"x": 32, "y": 3}]},
                 {"kind": "distance", "points": [angle["points"][0], {"x": 3, "y": 48}]},
                 {"kind": "distance", "points": [angle["points"][0], {"x": -1, "y": 2}]}]:
        assert team.post(primary + "/measurements", json=body).status_code == 422
    for invalid_id in ("0", "01", "9", "wrong"):
        assert team.post(path + f"/images/{invalid_id}/measurements", json=angle).status_code == 404
    with ThreadPoolExecutor(2) as pool:
        added = list(pool.map(lambda m: team.post(secondary + "/measurements", json=m), (distance, angle)))
    assert [r.status_code for r in added] == [201, 201]
    with TestClient(create_app(tmp_path)) as restarted:
        sign_in_member(restarted, "liqihang")
        stored = restarted.get(path).json()["images"]
        assert stored[0]["measurements"] == [] and stored[1]["quality"] == "一般"
        marks = {m["kind"]: m for m in stored[1]["measurements"]}
        for kind, expected, value, unit in (("distance", distance, 5, "px"), ("angle", angle, 90, "°")):
            mark = marks[kind]
            assert mark["points"] == expected["points"] and mark["value"] == pytest.approx(value) and mark["unit"] == unit
            assert mark["created_by"] == "朱诗嘉" and mark["created_at"]
        suffix = "/measurements/" + marks["distance"]["id"]
        assert restarted.delete(primary + suffix).status_code == restarted.delete(other + suffix).status_code == 404
        with ThreadPoolExecutor(2) as pool:
            deleting = pool.submit(restarted.delete, secondary + suffix)
            adding = pool.submit(restarted.post, secondary + "/measurements", json=angle)
            assert deleting.result().status_code == 200 and adding.result().status_code == 201
        remaining = restarted.get(path).json()["images"][1]["measurements"]
        assert len(remaining) == 2 and marks["angle"] in remaining
        assert all(mark["kind"] == "angle" for mark in remaining) and restarted.delete(secondary + suffix).status_code == 404
        assert restarted.delete(path).status_code == 204
    with sqlite3.connect(tmp_path / "cases.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM image_measurements").fetchone() == (0,)


@pytest.mark.parametrize("change", [
    "horse_name", "comment", "attachment", "projection", "quality", "add_measurement", "remove_measurement",
])
def test_signing_checks_evidence_changed_by_another_member(team, monkeypatch, change):
    sign_in_member(team, "wangyiyi")
    case = upload(team).json()
    path = "/api/cases/" + case["id"]
    primary = case["images"][0]["url"]
    measure_url = primary + "/measurements"
    distance = {"kind": "distance", "points": [{"x": 1, "y": 2}, {"x": 4, "y": 6}]}
    monkeypatch.setattr(detector, "detect", lambda _: [])
    monkeypatch.setattr(screening, "classify", lambda _: {
        "model": "DINOv2-test", "score": 0.1, "threshold": 0.18, "predicted_positive": False,
    })
    assert team.post(path + "/analyze").status_code == 200
    if change == "remove_measurement":
        added = team.post(measure_url, json=distance)
        assert added.status_code == 201
        mark = added.json()["images"][0]["measurements"][0]["id"]

    detail = path + "?include_signed_evidence=true"
    original = team.get(detail).json()["review_context"]
    sign_in_member(team, "zhushijia")
    if change == "attachment":
        edit = append_image(team, path, picture(36, 40))
    elif change == "add_measurement":
        edit = team.post(measure_url, json=distance)
    elif change == "remove_measurement":
        edit = team.delete(f"{measure_url}/{mark}")
    elif change in {"horse_name", "comment"}:
        edit = team.patch(path, json={change: "补录资料"})
    else:
        edit = team.patch(primary, json={change: "正位" if change == "projection" else "一般"})
    assert edit.is_success, edit.text

    sign_in_member(team, "wangyiyi")
    draft = {"findings": "已核对主片", "conclusion": "", "revision": 0}
    stale = team.post(path + "/review", json=draft | {"review_context": original})
    assert stale.status_code == 409
    assert team.get(path).json()["status"] == "analyzed"
    saved = team.put(path + "/report", json=draft)
    assert saved.status_code == 200
    refreshed = team.get(detail).json()
    assert refreshed["review_context"] != original
    updated = draft | {"revision": saved.json()["report"]["revision"], "review_context": refreshed["review_context"]}
    signed = team.post(path + "/review", json=updated)
    assert signed.status_code == 200, signed.text
    assert signed.json()["status"] == "reviewed" and signed.json()["reviewer"] == "王一伊"


def test_shared_structured_report_conflicts_signatures_and_read_only_history(team, tmp_path, monkeypatch):
    sign_in_member(team, "wangyiyi")
    visit = team.post("/api/visits", json={"horse_name": "追风"}).json()
    path = "/api/cases/" + visit["id"]
    draft = {"findings": "左前球节侧位已核对", "conclusion": "", "revision": 0}
    assert team.put(path + "/report", json=draft).status_code == 409
    primary = append_image(team, path, picture()).json()["images"][0]["url"]
    assert team.put(path + "/report", json=draft, headers={"Origin": "https://other.example"}).status_code == 403
    assert team.put(path + "/report", json=draft | {"findings": " "}).status_code == 422
    competing = [draft | {"findings": finding} for finding in ("已核对正位", "已核对侧位")]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda body: team.put(path + "/report", json=body), competing))
    assert sorted(r.status_code for r in results) == [200, 409]
    first = team.get(path).json()["report"]
    assert first["revision"] == 1 and first["author"] == "王一伊" and first["updated_at"]
    report = {"findings": first["findings"], "conclusion": "结合随访复查", "revision": 1}
    saved = team.put(path + "/report", json=report).json()["report"]
    assert saved["revision"] == 2
    assert team.post(path + "/review", json=report | {"revision": 2}).status_code == 409
    with TestClient(create_app(tmp_path)) as restarted:
        sign_in_member(restarted, "alimire")
        assert restarted.get(path).json()["report"] == saved
        monkeypatch.setattr(detector, "detect", lambda _: [])
        assert restarted.post(path + "/analyze").status_code == 200
        assert restarted.post(path + "/review", json=report).status_code == 409
        assert restarted.post(path + "/review", json={"note": "旧客户端"}).status_code == 409
        assert restarted.post(path + "/review", json={"findings": "", "conclusion": " ", "revision": 2}).status_code == 422
        mark = {"kind": "distance", "points": [{"x": 3, "y": 4}, {"x": 0, "y": 0}]}
        mark_id = restarted.post(primary + "/measurements", json=mark).json()["images"][0]["measurements"][0]["id"]
        confirmed = restarted.post(path + "/review", json=report | {"revision": 2}).json()
        assert confirmed["status"] == "reviewed" and confirmed["report"] == saved and confirmed["archived_at"] is None
        assert confirmed["reviewer"] == "阿里米热" and confirmed["reviewed_at"] and confirmed["review_note"] == report["conclusion"]
        assert restarted.post(path + "/archive").json()["archived_at"]
        assert restarted.put(path + "/report", json=report | {"revision": 2}).status_code == 409
        assert restarted.patch(primary, json={"quality": "一般"}).status_code == 409
        assert restarted.post(primary + "/measurements", json=mark).status_code == 409
        assert restarted.delete(primary + "/measurements/" + mark_id).status_code == 409
        assert restarted.get(path).json()["report"] == saved


@pytest.mark.parametrize(("findings", "conclusion"), [("已观察正位片", ""), ("", "建议核对侧位")])
def test_one_review_field_signs_directly(team, monkeypatch, findings, conclusion):
    sign_in_member(team, "wangyiyi")
    path = "/api/cases/" + upload(team).json()["id"]
    monkeypatch.setattr(detector, "detect", lambda _: [])
    assert team.post(path + "/analyze").status_code == 200
    signed = team.post(path + "/review", json={"findings": findings, "conclusion": conclusion, "revision": 0}).json()
    assert signed["report"]["findings"] == findings and signed["report"]["conclusion"] == conclusion
    assert signed["status"] == "reviewed" and signed["reviewer"] == "王一伊"
    assert signed["empty_review_confirmed"] is False and signed["archived_at"] is None


def test_blank_review_requires_acknowledgement_of_current_draft_and_persists(team, tmp_path, monkeypatch):
    sign_in_member(team, "wangyiyi")
    path = "/api/cases/" + upload(team).json()["id"]
    monkeypatch.setattr(detector, "detect", lambda _: [])
    team.post(path + "/analyze")
    blank = {"findings": "\n ", "conclusion": "　", "revision": 0}
    for flag in ({}, {"empty_review_confirmed": False}, {"empty_review_confirmed": "true"}):
        assert team.post(path + "/review", json=blank | flag).status_code == 422
    unchanged = team.get(path).json()
    assert unchanged["status"] == "analyzed" and unchanged["report"]["revision"] == 0
    assert unchanged["empty_review_confirmed"] is False and unchanged["reviewed_at"] is None

    team.put(path + "/report", json=blank | {"findings": "同事已填写草稿"})
    acknowledged = blank | {"empty_review_confirmed": True}
    assert team.post(path + "/review", json=acknowledged).status_code == 409
    assert team.post(path + "/review", json={"note": "", "empty_review_confirmed": True}).status_code == 409
    assert team.get(path).json()["report"]["findings"] == "同事已填写草稿"
    sign_in_member(team, "zhushijia")
    signed = team.post(path + "/review", json=acknowledged | {"revision": 1}).json()
    assert signed["status"] == "reviewed" and signed["empty_review_confirmed"] is True
    assert signed["report"]["findings"] == signed["report"]["conclusion"] == signed["review_note"] == ""
    assert signed["report"]["revision"] == 2 and signed["reviewer"] == "朱诗嘉" and signed["reviewed_at"]
    assert signed["archived_at"] is None and team.post(path + "/review", json=acknowledged).status_code == 409
    with TestClient(create_app(tmp_path)) as restarted:
        sign_in_member(restarted, "alimire")
        assert restarted.get(path).json() == signed


def test_viewer_writes_stay_within_shared_workspace(team, tmp_path, monkeypatch):
    sign_in_member(team, "wangyiyi")
    case = upload(team).json()
    path = "/api/cases/" + case["id"]
    image = case["images"][0]["url"]
    monkeypatch.setenv("SPORT_MODEL_USERNAME", "external")
    monkeypatch.setenv("SPORT_MODEL_PASSWORD", "test-password")
    with TestClient(create_app(tmp_path)) as stranger:
        login(stranger, "external")
        assert stranger.patch(image, json={"quality": "良好"}).status_code == 404
        assert stranger.post(image + "/measurements", json={"kind": "distance",
                            "points": [{"x": 3, "y": 4}, {"x": 0, "y": 0}]}).status_code == 404
        assert stranger.delete(image + "/measurements/1").status_code == 404
        assert stranger.put(path + "/report", json={"findings": "无权限", "conclusion": "", "revision": 0}).status_code == 404
    assert team.get(path).json() == case


def test_existing_multi_image_database_upgrades_without_guessing_quality_or_changing_blobs(client, tmp_path):
    login(client)
    first, second = picture(), picture(61, 80)
    case = upload(client, first).json()
    path = "/api/cases/" + case["id"]
    images = append_image(client, path, second).json()["images"]
    with sqlite3.connect(tmp_path / "cases.sqlite3") as database:
        database.execute("ALTER TABLE cases DROP COLUMN empty_review_confirmed")
        database.execute("ALTER TABLE cases DROP COLUMN screening")
        database.execute("DROP TABLE image_annotations")
        database.execute("ALTER TABLE cases DROP COLUMN annotation_revision")
        for column in ("metadata_reviewed_at", "metadata_reviewer", "quality"):
            database.execute(f"ALTER TABLE case_images DROP COLUMN {column}")
        database.execute("DROP TABLE image_measurements")
        database.execute("DROP TABLE case_reports")
    for _ in range(2):
        with TestClient(create_app(tmp_path)) as upgraded:
            login(upgraded)
            case = upgraded.get(path).json()
            assert case["empty_review_confirmed"] is False
            assert case["screening"] is None
            assert case["annotation_revision"] == 0
            assert case["images"] == images and case["report"]["revision"] == 0
            assert [upgraded.get(image["url"]).content for image in images] == [first, second]


def rectangle(label="关节边缘"):
    return {"revision": 0, "kind": "rectangle", "label": label,
            "points": [{"x": 20.5, "y": 24}, {"x": 1, "y": 3}]}


def test_annotations_keep_image_ownership_snapshots_and_signature(team, tmp_path, monkeypatch):
    sign_in_member(team, "wangyiyi")
    case = upload(team).json()
    path = "/api/cases/" + case["id"]
    second = append_image(team, path, picture(61, 80)).json()["images"][1]
    url = second["url"] + "/annotations/" + str(uuid4())
    body = rectangle()
    created = team.put(url, json=body).json()
    assert created["annotation_revision"] == 1 and created["detections"] == []
    assert created["images"][0]["annotations"] == []
    mark, = created["images"][1]["annotations"]
    assert (mark["created_by"], mark["source"], mark["revision"]) == ("王一伊", "manual", 1)
    assert mark["points"] == [{"x": 1, "y": 3}, {"x": 20.5, "y": 24}]
    assert team.put(url, json=body).json() == created
    with TestClient(create_app(tmp_path)) as other:
        assert other.put(url, json=body).status_code == 401
        sign_in_member(other, "zhushijia")
        update = body | {"label": "待复核区域", "revision": 1}
        assert other.put(url, json=update).json()["images"][1]["annotations"][0]["updated_by"] == "朱诗嘉"
    stale = team.put(url, json=body | {"label": "旧值", "revision": 1})
    assert stale.status_code == 409
    monkeypatch.setattr(screening, "classify", lambda _: {"model": "test", "score": 0.1, "threshold": 0.2,
                                                         "predicted_positive": False})
    monkeypatch.setattr(detector, "detect", lambda _: [])
    assert team.post(path + "/analyze").status_code == 200
    assert team.post(path + "/review", json={"note": "签署", "annotation_revision": 1}).status_code == 409
    assert team.post(path + "/review", json={"note": "已核对标注", "annotation_revision": 2}).status_code == 200
    assert team.put(url, json=body | {"revision": 2, "label": "签署后编辑"}).status_code == 409
    assert team.delete(url + "?revision=2").status_code == 409
    archived = team.post(path + "/archive").json()
    assert archived["archived_at"] and archived["images"][1]["annotations"][0]["label"] == "待复核区域"
    with TestClient(create_app(tmp_path)) as reopened:
        sign_in_member(reopened, "zhushijia")
        assert reopened.get(path).json() == archived
    monkeypatch.setenv("SPORT_MODEL_USERNAME", "outsider")
    monkeypatch.setenv("SPORT_MODEL_PASSWORD", "test-password")
    with TestClient(create_app(tmp_path)) as outsider:
        login(outsider, "outsider")
        assert outsider.put(url, json=body).status_code == 404
        assert outsider.delete(url + "?revision=2").status_code == 404


def test_annotation_conflicting_writes_and_delete_retry(client):
    login(client)
    case = upload(client).json()
    path = "/api/cases/" + case["id"]
    url = case["images"][0]["url"] + "/annotations/" + str(uuid4())
    body = rectangle()
    assert client.put(url, json=body).status_code == 200
    with ThreadPoolExecutor(2) as pool:
        writes = list(pool.map(lambda label: client.put(url, json=body | {"label": label, "revision": 1}),
                               ["第一项", "第二项"]))
    assert sorted(r.status_code for r in writes) == [200, 409]
    assert client.delete(url + "?revision=1").status_code == 409
    removed = client.delete(url + "?revision=2").json()
    assert removed["annotation_revision"] == 3 and not removed["images"][0]["annotations"]
    assert client.delete(url + "?revision=2").json() == removed
    assert client.put(url, json=body).status_code == 409  # A delayed create cannot resurrect a deleted mark.
    assert client.get(path).json() == removed


def test_annotation_geometry_is_pixel_space_and_bounded(client):
    login(client)
    image = upload(client).json()["images"][0]
    url = image["url"] + "/annotations/" + str(uuid4())
    def polygon(coords):
        return {"revision": 0, "kind": "polygon", "label": "人工区域",
                "points": [{"x": x, "y": y} for x, y in coords]}
    for invalid in ([(1,1),(2,2)], [(1,1),(1,1),(5,6)], [(1,1),(2,2),(3,3)],
                    [(0,0),(16,0),(10,12)], [(0,0),(18.5,0),(5.5,7),(20,20),(0,18)],
                    [(1,1),(2,2),(3,3),(4,4)], [(1,1),(1,1),(3,0),(1,3)], [(0,0),(32,0),(7,8)]):
        assert client.put(url, json=polygon(invalid)).status_code == 422
    for invalid in (rectangle("  "), rectangle() | {"points": [{"x": 1,"y": 0},{"x": 1,"y": 5}]},
                    rectangle() | {"source": "ai"}):
        assert client.put(url, json=invalid).status_code == 422
    valid = polygon([(0,0),(18.5,0),(20,20),(0,18)])
    saved = client.put(url, json=valid).json()["images"][0]["annotations"][0]
    assert saved["points"] == valid["points"] and saved["kind"] == "polygon"


@pytest.mark.parametrize("vertices,expected_area", [
    ([(1,1),(21,1),(21,15),(1,15)], 280),
    ([(0,0),(20,0),(0,20),(2,8)], 180),  # The fourth vertex is inside the triangle.
    ([(0,0),(8,0),(20,0),(0,20)], 200),
])
def test_four_point_outline_is_largest_for_every_click_order(client, vertices, expected_area):
    login(client)
    image = upload(client).json()["images"][0]
    url = image["url"] + "/annotations/" + str(uuid4())
    first = None
    for order in permutations(vertices):
        response = client.put(url, json={"revision": 0, "kind": "polygon", "label": "人工区域",
                                         "points": [{"x": x, "y": y} for x, y in order]})
        assert response.status_code == 200
        annotation, = response.json()["images"][0]["annotations"]
        outline = [(p["x"], p["y"]) for p in annotation["points"]]
        if first is None:
            first = outline
        assert outline == first and set(outline) == set(vertices) and annotation["revision"] == 1
        twice_area = sum(x*y2-x2*y for (x,y),(x2,y2) in zip(outline, outline[1:] + outline[:1]))
        assert twice_area == 2*expected_area


@pytest.mark.parametrize("vertices", [
    [(0,0),(20,0),(0,20)],
    [(0,0),(2,8),(20,0),(0,20)],  # Four points saved in a different simple order.
    [(0,0),(18,0),(5,7),(20,20),(0,18)],
])
def test_renaming_legacy_polygon_preserves_its_boundary(client, tmp_path, vertices):
    login(client)
    case = upload(client).json()
    mark_id = str(uuid4())
    url = case["images"][0]["url"] + "/annotations/" + mark_id
    body = {"revision": 0, "kind": "polygon", "label": "原名", "points":
            [{"x": x, "y": y} for x, y in [(0,0),(20,0),(20,20),(0,20)]]}
    assert client.put(url, json=body).is_success
    old_outline = [{"x": x, "y": y} for x, y in vertices]
    with sqlite3.connect(tmp_path / "cases.sqlite3") as database:
        database.execute("UPDATE image_annotations SET points = ? WHERE id = ?", (json.dumps(old_outline), mark_id))
    rename = {**body, "points": old_outline, "revision": 1, "label": "更正名称"}
    for _ in range(2):
        renamed = client.put(url, json=rename)
        assert renamed.is_success
        mark, = renamed.json()["images"][0]["annotations"]
        assert mark["points"] == old_outline and mark["revision"] == 2
    if len(vertices) != 4:
        moved = [{"x": 1, "y": 1}, *old_outline[1:]]
        assert client.put(url, json={**rename, "revision": 2, "points": moved}).status_code == 422
    response = client.get(case["images"][0]["url"] + "/annotations.png")
    with Image.open(io.BytesIO(response.content)) as png:
        legend, = json.loads(png.info["EQUIVISION.manual_annotations"])
    assert legend["points"] == old_outline and legend["label"] == "更正名称"
