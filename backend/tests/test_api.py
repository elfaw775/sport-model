import io
import sqlite3

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

    monkeypatch.setattr(detector, "detect", unavailable)
    assert client.post(resource + "/analyze").status_code == 503
    assert client.get(resource).json()["status"] == "uploaded"
    predicted = [{"x1": 3.0, "y1": 4.0, "x2": 8.0, "y2": 12.0, "score": 0.82}]
    monkeypatch.setattr(detector, "detect", lambda image: predicted if image.size == (32, 48) else [])
    analyzed = client.post(resource + "/analyze").json()
    assert analyzed["detections"] == predicted and analyzed["model"] == detector.MODEL_ID
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
    with TestClient(create_app(tmp_path)) as migrated:
        login(migrated)
        assert migrated.post("/api/visits", json={"horse_name": "新病例"}).json()["id"] == "EV-000012"
