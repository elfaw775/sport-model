import hashlib
import io
import json
import logging
import math
import os
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from . import detector, screening


ROOT = Path(__file__).resolve().parents[1]
COOKIE = "equivision_session"
SESSION_TTL = 8 * 3600
MAX_BYTES = 10 * 1024 * 1024
MIMES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
TEAM = {"wangyiyi": "王一伊", "zhushijia": "朱诗嘉", "alimire": "阿里米热", "liqihang": "李其杭"}
CASE_FIELDS = """number, owner AS created_by, horse_name, comment, created_at, archived_at, status,
    detections, model, threshold, screening, analyzed_at, reviewer, review_note, reviewed_at, empty_review_confirmed"""
CASE_SCHEMA = """CREATE TABLE IF NOT EXISTS cases (
    number INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL,
    horse_name TEXT NOT NULL, comment TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
    archived_at TEXT, status TEXT NOT NULL DEFAULT 'registered', detections TEXT NOT NULL DEFAULT '[]',
    model TEXT, threshold REAL, screening TEXT, analyzed_at TEXT, reviewer TEXT, review_note TEXT, reviewed_at TEXT,
    empty_review_confirmed INTEGER NOT NULL DEFAULT 0
)"""
IMAGES_SCHEMA = """CREATE TABLE IF NOT EXISTS case_images (
    case_number INTEGER NOT NULL REFERENCES cases(number) ON DELETE CASCADE,
    position INTEGER NOT NULL, filename TEXT NOT NULL, projection TEXT NOT NULL,
    quality_confirmed INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
    quality TEXT NOT NULL DEFAULT '未确认', metadata_reviewer TEXT, metadata_reviewed_at TEXT,
    mime TEXT NOT NULL, original BLOB NOT NULL, PRIMARY KEY(case_number, position)
)"""
Projection = Literal["正位", "侧位", "斜位", "背内侧-掌外侧斜位", "背外侧-掌内侧斜位", "其他"]


def prepare_database(database, team_mode):
    database.execute("BEGIN IMMEDIATE")
    database.execute("""CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY, username TEXT NOT NULL, expires INTEGER NOT NULL)""")
    legacy = "original" in {row["name"] for row in database.execute("PRAGMA table_info(cases)")}
    if legacy:
        high_water = database.execute("SELECT seq FROM sqlite_sequence WHERE name='cases'").fetchone()
        database.execute("ALTER TABLE cases RENAME TO cases_v1")
    database.execute(CASE_SCHEMA)
    case_columns = {row["name"] for row in database.execute("PRAGMA table_info(cases)")}
    if "empty_review_confirmed" not in case_columns:
        database.execute("ALTER TABLE cases ADD COLUMN empty_review_confirmed INTEGER NOT NULL DEFAULT 0")
    if "screening" not in case_columns:
        database.execute("ALTER TABLE cases ADD COLUMN screening TEXT")
    database.execute(IMAGES_SCHEMA)
    if legacy:
        database.execute("""INSERT INTO cases(number, owner, horse_name, created_at, status, detections,
            model, threshold, analyzed_at, reviewer, review_note, reviewed_at, archived_at)
            SELECT number, owner, horse_name, created_at, status, detections, model, threshold,
            analyzed_at, reviewer, review_note, reviewed_at, CASE WHEN status='reviewed' THEN reviewed_at END
            FROM cases_v1""")
        database.execute("""INSERT INTO case_images(case_number, position, filename, projection,
            quality_confirmed, width, height, mime, original) SELECT number, 0, filename, projection,
            quality_confirmed, width, height, mime, original FROM cases_v1""")
        database.execute("DROP TABLE cases_v1")
        if high_water:
            database.execute("UPDATE sqlite_sequence SET seq = max(seq, ?) WHERE name='cases'", (high_water["seq"],))
    database.execute("CREATE INDEX IF NOT EXISTS cases_owner ON cases(owner, number)")
    image_columns = {row["name"] for row in database.execute("PRAGMA table_info(case_images)")}
    if "quality" not in image_columns:
        database.execute("ALTER TABLE case_images ADD COLUMN quality TEXT NOT NULL DEFAULT '未确认'")
        database.execute("ALTER TABLE case_images ADD COLUMN metadata_reviewer TEXT")
        database.execute("ALTER TABLE case_images ADD COLUMN metadata_reviewed_at TEXT")
    database.execute("""CREATE TABLE IF NOT EXISTS image_measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT, case_number INTEGER NOT NULL, position INTEGER NOT NULL,
        kind TEXT NOT NULL, points TEXT NOT NULL, value REAL NOT NULL, created_by TEXT NOT NULL, created_at TEXT NOT NULL,
        FOREIGN KEY (case_number,position) REFERENCES case_images(case_number,position) ON DELETE CASCADE)""")
    database.execute("CREATE INDEX IF NOT EXISTS measurements_image ON image_measurements(case_number,position)")
    database.execute("""CREATE TABLE IF NOT EXISTS case_reports (
        case_number INTEGER PRIMARY KEY REFERENCES cases(number) ON DELETE CASCADE, findings TEXT NOT NULL,
        conclusion TEXT NOT NULL, author TEXT NOT NULL, updated_at TEXT NOT NULL, revision INTEGER NOT NULL)""")
    if team_mode and database.execute("PRAGMA user_version").fetchone()[0] == 0:
        database.execute("UPDATE cases SET owner='wangyiyi' WHERE owner='demo'")
        for username, name in TEAM.items():
            database.execute("UPDATE cases SET owner=? WHERE owner=?", (username, name))
        database.execute("DELETE FROM sessions")
        database.execute("PRAGMA user_version=1")


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str = Field(max_length=2000)
    empty_review_confirmed: bool = Field(default=False, strict=True)


class ReportDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: str = Field(max_length=4000)
    conclusion: str = Field(max_length=2000)
    revision: int = Field(ge=0, strict=True)


class ReportConfirmation(ReportDraft):
    empty_review_confirmed: bool = Field(default=False, strict=True)


class ImageReview(BaseModel):
    projection: Projection | Literal["未确认"] | None = None
    quality: Literal["未确认", "良好", "一般", "不适合"] | None = None


class ImagePoint(BaseModel):
    x: float = Field(ge=0, allow_inf_nan=False, strict=True)
    y: float = Field(ge=0, allow_inf_nan=False, strict=True)


class Measurement(BaseModel):
    kind: Literal["distance", "angle"]
    points: list[ImagePoint] = Field(min_length=2, max_length=3)

    def value(self, width, height):
        if any(p.x >= width or p.y >= height for p in self.points):
            raise HTTPException(422, "测量点超出原图范围")
        if len(self.points) != (2 if self.kind == "distance" else 3):
            raise HTTPException(422, "距离需要两个点，角度需要三个点")
        if any(a == b for a, b in zip(self.points, self.points[1:])):
            raise HTTPException(422, "相邻测量点不能重合")
        a, b = self.points[:2]
        if self.kind == "distance":
            return math.hypot(a.x - b.x, a.y - b.y)
        c = self.points[2]
        angle = abs(math.degrees(math.atan2(a.y - b.y, a.x - b.x) - math.atan2(c.y - b.y, c.x - b.x)))
        return min(angle, 360 - angle)


class Visit(BaseModel):
    horse_name: str = Field(min_length=1, max_length=80)
    comment: str = Field(default="", max_length=4000)


class VisitUpdate(BaseModel):
    horse_name: str | None = Field(default=None, min_length=1, max_length=80)
    comment: str | None = Field(default=None, max_length=4000)


def timestamp():
    return datetime.now(UTC).isoformat(timespec="seconds")


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def case_number(identifier):
    digits = identifier.removeprefix("EV-")
    if not digits.isascii() or not digits.isdecimal() or not 6 <= len(digits) <= 18:
        raise HTTPException(404, "未找到此病例")
    number = int(digits)
    if identifier != f"EV-{number:06d}":
        raise HTTPException(404, "未找到此病例")
    return number


def image_position(image_id):
    if image_id == "primary":
        return 0
    if image_id.isascii() and image_id.isdecimal() and len(image_id) <= 18 and not image_id.startswith("0"):
        return int(image_id)
    raise HTTPException(404, "未找到此影像")


def load_cases(database, owners, number=None):
    if not database.in_transaction:
        database.execute("BEGIN")
    condition = "owner IN (" + ",".join("?" for _ in owners) + ")"
    params = tuple(owners)
    if number is not None:
        condition += " AND number = ?"
        params += (number,)
    records = {}
    for row in database.execute(f"SELECT {CASE_FIELDS} FROM cases WHERE {condition} ORDER BY number DESC", params):
        case = dict(row)
        key = case.pop("number")
        case.update(id=f"EV-{key:06d}", detections=json.loads(case["detections"]), images=[],
                    screening=json.loads(case["screening"]) if case["screening"] is not None else None,
                    empty_review_confirmed=bool(case["empty_review_confirmed"]),
                    has_image=False, filename=None, projection=None, width=None, height=None, quality_confirmed=False)
        case["report"] = {"findings": "", "conclusion": case["review_note"] or "", "author": case["reviewer"],
                          "updated_at": case["reviewed_at"], "revision": 0}
        records[key] = case
    for row in database.execute(f"""SELECT case_number, findings, conclusion, author, updated_at, revision
        FROM case_reports JOIN cases ON number=case_number WHERE {condition}""", params):
        report = dict(row)
        records[report.pop("case_number")]["report"] = report
    images = {}
    for row in database.execute(f"""SELECT case_number, position, filename, projection, width, height, quality_confirmed,
        quality, metadata_reviewer, metadata_reviewed_at
        FROM case_images JOIN cases ON number=case_number WHERE {condition} ORDER BY case_number, position""", params):
        image = dict(row)
        key = image.pop("case_number")
        case = records[key]
        position = image.pop("position")
        image["quality_confirmed"] = bool(image["quality_confirmed"])
        if position == 0:
            case.update(image, has_image=True)
        image["id"] = str(position) if position else "primary"
        image["url"] = f"/api/cases/{case['id']}/images/{image['id']}"
        image["measurements"] = []
        images[key, position] = image
        case["images"].append(image)
    for row in database.execute(f"""SELECT case_number, position, id, kind, points, value, created_by,
        image_measurements.created_at FROM image_measurements JOIN cases ON number=case_number
        WHERE {condition} ORDER BY id""", params):
        mark = dict(row)
        image = images[mark.pop("case_number"), mark.pop("position")]
        mark.update(id=str(mark["id"]), points=json.loads(mark["points"]), unit="px" if mark["kind"] == "distance" else "°")
        image["measurements"].append(mark)
    return list(records.values())


def read_case(database, number, owner):
    records = load_cases(database, (owner,), number)
    if not records:
        raise HTTPException(404, "未找到此病例")
    return records[0]


def original(database, number, owner, position=0):
    row = database.execute(
        """SELECT original, mime, status FROM case_images JOIN cases ON number=case_number
           WHERE number=? AND owner=? AND position=?""", (number, owner, position)
    ).fetchone()
    if row is None:
        raise HTTPException(404, "未找到此影像")
    return row


def editable_image(database, number, owner, position):
    database.execute("BEGIN IMMEDIATE")
    row = database.execute("""SELECT width, height, status FROM case_images JOIN cases ON number=case_number
        WHERE number=? AND owner=? AND position=?""", (number, owner, position)).fetchone()
    if row is None:
        raise HTTPException(404, "未找到此影像")
    if row["status"] == "reviewed":
        raise HTTPException(409, "病例已复核，阅片记录只读")
    return row


def store_report(database, number, case, draft, author, *, confirmed_empty=False):
    findings, conclusion = draft.findings.strip(), draft.conclusion.strip()
    if not case["has_image"] or case["status"] == "reviewed":
        raise HTTPException(409, "请在上传影像后、确认复核前编辑报告")
    report = case["report"]
    if draft.revision != report["revision"]:
        raise HTTPException(409, "报告已更新，请重新读取病例后核对")
    if not (findings or conclusion or confirmed_empty):
        raise HTTPException(422, "请填写影像所见或人工结论")
    if confirmed_empty or (findings, conclusion) != (report["findings"], report["conclusion"]):
        database.execute("""INSERT INTO case_reports(case_number,findings,conclusion,author,updated_at,revision)
            VALUES (?,?,?,?,?,?) ON CONFLICT(case_number) DO UPDATE SET findings=excluded.findings,
            conclusion=excluded.conclusion, author=excluded.author, updated_at=excluded.updated_at,
            revision=excluded.revision""", (number, findings, conclusion, author, timestamp(), draft.revision + 1))


def inspect_image(payload):
    try:
        with Image.open(io.BytesIO(payload), formats=tuple(MIMES)) as image:
            if max(image.size) > 8000 or image.width * image.height > 24_000_000:
                raise HTTPException(413, "影像超过尺寸限制（最长边 8000 像素，总计 2400 万像素）")
            if getattr(image, "n_frames", 1) != 1:
                raise HTTPException(422, "请上传单帧影像")
            mime = MIMES[image.format]
            image.load()
            width, height = ImageOps.exif_transpose(image).size
            return mime, width, height
    except Image.DecompressionBombError as error:
        raise HTTPException(413, "影像超过尺寸限制") from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise HTTPException(422, "无法读取影像，请上传完整的 PNG、JPEG 或 WebP 文件") from error


def attachment(image, projection, quality_confirmed):
    if not quality_confirmed:
        raise HTTPException(422, "请先确认影像质量")
    payload = image.file.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise HTTPException(413, "文件不能超过 10 MiB")
    mime, width, height = inspect_image(payload)
    filename = Path((image.filename or "影像").replace("\\", "/")).name[:255] or "影像"
    return filename, projection, 1, width, height, mime, payload


def save_attachment(database, number, owner, values):
    database.execute("""INSERT INTO case_images(case_number,position,filename,projection,quality_confirmed,width,height,mime,original)
        SELECT number, COALESCE((SELECT MAX(position)+1 FROM case_images WHERE case_number=number), 0),
        ?, ?, ?, ?, ?, ?, ? FROM cases WHERE number=? AND owner=?""", (*values, number, owner))
    database.execute("""UPDATE cases SET status='uploaded', archived_at=NULL
        WHERE number=? AND owner=? AND status='registered'""", (number, owner))


def create_case(database, owner, horse_name, comment):
    if not horse_name.strip():
        raise HTTPException(422, "请填写马匹名称或编号")
    return database.execute("INSERT INTO cases(owner, horse_name, comment, created_at) VALUES (?, ?, ?, ?)",
                            (owner, horse_name.strip(), comment.strip(), timestamp())).lastrowid


def create_app(data_dir=None):
    directory = Path(data_dir or os.environ.get("SPORT_MODEL_DATA_DIR", ROOT / "var"))
    team_mode = not any(key in os.environ for key in
                        ("SPORT_MODEL_USERNAME", "SPORT_MODEL_PASSWORD", "SPORT_MODEL_DISPLAY_NAME"))
    if team_mode:
        accounts = {username: (username + "sportsHorse", name) for username, name in TEAM.items()}
    else:
        username = os.environ.get("SPORT_MODEL_USERNAME", "demo")
        password = os.environ.get("SPORT_MODEL_PASSWORD", "")
        display_name = os.environ.get("SPORT_MODEL_DISPLAY_NAME", "王一伊")
        if username in TEAM or username in TEAM.values() or not all((username, password, display_name)):
            raise ValueError("自定义账号名不能使用团队保留账号，且账号、密码和显示名不能为空")
        accounts = {username: (password, display_name)}
    secure_cookie = os.environ.get("SPORT_MODEL_SECURE_COOKIE") == "1"

    @contextmanager
    def connect():
        connection = sqlite3.connect(directory / "cases.sqlite3", timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @asynccontextmanager
    async def lifespan(_app):
        directory.mkdir(parents=True, exist_ok=True)
        with connect() as database:
            database.execute("PRAGMA journal_mode = WAL")
            prepare_database(database, team_mode)
        yield

    api = FastAPI(title="EQUIVISION 运动马影像工作台", lifespan=lifespan)

    @api.middleware("http")
    async def protect_api(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if request.method in {"POST", "PUT", "PATCH", "DELETE"} and origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "仅接受同源操作"}, status_code=403)
            size = request.headers.get("content-length", "")
            if size.isdigit() and int(size) > MAX_BYTES + 65536:
                return JSONResponse({"detail": "文件不能超过 10 MiB"}, status_code=413)
        result = await call_next(request)
        result.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            result.headers["Cache-Control"] = "no-store"
        return result

    def current_user(request: Request):
        token = request.cookies.get(COOKIE)
        if token:
            with connect() as database:
                session = database.execute(
                    "SELECT username FROM sessions WHERE token_hash = ? AND expires > ?",
                    (token_hash(token), int(time.time())),
                ).fetchone()
            if session and session["username"] in accounts:
                return session["username"]
        raise HTTPException(401, "请先登录工作台")

    def record_owner(identifier: str, _user: str = Depends(current_user)):
        with connect() as database:
            case = database.execute("SELECT owner FROM cases WHERE number=?", (case_number(identifier),)).fetchone()
        if case is None or case["owner"] not in accounts:
            raise HTTPException(404, "未找到此病例")
        return case["owner"]

    @api.get("/api/login-options")
    def login_options():
        return TEAM if team_mode else {}

    @api.post("/api/login")
    def login(credentials: Credentials, request: Request, response: Response):
        username = credentials.username
        account = accounts.get(username)
        if not account or not secrets.compare_digest(account[0].encode(), credentials.password.encode()):
            raise HTTPException(401, "账号或密码不正确")
        now = int(time.time())
        token = secrets.token_urlsafe(32)
        with connect() as database:
            database.execute("DELETE FROM sessions WHERE expires <= ? OR token_hash = ?",
                             (now, token_hash(request.cookies.get(COOKIE, ""))))
            database.execute("INSERT INTO sessions VALUES (?, ?, ?)", (token_hash(token), username, now + SESSION_TTL))
        response.set_cookie(COOKIE, token, httponly=True, secure=secure_cookie, samesite="strict", max_age=SESSION_TTL)
        return {"username": username, "display_name": account[1]}

    @api.get("/api/me")
    def me(owner: str = Depends(current_user)):
        return {"username": owner, "display_name": accounts[owner][1]}

    @api.post("/api/logout", status_code=204)
    def logout(request: Request):
        with connect() as database:
            database.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(request.cookies.get(COOKIE, "")),))
        response = Response(status_code=204)
        response.delete_cookie(COOKIE, httponly=True, secure=secure_cookie, samesite="strict")
        return response

    @api.post("/api/visits", status_code=201)
    def visit(visit: Visit, owner: str = Depends(current_user)):
        with connect() as database:
            number = create_case(database, owner, visit.horse_name, visit.comment)
            return read_case(database, number, owner)

    @api.post("/api/cases", status_code=201)
    def upload(image: UploadFile = File(), horse_name: str = Form(min_length=1, max_length=80),
               projection: Projection = Form(), quality_confirmed: bool = Form(),
               owner: str = Depends(current_user)):
        values = attachment(image, projection, quality_confirmed)
        with connect() as database:
            number = create_case(database, owner, horse_name, "")
            save_attachment(database, number, owner, values)
            return read_case(database, number, owner)

    @api.get("/api/cases", dependencies=[Depends(current_user)])
    def cases():
        with connect() as database:
            return load_cases(database, accounts)

    @api.get("/api/cases/{identifier}")
    def case_detail(identifier: str, owner: str = Depends(record_owner)):
        with connect() as database:
            return read_case(database, case_number(identifier), owner)

    @api.patch("/api/cases/{identifier}")
    def edit(identifier: str, update: VisitUpdate, owner: str = Depends(record_owner)):
        if update.horse_name is None and update.comment is None:
            raise HTTPException(422, "请填写需要更新的资料")
        if update.horse_name is not None and not update.horse_name.strip():
            raise HTTPException(422, "请填写马匹名称或编号")
        number = case_number(identifier)
        with connect() as database:
            database.execute("""UPDATE cases SET horse_name=COALESCE(?, horse_name), comment=COALESCE(?, comment)
                WHERE number=? AND owner=?""", (update.horse_name.strip() if update.horse_name is not None else None,
                update.comment.strip() if update.comment is not None else None, number, owner))
            return read_case(database, number, owner)

    @api.delete("/api/cases/{identifier}", status_code=204)
    def delete(identifier: str, owner: str = Depends(record_owner)):
        with connect() as database:
            changed = database.execute("DELETE FROM cases WHERE number=? AND owner=?", (case_number(identifier), owner))
            if not changed.rowcount:
                raise HTTPException(404, "未找到此病例")
        return Response(status_code=204)

    @api.post("/api/cases/{identifier}/archive")
    def archive(identifier: str, owner: str = Depends(record_owner)):
        number = case_number(identifier)
        with connect() as database:
            changed = database.execute("""UPDATE cases SET archived_at=COALESCE(archived_at, ?)
                WHERE number=? AND owner=? AND (archived_at IS NOT NULL OR status='reviewed'
                OR NOT EXISTS (SELECT 1 FROM case_images WHERE case_number=cases.number))""", (timestamp(), number, owner))
            case = read_case(database, number, owner)
            if not changed.rowcount:
                raise HTTPException(409, "有影像的病例须完成分析并确认人工复核后才能归档")
            return case

    @api.post("/api/cases/{identifier}/images", status_code=201)
    def add_image(identifier: str, image: UploadFile = File(), projection: Projection = Form(),
                  quality_confirmed: bool = Form(), owner: str = Depends(record_owner)):
        number = case_number(identifier)
        values = attachment(image, projection, quality_confirmed)
        with connect() as database:
            save_attachment(database, number, owner, values)
            return read_case(database, number, owner)

    @api.get("/api/cases/{identifier}/images/{image_id}")
    def visit_image(identifier: str, image_id: str, owner: str = Depends(record_owner)):
        with connect() as database:
            row = original(database, case_number(identifier), owner, image_position(image_id))
        return Response(row["original"], media_type=row["mime"])

    @api.patch("/api/cases/{identifier}/images/{image_id}")
    def review_image(identifier: str, image_id: str, update: ImageReview, owner: str = Depends(record_owner),
                     user: str = Depends(current_user)):
        if update.quality is None and update.projection is None:
            raise HTTPException(422, "请选择体位或影像质量")
        number, position = case_number(identifier), image_position(image_id)
        with connect() as database:
            editable_image(database, number, owner, position)
            database.execute("""UPDATE case_images SET projection=COALESCE(?, projection), quality=COALESCE(?, quality),
                metadata_reviewer=?, metadata_reviewed_at=? WHERE case_number=? AND position=?""",
                (update.projection, update.quality, accounts[user][1], timestamp(), number, position))
            return read_case(database, number, owner)

    @api.post("/api/cases/{identifier}/images/{image_id}/measurements", status_code=201)
    def add_measurement(identifier: str, image_id: str, measurement: Measurement, owner: str = Depends(record_owner),
                        user: str = Depends(current_user)):
        number, position = case_number(identifier), image_position(image_id)
        with connect() as database:
            image = editable_image(database, number, owner, position)
            value = measurement.value(image["width"], image["height"])
            points = json.dumps([p.model_dump() for p in measurement.points], allow_nan=False)
            database.execute("""INSERT INTO image_measurements(case_number,position,kind,points,value,created_by,created_at)
                VALUES (?,?,?,?,?,?,?)""", (number, position, measurement.kind, points, value, accounts[user][1], timestamp()))
            return read_case(database, number, owner)

    @api.delete("/api/cases/{identifier}/images/{image_id}/measurements/{measurement_id}")
    def remove_measurement(identifier: str, image_id: str, measurement_id: str, owner: str = Depends(record_owner)):
        number, position = case_number(identifier), image_position(image_id)
        with connect() as database:
            editable_image(database, number, owner, position)
            changed = database.execute("DELETE FROM image_measurements WHERE case_number=? AND position=? AND id=?",
                                       (number, position, measurement_id))
            if not changed.rowcount:
                raise HTTPException(404, "未找到此影像的测量记录")
            return read_case(database, number, owner)

    @api.get("/api/cases/{identifier}/image")
    def case_image(identifier: str, owner: str = Depends(record_owner)):
        with connect() as database:
            row = original(database, case_number(identifier), owner)
        return Response(row["original"], media_type=row["mime"])

    @api.post("/api/cases/{identifier}/analyze")
    def analyze(identifier: str, owner: str = Depends(record_owner)):
        number = case_number(identifier)
        with connect() as database:
            if read_case(database, number, owner)["status"] == "registered":
                raise HTTPException(409, "请先上传用于分析的主影像")
            row = original(database, number, owner)
        if row["status"] != "uploaded":
            raise HTTPException(409, "本病例已有分析结果")
        stage = "读取影像"
        try:
            with Image.open(io.BytesIO(row["original"])) as image:
                stage = "DINOv2 整片筛查"
                film_result = screening.classify(image)
                stage = "YOLO11s 病灶定位"
                predictions = detector.detect(ImageOps.exif_transpose(image))
        except Exception as error:
            logging.getLogger(__name__).exception("病例 %s %s失败", identifier, stage)
            raise HTTPException(503, f"{stage}失败，原片已保留，可稍后重试") from error
        with connect() as database:
            changed = database.execute("""
                UPDATE cases SET status='analyzed', detections=?, model=?, threshold=?, screening=?, analyzed_at=?
                WHERE number=? AND owner=? AND status='uploaded'
            """, (json.dumps(predictions, allow_nan=False), detector.MODEL_ID, detector.THRESHOLD,
                  json.dumps(film_result, allow_nan=False), timestamp(), number, owner))
            if not changed.rowcount:
                raise HTTPException(409, "本病例已有分析结果")
            return read_case(database, number, owner)

    @api.put("/api/cases/{identifier}/report")
    def report_draft(identifier: str, draft: ReportDraft, owner: str = Depends(record_owner),
                     user: str = Depends(current_user)):
        number = case_number(identifier)
        with connect() as database:
            database.execute("BEGIN IMMEDIATE")
            case = read_case(database, number, owner)
            store_report(database, number, case, draft, accounts[user][1])
            return read_case(database, number, owner)

    @api.post("/api/cases/{identifier}/review")
    def review(identifier: str, review: ReportConfirmation | Review, owner: str = Depends(record_owner),
               reviewer: str = Depends(current_user)):
        number = case_number(identifier)
        with connect() as database:
            database.execute("BEGIN IMMEDIATE")
            case = read_case(database, number, owner)
            if case["status"] != "analyzed":
                raise HTTPException(409, "仅待复核病例可以提交复核")
            name = accounts[reviewer][1]
            if isinstance(review, ReportConfirmation):
                note = review.conclusion.strip()
                empty = not (review.findings.strip() or note)
            else:
                if case["report"]["revision"]:
                    raise HTTPException(409, "已有结构化草稿，请核对最新报告后提交")
                note = review.note.strip()
                empty = not note
            if empty and not review.empty_review_confirmed:
                raise HTTPException(422, "复核内容为空，请再次确认留空提交")
            if isinstance(review, ReportConfirmation):
                store_report(database, number, case, review, name, confirmed_empty=empty)
            database.execute("""UPDATE cases SET status='reviewed', reviewer=?, review_note=?, reviewed_at=?,
                empty_review_confirmed=? WHERE number=? AND owner=?""", (name, note, timestamp(), empty, number, owner))
            return read_case(database, number, owner)

    api.mount("/static", StaticFiles(directory=ROOT / "backend/static"), name="assets")
    api.mount("/", StaticFiles(directory=ROOT / "backend/static", html=True), name="workstation")
    return api


app = create_app()
