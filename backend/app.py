import hashlib
import io
import json
import logging
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
from pydantic import BaseModel, Field

from . import detector


ROOT = Path(__file__).resolve().parents[1]
COOKIE = "equivision_session"
SESSION_TTL = 8 * 3600
MAX_BYTES = 10 * 1024 * 1024
MIMES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
CASE_FIELDS = """number, horse_name, comment, created_at, archived_at, status,
    detections, model, threshold, analyzed_at, reviewer, review_note, reviewed_at"""
CASE_SCHEMA = """CREATE TABLE IF NOT EXISTS cases (
    number INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL,
    horse_name TEXT NOT NULL, comment TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
    archived_at TEXT, status TEXT NOT NULL DEFAULT 'registered', detections TEXT NOT NULL DEFAULT '[]',
    model TEXT, threshold REAL, analyzed_at TEXT, reviewer TEXT, review_note TEXT, reviewed_at TEXT
)"""
IMAGES_SCHEMA = """CREATE TABLE IF NOT EXISTS case_images (
    case_number INTEGER NOT NULL REFERENCES cases(number) ON DELETE CASCADE,
    position INTEGER NOT NULL, filename TEXT NOT NULL, projection TEXT NOT NULL,
    quality_confirmed INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
    mime TEXT NOT NULL, original BLOB NOT NULL, PRIMARY KEY(case_number, position)
)"""
Projection = Literal["正位", "侧位", "斜位", "其他"]


def prepare_database(database):
    database.execute("BEGIN IMMEDIATE")
    database.execute("""CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY, username TEXT NOT NULL, expires INTEGER NOT NULL)""")
    legacy = "original" in {row["name"] for row in database.execute("PRAGMA table_info(cases)")}
    if legacy:
        high_water = database.execute("SELECT seq FROM sqlite_sequence WHERE name='cases'").fetchone()
        database.execute("ALTER TABLE cases RENAME TO cases_v1")
    database.execute(CASE_SCHEMA)
    database.execute(IMAGES_SCHEMA)
    if legacy:
        database.execute("""INSERT INTO cases(number, owner, horse_name, created_at, status, detections,
            model, threshold, analyzed_at, reviewer, review_note, reviewed_at, archived_at)
            SELECT number, owner, horse_name, created_at, status, detections, model, threshold,
            analyzed_at, reviewer, review_note, reviewed_at, CASE WHEN status='reviewed' THEN reviewed_at END
            FROM cases_v1""")
        database.execute("""INSERT INTO case_images SELECT number, 0, filename, projection,
            quality_confirmed, width, height, mime, original FROM cases_v1""")
        database.execute("DROP TABLE cases_v1")
        if high_water:
            database.execute("UPDATE sqlite_sequence SET seq = max(seq, ?) WHERE name='cases'", (high_water["seq"],))
    database.execute("CREATE INDEX IF NOT EXISTS cases_owner ON cases(owner, number)")


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class Review(BaseModel):
    note: str = Field(min_length=1, max_length=2000)


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


def load_cases(database, owner, number=None):
    if not database.in_transaction:
        database.execute("BEGIN")
    condition = "owner = ?" + (" AND number = ?" if number is not None else "")
    params = (owner, number) if number is not None else (owner,)
    records = {}
    for row in database.execute(f"SELECT {CASE_FIELDS} FROM cases WHERE {condition} ORDER BY number DESC", params):
        case = dict(row)
        key = case.pop("number")
        case.update(id=f"EV-{key:06d}", detections=json.loads(case["detections"]), images=[],
                    has_image=False, filename=None, projection=None, width=None, height=None, quality_confirmed=False)
        records[key] = case
    for row in database.execute(f"""SELECT case_number, position, filename, projection, width, height, quality_confirmed
        FROM case_images JOIN cases ON number=case_number WHERE {condition} ORDER BY case_number, position""", params):
        image = dict(row)
        case = records[image.pop("case_number")]
        position = image.pop("position")
        image["quality_confirmed"] = bool(image["quality_confirmed"])
        if position == 0:
            case.update(image, has_image=True)
        image["id"] = str(position) if position else "primary"
        image["url"] = f"/api/cases/{case['id']}/images/{image['id']}"
        case["images"].append(image)
    return list(records.values())


def read_case(database, number, owner):
    records = load_cases(database, owner, number)
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
    database.execute("UPDATE cases SET status='uploaded' WHERE number=? AND owner=? AND status='registered'", (number, owner))


def create_case(database, owner, horse_name, comment):
    if not horse_name.strip():
        raise HTTPException(422, "请填写马匹名称或编号")
    return database.execute("INSERT INTO cases(owner, horse_name, comment, created_at) VALUES (?, ?, ?, ?)",
                            (owner, horse_name.strip(), comment.strip(), timestamp())).lastrowid


def create_app(data_dir=None):
    directory = Path(data_dir or os.environ.get("SPORT_MODEL_DATA_DIR", ROOT / "var"))
    username = os.environ.get("SPORT_MODEL_USERNAME", "demo")
    password = os.environ.get("SPORT_MODEL_PASSWORD", "EquiDemo2026!")
    display_name = os.environ.get("SPORT_MODEL_DISPLAY_NAME", "王一伊")
    secure_cookie = os.environ.get("SPORT_MODEL_SECURE_COOKIE") == "1"
    if not all((username, password, display_name)):
        raise ValueError("账号、密码与显示名不能为空")

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
            prepare_database(database)
        yield

    api = FastAPI(title="EQUIVISION 运动马影像工作台", lifespan=lifespan)

    @api.middleware("http")
    async def protect_api(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if request.method in {"POST", "PATCH", "DELETE"} and origin and origin != str(request.base_url).rstrip("/"):
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
            if session and session["username"] == username:
                return username
        raise HTTPException(401, "请先登录工作台")

    @api.post("/api/login")
    def login(credentials: Credentials, request: Request, response: Response):
        if not (secrets.compare_digest(username.encode(), credentials.username.encode())
                and secrets.compare_digest(password.encode(), credentials.password.encode())):
            raise HTTPException(401, "账号或密码不正确")
        now = int(time.time())
        token = secrets.token_urlsafe(32)
        with connect() as database:
            database.execute("DELETE FROM sessions WHERE expires <= ? OR token_hash = ?",
                             (now, token_hash(request.cookies.get(COOKIE, ""))))
            database.execute("INSERT INTO sessions VALUES (?, ?, ?)", (token_hash(token), username, now + SESSION_TTL))
        response.set_cookie(COOKIE, token, httponly=True, secure=secure_cookie, samesite="strict", max_age=SESSION_TTL)
        return {"username": username, "display_name": display_name}

    @api.get("/api/me")
    def me(owner: str = Depends(current_user)):
        return {"username": owner, "display_name": display_name}

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

    @api.get("/api/cases")
    def cases(owner: str = Depends(current_user)):
        with connect() as database:
            return load_cases(database, owner)

    @api.get("/api/cases/{identifier}")
    def case_detail(identifier: str, owner: str = Depends(current_user)):
        with connect() as database:
            return read_case(database, case_number(identifier), owner)

    @api.patch("/api/cases/{identifier}")
    def edit(identifier: str, update: VisitUpdate, owner: str = Depends(current_user)):
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
    def delete(identifier: str, owner: str = Depends(current_user)):
        with connect() as database:
            changed = database.execute("DELETE FROM cases WHERE number=? AND owner=?", (case_number(identifier), owner))
            if not changed.rowcount:
                raise HTTPException(404, "未找到此病例")
        return Response(status_code=204)

    @api.post("/api/cases/{identifier}/archive")
    def archive(identifier: str, owner: str = Depends(current_user)):
        number = case_number(identifier)
        with connect() as database:
            database.execute("UPDATE cases SET archived_at=COALESCE(archived_at, ?) WHERE number=? AND owner=?",
                             (timestamp(), number, owner))
            return read_case(database, number, owner)

    @api.post("/api/cases/{identifier}/images", status_code=201)
    def add_image(identifier: str, image: UploadFile = File(), projection: Projection = Form(),
                  quality_confirmed: bool = Form(), owner: str = Depends(current_user)):
        number = case_number(identifier)
        values = attachment(image, projection, quality_confirmed)
        with connect() as database:
            save_attachment(database, number, owner, values)
            return read_case(database, number, owner)

    @api.get("/api/cases/{identifier}/images/{image_id}")
    def visit_image(identifier: str, image_id: str, owner: str = Depends(current_user)):
        if image_id == "primary":
            position = 0
        elif image_id.isascii() and image_id.isdecimal() and 0 < len(image_id) <= 18 and not image_id.startswith("0"):
            position = int(image_id)
        else:
            raise HTTPException(404, "未找到此影像")
        with connect() as database:
            row = original(database, case_number(identifier), owner, position)
        return Response(row["original"], media_type=row["mime"])

    @api.get("/api/cases/{identifier}/image")
    def case_image(identifier: str, owner: str = Depends(current_user)):
        with connect() as database:
            row = original(database, case_number(identifier), owner)
        return Response(row["original"], media_type=row["mime"])

    @api.post("/api/cases/{identifier}/analyze")
    def analyze(identifier: str, owner: str = Depends(current_user)):
        number = case_number(identifier)
        with connect() as database:
            if read_case(database, number, owner)["status"] == "registered":
                raise HTTPException(409, "请先上传用于分析的主影像")
            row = original(database, number, owner)
        if row["status"] != "uploaded":
            raise HTTPException(409, "本病例已有分析结果")
        try:
            with Image.open(io.BytesIO(row["original"])) as image:
                predictions = detector.detect(ImageOps.exif_transpose(image))
        except Exception as error:
            logging.getLogger(__name__).exception("病例 %s 推理失败", identifier)
            raise HTTPException(503, "模型推理失败，原片已保留，可稍后重试") from error
        with connect() as database:
            changed = database.execute("""
                UPDATE cases SET status='analyzed', detections=?, model=?, threshold=?, analyzed_at=?
                WHERE number=? AND owner=? AND status='uploaded'
            """, (json.dumps(predictions, allow_nan=False), detector.MODEL_ID, detector.THRESHOLD, timestamp(), number, owner))
            if not changed.rowcount:
                raise HTTPException(409, "本病例已有分析结果")
            return read_case(database, number, owner)

    @api.post("/api/cases/{identifier}/review")
    def review(identifier: str, review: Review, owner: str = Depends(current_user)):
        if not review.note.strip():
            raise HTTPException(422, "请填写复核意见")
        number = case_number(identifier)
        now = timestamp()
        with connect() as database:
            changed = database.execute("""
                UPDATE cases SET status='reviewed', reviewer=?, review_note=?, reviewed_at=?, archived_at=COALESCE(archived_at, ?)
                WHERE number=? AND owner=? AND status='analyzed'
            """, (display_name, review.note.strip(), now, now, number, owner))
            case = read_case(database, number, owner)
            if not changed.rowcount:
                raise HTTPException(409, "仅待复核病例可以提交复核")
            return case

    api.mount("/static", StaticFiles(directory=ROOT / "backend/static"), name="assets")
    api.mount("/", StaticFiles(directory=ROOT / "backend/static", html=True), name="workstation")
    return api


app = create_app()
