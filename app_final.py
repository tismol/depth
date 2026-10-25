import io
import logging
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager, closing, contextmanager
from pathlib import Path
from threading import Lock
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response
from PIL import Image

from model import predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("depth")


@contextmanager
def connect(path):
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            yield connection


def create_app():
    database = os.getenv("DATABASE_PATH", "data/depth.db")
    weights = os.getenv("MODEL_WEIGHTS_PATH", "models/model.pt")
    version = os.getenv("APP_VERSION", "1.0.0")
    max_bytes = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    if max_bytes <= 0:
        raise ValueError("MAX_UPLOAD_MB должен быть положительным")
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        Path(database).parent.mkdir(parents=True, exist_ok=True)
        with connect(database) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    request_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    depth_png BLOB NOT NULL,
                    depth_npy BLOB NOT NULL
                )
            """)
        logger.info("Depth API started version=%s", version)
        yield

    app = FastAPI(title="Depth API", version=version, lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/docs")

    @app.get("/health")
    def health():
        try:
            with connect(database) as connection:
                connection.execute("SELECT 1 FROM results LIMIT 1")
        except sqlite3.Error:
            raise HTTPException(503, "База данных недоступна") from None
        if not Path(weights).is_file():
            raise HTTPException(503, "Файл модели не найден")
        return {"status": "ok"}

    @app.get("/version")
    def app_version():
        return {"version": version}

    @app.post("/predict", status_code=201)
    def predict_image(image: UploadFile = File(...)):
        with image.file as source:
            raw = source.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise HTTPException(413, "Файл превышает MAX_UPLOAD_MB")
        try:
            with Image.open(io.BytesIO(raw)) as picture:
                if picture.width * picture.height > 20_000_000:
                    raise HTTPException(413, "Изображение превышает 20 млн пикселей")
                if picture.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(400, "Поддерживаются JPG, PNG и WebP")
                picture.load()
        except Image.DecompressionBombError:
            raise HTTPException(413, "Слишком большое изображение") from None
        except (OSError, ValueError):
            raise HTTPException(400, "Некорректное изображение") from None
        try:
            with lock:
                png, npy = predict(raw, weights)
        except Exception:
            logger.exception("Prediction failed")
            raise HTTPException(503, "Ошибка обработки изображения") from None
        request_id = str(uuid.uuid4())
        name = Path((image.filename or "image").replace("\\", "/")).name
        with connect(database) as connection:
            row = connection.execute("""
                INSERT INTO results (request_id, name, depth_png, depth_npy)
                VALUES (?, ?, ?, ?)
                RETURNING request_id, name, created_at
            """, (request_id, name, png, npy)).fetchone()
        logger.info("Prediction saved request_id=%s", request_id)
        return dict(row)

    @app.get("/items")
    def items(limit: int = Query(20, ge=1, le=100)):
        with connect(database) as connection:
            return [dict(row) for row in connection.execute("""
                SELECT request_id, name, created_at FROM results
                ORDER BY created_at DESC, rowid DESC LIMIT ?
            """, (limit,))]

    def download(request_id, extension):
        with connect(database) as connection:
            row = connection.execute(
                f"SELECT name, depth_{extension} AS content FROM results WHERE request_id = ?", (request_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Результат не найден")
        filename = quote(f"{Path(row['name']).stem}-result.{extension}", safe="")
        return Response(
            row["content"], media_type="image/png" if extension == "png" else "application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
        )

    @app.get("/items/{request_id}/depth.png")
    def depth_png(request_id):
        return download(request_id, "png")

    @app.get("/items/{request_id}/depth.npy")
    def depth_npy(request_id):
        return download(request_id, "npy")

    return app


app = create_app()
