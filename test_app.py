import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app_final import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "depth.db"))
    monkeypatch.setenv("MODEL_WEIGHTS_PATH", str(Path("models/model.pt").resolve()))
    monkeypatch.setenv("APP_VERSION", "test")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    with TestClient(create_app()) as client:
        yield client


def test_health_and_version(client):
    assert client.get("/health").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/version").json() == {"version": "test"}


def test_prediction_history_and_downloads(client):
    response = client.post("/predict", files={
        "image": ("example.jpg", Path("examples/example1.jpg").read_bytes(), "image/jpeg")
    })
    assert response.status_code == 201
    item = response.json()
    assert item["name"] == "example.jpg"
    path = f"/items/{item['request_id']}"
    png = client.get(f"{path}/depth.png")
    assert png.status_code == 200
    assert Image.open(io.BytesIO(png.content)).size == (224, 224)
    npy = client.get(f"{path}/depth.npy")
    assert npy.status_code == 200
    depth = np.load(io.BytesIO(npy.content), allow_pickle=False)
    assert depth.shape == (224, 224)
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    with TestClient(create_app()) as restarted:
        assert restarted.get("/items?limit=1").json() == [item]
        assert restarted.get(f"{path}/depth.npy").content == npy.content


def test_invalid_image(client):
    response = client.post("/predict", files={"image": ("broken.jpg", b"broken")})
    assert response.status_code == 400
    assert client.get("/items").json() == []


def test_unknown_result(client):
    assert client.get("/items/missing/depth.png").status_code == 404


def test_upload_limit(client):
    response = client.post("/predict", files={"image": ("large.jpg", b"x" * (1024 * 1024 + 1))})
    assert response.status_code == 413
    assert client.get("/items").json() == []
