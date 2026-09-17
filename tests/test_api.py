from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from code.deployment.api import main
from code.models.features import DEFAULT_HOG_PARAMS


class DummyModel:
    classes_ = np.array(["whole_note", "quarter_note"])

    def predict_proba(self, features):
        return np.array([[0.1, 0.9]])


@pytest.fixture()
def client(monkeypatch):
    bundle = {"model": DummyModel(), "config": {"model_version": "test-v1", "hog_params": DEFAULT_HOG_PARAMS}}
    monkeypatch.setattr(main, "load_bundle", lambda: bundle)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_image(client):
    image = Image.new("L", (20, 30), color=255)
    image.putpixel((10, 10), 0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = client.post("/predict", files={"file": ("note.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json()["class"] == "quarter_note"
    assert response.json()["display_name_ru"] == "Четвертная нота"
    assert response.json()["confidence"] == pytest.approx(0.9)


def test_predict_rejects_invalid_file(client):
    response = client.post("/predict", files={"file": ("not-a-note.txt", b"not image", "text/plain")})
    assert response.status_code == 400
