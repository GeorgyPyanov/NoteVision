"""FastAPI service exposing NoteVision predictions."""
from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from code.datasets.preprocess import CLASS_DISPLAY_NAMES
from code.models.features import extract_hog_features

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path(os.getenv("MODEL_PATH", REPO_ROOT / "models" / "model.joblib"))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}
RUSSIAN_NAMES = {
    "whole_note": "Целая нота", "half_note": "Половинная нота",
    "quarter_note": "Четвертная нота", "eighth_note": "Восьмая нота",
    "sixteenth_note": "Шестнадцатая нота",
}


def load_bundle(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the packaged sklearn model exactly once during application startup."""
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "config" not in bundle:
        raise ValueError("model.joblib must contain model and config")
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.bundle = load_bundle()
        app.state.model_load_error = None
        logging.info("Loaded NoteVision model version %s", app.state.bundle["config"].get("model_version"))
    except Exception as exc:  # health endpoint reports a clear service failure
        app.state.bundle = None
        app.state.model_load_error = str(exc)
        logging.exception("Could not load model from %s", MODEL_PATH)
    yield


app = FastAPI(title="NoteVision API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail={"status": "unavailable", "error": app.state.model_load_error})
    return {"status": "ok", "model_version": bundle["config"].get("model_version", "unknown")}


def _decode_upload(contents: bytes, content_type: str | None) -> Image.Image:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PNG, JPG, and JPEG images are accepted")
    if not contents or len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image must be between 1 byte and 5 MB")
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))
        image.load()
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("unexpected file format")
        return image
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable PNG/JPEG image") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    contents = await file.read()
    image = _decode_upload(contents, file.content_type)
    try:
        config = bundle["config"]
        model = bundle["model"]
        features = extract_hog_features(image, config["hog_params"]).reshape(1, -1)
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        best_index = int(np.argmax(probabilities))
        label = classes[best_index]
        probability_map = {name: float(probabilities[index]) for index, name in enumerate(classes)}
        return {
            "class": label,
            "display_name": CLASS_DISPLAY_NAMES[label],
            "display_name_ru": RUSSIAN_NAMES[label],
            "confidence": float(probabilities[best_index]),
            "probabilities": probability_map,
            "model_version": config.get("model_version", "unknown"),
        }
    finally:
        image.close()
