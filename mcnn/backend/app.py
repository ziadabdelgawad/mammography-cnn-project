from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.config import FRONTEND_DIR, IMAGE_SIZE, MODELS_DIR

app = FastAPI(title="MammoView CNN Project", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

MODEL_PATH = MODELS_DIR / "best_model.keras"
METADATA_PATH = MODELS_DIR / "best_model_metadata.json"
_model = None
_lock = threading.Lock()
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


def get_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(
                    "No selected trained model found. Train, evaluate, then run SELECT_BEST_MODEL.bat."
                )
            _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def metadata():
    if METADATA_PATH.is_file():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {"selected_model": "unknown", "threshold": 0.5}


def decode_image(data: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc
    image = image.resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32)


def predict_image(model, image: np.ndarray) -> float:
    batch = np.expand_dims(image, axis=0)
    return float(model.predict(batch, verbose=0).reshape(-1)[0])


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(FRONTEND_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/health")
def health():
    meta = metadata()
    return {
        "status": "ok",
        "model_available": MODEL_PATH.is_file(),
        "selected_model": meta.get("selected_model"),
    }


async def read_upload(upload: UploadFile) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Each image must be below 30 MB.")
    return data


@app.post("/api/predict")
async def predict(cc_file: UploadFile = File(...), mlo_file: UploadFile = File(...)):
    try:
        cc = decode_image(await read_upload(cc_file))
        mlo = decode_image(await read_upload(mlo_file))
        model = get_model()
        cc_prob = predict_image(model, cc)
        mlo_prob = predict_image(model, mlo)
        probability = (cc_prob + mlo_prob) / 2.0
        meta = metadata()
        threshold = float(meta.get("threshold", 0.5))
        prediction = "malignant" if probability >= threshold else "benign"
        confidence = probability if prediction == "malignant" else 1.0 - probability
        return {
            "prediction": prediction,
            "malignant_probability": probability,
            "benign_probability": 1.0 - probability,
            "confidence": confidence,
            "cc_probability": cc_prob,
            "mlo_probability": mlo_prob,
            "threshold": threshold,
            "selected_model": meta.get("selected_model", "unknown"),
            "fusion": "mean of independent CC and MLO probabilities",
            "warning": "Educational research prototype only; not for clinical diagnosis.",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
