"""Static UI server + inference API for the Cambodian News Classifier.

Serves the hand-built front-end in ``app/html/`` and exposes a small JSON API
backed by the real fine-tuned checkpoints (via ``app.inference.predictor``):

    GET  /              -> app/html/index.html (+ styles.css, app.js, ...)
    GET  /api/meta      -> labels, model list, availability, defaults
    POST /api/classify  -> {category, confidence, scores, model} for one article

Run from the project root:

    python -m uvicorn app.server:app --reload --port 8000
    # or simply: python app/server.py
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from app.inference.predictor import (
    DEFAULT_MODEL,
    MODEL_INFO,
    available_models,
    classify,
    get_labels,
)

HTML_DIR = Path(__file__).resolve().parent / "html"
MIN_WORDS = 50


async def meta(_: Request) -> JSONResponse:
    avail = set(available_models())
    return JSONResponse(
        {
            "labels": get_labels(),
            "default_model": DEFAULT_MODEL if DEFAULT_MODEL in avail else (
                next(iter(avail), None)
            ),
            "min_words": MIN_WORDS,
            "models": [
                {
                    "key": key,
                    "display": info["display"],
                    "accuracy": info["accuracy"],
                    "macro_f1": info["macro_f1"],
                    "available": key in avail,
                }
                for key, info in MODEL_INFO.items()
            ],
        }
    )


async def classify_endpoint(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    text = (payload.get("text") or "").strip()
    model_key = payload.get("model") or DEFAULT_MODEL

    if not text:
        return JSONResponse({"error": "No text provided."}, status_code=400)
    if model_key not in available_models():
        return JSONResponse(
            {"error": f"Model '{model_key}' is not available."}, status_code=400
        )

    # Torch inference is blocking; keep the event loop responsive.
    scores = await run_in_threadpool(classify, text, model_key)
    top_cat = max(scores, key=scores.get)
    return JSONResponse(
        {
            "category": top_cat,
            "confidence": scores[top_cat],
            "scores": scores,
            "model": MODEL_INFO[model_key]["display"],
            "model_key": model_key,
        }
    )


def _pdf_to_text(data: bytes) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


async def extract_pdf(request: Request) -> JSONResponse:
    try:
        form = await request.form()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "Invalid upload."}, status_code=400)

    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse({"error": "No PDF file provided."}, status_code=400)

    data = await upload.read()
    if not data:
        return JSONResponse({"error": "Uploaded file is empty."}, status_code=400)

    try:
        text = await run_in_threadpool(_pdf_to_text, data)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"Could not read PDF: {exc}"}, status_code=400)

    if not text:
        return JSONResponse(
            {"error": "No extractable text found (the PDF may be scanned images)."},
            status_code=400,
        )
    return JSONResponse({"text": text, "words": len(text.split())})


async def _warm() -> None:
    """Preload the default model so the first real request is fast."""
    avail = available_models()
    if not avail:
        return
    key = DEFAULT_MODEL if DEFAULT_MODEL in avail else avail[0]
    await run_in_threadpool(classify, "warm up the model cache", key)


@asynccontextmanager
async def lifespan(_: Starlette):
    await _warm()
    yield


routes = [
    Route("/api/meta", meta, methods=["GET"]),
    Route("/api/classify", classify_endpoint, methods=["POST"]),
    Route("/api/extract-pdf", extract_pdf, methods=["POST"]),
    Mount("/", app=StaticFiles(directory=str(HTML_DIR), html=True), name="static"),
]

app = Starlette(routes=routes, lifespan=lifespan)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
