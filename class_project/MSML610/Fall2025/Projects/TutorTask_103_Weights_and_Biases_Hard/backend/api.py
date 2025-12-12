from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import PredictRequest, PredictResponse
from backend.model_service import ModelService
from src.exception import ProjectException


def create_app() -> FastAPI:
    app = FastAPI(title="TutorTask103 API", version="1.0.0")

    # Allow local React dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = ModelService(project_root=Path(__file__).resolve().parents[1])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        p = service.metrics_path
        return {"path": str(p), "exists": p.exists(), "content": p.read_text() if p.exists() else None}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        try:
            out = service.predict_next_close(req.ticker, req.lookback_days)
        except ProjectException as e:
            raise RuntimeError(str(e))
        return PredictResponse(**out)

    return app


app = create_app()


