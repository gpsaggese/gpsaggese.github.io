from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import PredictRequest, PredictResponse, PredictFeaturesRequest
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
            out = service.predict_ticker(
                req.ticker,
                req.lookback_days,
                req.horizon_days,
                investment_usd=req.investment_usd,
                shares=req.shares,
            )
        except ProjectException as e:
            raise HTTPException(status_code=400, detail=str(e))
        return PredictResponse(**out)

    @app.post("/predict_features", response_model=PredictResponse)
    def predict_features(req: PredictFeaturesRequest):
        try:
            out = service.predict_from_features(
                req.features,
                horizon_days=req.horizon_days,
                current_price=req.current_price,
                investment_usd=req.investment_usd,
                shares=req.shares,
            )
        except ProjectException as e:
            raise HTTPException(status_code=400, detail=str(e))
        return PredictResponse(**out)

    @app.get("/feature_names")
    def feature_names():
        try:
            return {"feature_names": service.load_best().feature_names}
        except ProjectException as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/feature_template")
    def feature_template():
        """
        Returns a JSON object you can copy/paste for manual prediction.
        Values default to 0.0; you must fill in meaningful values for best results.
        """
        try:
            return {"features": service.feature_template()}
        except ProjectException as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app


app = create_app()


