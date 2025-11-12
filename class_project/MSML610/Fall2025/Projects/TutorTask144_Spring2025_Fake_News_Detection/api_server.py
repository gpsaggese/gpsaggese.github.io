#!/usr/bin/env python3
"""
FastAPI server for fake news detection inference.

Provides REST endpoints for:
- Single prediction: /predict
- Batch predictions: /predict_batch
- Model info: /info
- Health check: /health
"""

import logging
from typing import List, Optional, Dict
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

from bert_utils import BertModelWrapper, TrainingConfig
from confidence_utils import ConfidenceScorer
from ensemble_utils import EnsembleModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="BERT-based fake news classification with confidence scoring",
    version="1.0.0"
)

# Global model state
model = None
scorer = None
ensemble = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Text to classify")
    use_confidence: bool = Field(False, description="Include confidence scores")
    use_ensemble: bool = Field(False, description="Use ensemble model")
    threshold: float = Field(0.5, description="Decision threshold (0.0-1.0)")


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="Texts to classify")
    use_confidence: bool = Field(False, description="Include confidence scores")
    use_ensemble: bool = Field(False, description="Use ensemble model")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=Fake, 1=Real")
    prediction_label: str = Field(..., description="Fake or Real")
    probability: float = Field(..., description="Probability of prediction")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    threshold: float = Field(..., description="Decision threshold used")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    num_labels: int
    max_text_length: int
    device: str
    ensemble_enabled: bool


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, scorer, ensemble

    try:
        logger.info("Loading BERT model...")
        config = TrainingConfig(
            model_name='distilbert-base-uncased',
            batch_size=16,
            learning_rate=2e-5,
            num_epochs=2,
            device='cuda',
            max_text_length=256
        )
        model = BertModelWrapper(config)

        # Try to load pre-trained weights
        model_path = Path('models/distilbert_weighted')
        if model_path.exists():
            logger.info(f"Loading weights from {model_path}...")
            model.load_model(str(model_path))
        else:
            logger.warning(f"Model weights not found at {model_path}. Using fresh model.")

        # Initialize confidence scorer
        scorer = ConfidenceScorer(model=model, device='cuda')

        # Try to initialize ensemble
        try:
            ensemble_path = Path('models/ensemble_combined')
            if ensemble_path.exists():
                logger.info("Loading ensemble model...")
                ensemble = EnsembleModel(bert_model=model)
                logger.info("Ensemble model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ensemble: {e}")

        logger.info("Model loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "ensemble_available": ensemble is not None
    }


@app.get("/info", response_model=ModelInfoResponse, tags=["Info"])
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name="distilbert-base-uncased",
        model_type="transformer",
        num_labels=2,
        max_text_length=256,
        device=str(model.config.device),
        ensemble_enabled=ensemble is not None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a single prediction.

    Args:
        request: Prediction request with text and options

    Returns:
        Prediction with optional confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get prediction
        if request.use_ensemble and ensemble is not None:
            preds, probs = ensemble.predict([request.text])
        else:
            preds, probs = model.predict_with_threshold([request.text], request.threshold)

        pred = preds[0]
        prob = probs[0]

        # Get confidence if requested
        confidence = None
        if request.use_confidence:
            conf_result = scorer.predict_with_confidence(
                [request.text],
                method='probability',
                threshold=request.threshold
            )
            confidence = conf_result['confidence'][0]

        return PredictionResponse(
            prediction=int(pred),
            prediction_label="Real" if pred == 1 else "Fake",
            probability=float(prob),
            confidence=confidence,
            threshold=request.threshold
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.

    Args:
        request: Batch prediction request

    Returns:
        Batch of predictions with timing info
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import time

        start_time = time.time()

        # Get predictions
        if request.use_ensemble and ensemble is not None:
            preds, probs = ensemble.predict(request.texts)
        else:
            preds, probs = model.predict_with_threshold(request.texts)

        # Get confidence if requested
        confidences = None
        if request.use_confidence:
            conf_result = scorer.predict_with_confidence(
                request.texts,
                method='probability'
            )
            confidences = conf_result['confidence']

        # Build response
        predictions = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            confidence = confidences[i] if confidences else None
            predictions.append(
                PredictionResponse(
                    prediction=int(pred),
                    prediction_label="Real" if pred == 1 else "Fake",
                    probability=float(prob),
                    confidence=confidence,
                    threshold=0.5
                )
            )

        processing_time_ms = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/confidence", tags=["Analysis"])
async def get_confidence(request: PredictionRequest):
    """
    Get detailed confidence analysis.

    Methods: 'probability', 'entropy', 'margin'
    """
    if model is None or scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = scorer.predict_with_confidence(
            [request.text],
            method='probability',
            threshold=request.threshold
        )

        return {
            "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
            "prediction": result['predictions'][0],
            "probability": result['probabilities'][0],
            "confidence": result['confidence'][0],
            "high_confidence": result['high_confidence'][0],
            "method": result['method']
        }

    except Exception as e:
        logger.error(f"Confidence analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Analysis"])
async def get_metrics():
    """Get model metrics from last evaluation."""
    try:
        metrics_file = Path('models/distilbert_weighted/metrics.json')
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        else:
            return {"message": "No metrics available"}
    except Exception as e:
        logger.error(f"Error reading metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the API server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
