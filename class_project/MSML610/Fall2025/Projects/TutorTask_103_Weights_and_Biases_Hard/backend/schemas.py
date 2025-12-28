from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class PredictRequest(BaseModel):
    ticker: str = Field(default="AAPL", description="Yahoo Finance ticker symbol (e.g., AAPL)")
    lookback_days: int = Field(default=365, ge=120, le=5000, description="How many days to pull to build features")
    horizon_days: int = Field(default=1, ge=1, le=120, description="How many future business days to forecast")
    investment_usd: Optional[float] = Field(default=None, ge=0, description="Optional: investment amount in USD to compute profit/loss")
    shares: Optional[float] = Field(default=None, ge=0, description="Optional: number of shares to compute profit/loss")


class PredictFeaturesRequest(BaseModel):
    features: Dict[str, float] = Field(description="Feature values (must match training feature_names)")
    horizon_days: int = Field(default=1, ge=1, le=120, description="How many future steps to forecast (iterative)")
    current_price: Optional[float] = Field(default=None, ge=0, description="Optional current price baseline for profit/loss (manual mode)")
    investment_usd: Optional[float] = Field(default=None, ge=0, description="Optional: investment amount in USD to compute profit/loss")
    shares: Optional[float] = Field(default=None, ge=0, description="Optional: number of shares to compute profit/loss")


class ForecastPoint(BaseModel):
    step: int
    date: str
    predicted_close: float


class PredictResponse(BaseModel):
    ticker: str
    last_date: str
    last_close: float
    horizon_days: int
    predictions: List[ForecastPoint]
    model_name: str
    extra: Optional[Dict[str, Any]] = None


