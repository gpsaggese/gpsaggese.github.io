from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class PredictRequest(BaseModel):
    ticker: str = Field(default="AAPL", description="Yahoo Finance ticker symbol (e.g., AAPL)")
    lookback_days: int = Field(default=365, ge=120, le=5000, description="How many days to pull to build features")


class PredictResponse(BaseModel):
    ticker: str
    last_date: str
    last_close: float
    predicted_next_close: float
    model_name: str
    extra: Optional[Dict[str, Any]] = None


