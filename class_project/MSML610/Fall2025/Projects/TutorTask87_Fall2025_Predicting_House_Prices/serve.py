from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import json
from azua_utils import load_model

app = FastAPI(title="Melbourne House Price API")
model = load_model()

with open("artifacts/metrics.json") as f:
    meta = json.load(f)
EXPECTED_NUM: List[str] = meta.get("num_cols", [])
EXPECTED_CAT: List[str] = meta.get("cat_cols", [])
EXPECTED_ALL: List[str] = EXPECTED_NUM + EXPECTED_CAT

class House(BaseModel):
    features: Dict[str, Any]

def prepare_features(feat: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([feat])

    # Ensure Date compatible with training
    if "Date" in EXPECTED_CAT:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            df["Date"] = pd.NaT

    # Derive Year/Month like training did
    if "Year" in EXPECTED_NUM:
        df["Year"] = df.get("Date")
        df["Year"] = pd.to_datetime(df["Year"], errors="coerce").dt.year
    if "Month" in EXPECTED_NUM:
        df["Month"] = df.get("Date")
        df["Month"] = pd.to_datetime(df["Month"], errors="coerce").dt.month

    # Add missing expected cols as NaN, drop extras, enforce order
    for col in EXPECTED_ALL:
        if col not in df.columns:
            df[col] = np.nan
    df = df[EXPECTED_ALL]

    return df

@app.get("/")
def root():
    return {"ok": True, "usage": "POST /predict with {'features': {...}}"}

@app.get("/schema")
def schema():
    return {"numeric": EXPECTED_NUM, "categorical": EXPECTED_CAT}

@app.post("/predict")
def predict(house: House):
    try:
        X = prepare_features(house.features)
        yhat = model.predict(X)[0]
        return {"predicted_price": float(yhat)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
