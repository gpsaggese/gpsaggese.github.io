# Employee Attrition — API

## Purpose
Stable, minimal interface for training and serving an attrition model. Notebooks should import from `Attrition_utils.py` and avoid embedding complex logic.

## Artifacts
- `AttritionConfig`: dataclass for hyperparameters, seed, I/O options.
- `AttritionAPI` (Protocol): abstract interface.
- `AttritionService` (reference impl): scikit-learn pipeline (OHE + GradientBoosting).

## Usage (snippet)
```python
import pandas as pd
from Attrition_utils import AttritionConfig, AttritionService

cfg = AttritionConfig(seed=42)
svc = AttritionService(cfg).fit(pd.read_csv("data.csv"))
proba = svc.predict_proba(pd.read_csv("scoring.csv"))
Design Notes

Keep notebooks focused on demos; put reusable logic in Attrition_utils.py.

The API surface (Config, service methods) stays stable even if the model changes (e.g., swap to LightGBM/XGBoost).
