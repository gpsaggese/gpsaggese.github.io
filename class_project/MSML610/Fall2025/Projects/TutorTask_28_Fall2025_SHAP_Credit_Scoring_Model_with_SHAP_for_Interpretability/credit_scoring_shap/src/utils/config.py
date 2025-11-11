from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    data_raw: Path = root / "data" / "raw" / "german_credit_data.csv"
    data_dir: Path = root / "data"
    interim_dir: Path = data_dir / "interim"
    processed_dir: Path = data_dir / "processed"
    models_dir: Path = root / "models"
    reports_dir: Path = root / "reports"
    pipeline_path: Path = models_dir / "pipeline.joblib"
    model_path: Path = models_dir / "xgb_model.json"
    metrics_path: Path = root / "reports" / "metrics.json"
    cm_png: Path = root / "reports" / "confusion_matrix.png"
    shap_summary_png: Path = root / "reports" / "shap_summary.png"
    shap_beeswarm_png: Path = root / "reports" / "shap_beeswarm.png"

# You can tweak these if your file layout differs.
