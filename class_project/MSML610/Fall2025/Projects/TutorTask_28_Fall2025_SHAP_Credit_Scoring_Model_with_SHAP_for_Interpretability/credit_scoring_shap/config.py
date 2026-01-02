from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """
    Configuration for loading and splitting the German Credit dataset.
    """
    target_col: str = "Risk"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    """
    Configuration for the XGBoost classifier.
    Defaults are a solid starting point for tabular credit data.
    """
    learning_rate: float = 0.05
    max_depth: int = 4
    n_estimators: int = 400
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0


@dataclass
class TrainingConfig:
    """
    Top-level training configuration tying data + model + reporting.
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reports_dir: str = "reports"

    def ensure_reports_dir(self) -> Path:
        """
        Make sure the reports directory exists and return it.
        """
        path = Path(self.reports_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
