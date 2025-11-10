# Auto-Sklearn Anomaly Detection: API

This document defines the "API," or programming contract, for the anomaly detection models used in this project, as specified by the class `README.md`.

## API Design

The `README.md` file for the class project requires a clear separation between the "API" (the interface) and the "Example" (the implementation). We defined our API in `autosklearn.API.ipynb`.

Our API consists of two primary components:

### 1. `ModelEvaluation` Dataclass
A simple Python `dataclass` to standardize how we store evaluation results. It holds the model name, precision, recall, and F1-score.

```python
@dataclass
class ModelEvaluation:
    model_name: str
    precision: float
    recall: float
    f1_score: float