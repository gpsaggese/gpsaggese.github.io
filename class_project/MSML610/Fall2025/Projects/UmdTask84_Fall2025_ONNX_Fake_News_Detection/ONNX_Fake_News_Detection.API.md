# ONNX Fake News Detection — API Documentation

## Purpose

This API defines the **internal programming interface** for the ONNX Fake News Detection tool.
It exposes a **stable contract** for training, converting, and running inference on a fake news classifier.

The API is designed so that:

* Notebooks call **only these functions**
* Consumers do **not** depend on TensorFlow or ONNX internals
* Implementation details can change without breaking users

---

## Scope of the API

Included:

* Dataset loading
* LSTM model training
* Conversion to ONNX
* ONNX-based inference
* Optional FastAPI prediction service

Excluded:

* Experimental models
* GPU / Torch training logic
* Visualization and benchmarking
* Notebook-specific code

---

## Data Conventions

### Labels

* `0` → Fake news
* `1` → Real news

### Input Text

* Raw English news text
* Title and body are combined internally
* No preprocessing required from the caller

---

## Core API Functions

### `load_fake_real_news`

```python
load_fake_real_news(
    fake_path: str = FAKE_PATH,
    real_path: str = REAL_PATH
) -> pandas.DataFrame
```

Loads the Kaggle Fake/True News dataset and returns a shuffled DataFrame.

**Output columns**

* `text`: combined title + article
* `label`: binary class (0 or 1)

This function centralizes dataset handling so downstream code does not depend on CSV structure.

---

### `train_lstm_model`

```python
train_lstm_model(
    num_samples: Optional[int] = None
) -> Dict[str, object]
```

Trains a BiLSTM-based fake news classifier using TensorFlow/Keras.

**Responsibilities**

* Dataset split (85% train / 15% test, stratified)
* Tokenization and padding
* Model training
* Saving artifacts:

  * Keras model (`.keras`)
  * Tokenizer (`.pkl`)
* Computing evaluation metrics

**Returns**

* Training history
* Test-set metrics (accuracy, precision, recall, F1)

This function defines the **only supported training workflow**.

---

### `convert_lstm_to_onnx`

```python
convert_lstm_to_onnx() -> str
```

Converts the trained Keras LSTM model to ONNX format.

**Why this exists**

* Keras `Sequential` models are not directly compatible with ONNX
* This wrapper ensures a valid, inference-safe ONNX graph

**Output**

* Path to the generated `.onnx` model

---

### `predict_lstm_onnx`

```python
predict_lstm_onnx(
    texts: List[str]
) -> List[Dict[str, float]]
```

Runs inference using ONNX Runtime.

**Input**

* List of raw text strings

**Output**

```json
{
  "text": "...",
  "label": 0 or 1,
  "score": probability_of_real_news
}
```

**Guarantees**

* Uses the same tokenizer as training
* Ensures correct data types for ONNX
* Produces predictions consistent with TensorFlow

This is the **primary inference entry point**.

---

## FastAPI Wrapper Layer

### Request Schema

```python
class FakeNewsRequest(BaseModel):
    text: str
```

Defines the input contract for prediction requests.

---

### `create_fastapi_app`

```python
create_fastapi_app(
    model_type: str = "lstm"
) -> FastAPI
```

Creates a minimal HTTP interface over ONNX inference.

**Endpoint**

```
POST /predict
```

**Request**

```json
{ "text": "news article..." }
```

**Response**

```json
{ "label": 1, "score": 0.92 }
```

**Design choices**

* Model and tokenizer loaded once at startup
* Only inference is exposed
* No training or conversion endpoints

---

## API Design Guarantees

The following are considered stable:

* Function names
* Function arguments
* Return formats
* Label semantics

Internal implementation may change without affecting API consumers.

---

## Intended Usage

* Notebooks import from `ONNX_Fake_News_Detection_utils.py`
* All ML logic lives in the API module
* Notebooks remain short, readable, and reproducible
* Applications interact only with ONNX inference or the FastAPI layer

---

## Summary

This API provides:

* A single, consistent training path
* A reliable ONNX inference interface
* A minimal deployment wrapper
* Clear separation between tooling and experimentation

