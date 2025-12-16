# CLIP ViT-L/14 Multimodal Sentiment Classifier API

This API provides a lightweight interface for **training and inference** of a **two-tower multimodal sentiment classifier** built on top of **precomputed CLIP ViT-L/14 image and text embeddings**.

The tool is model-agnostic to the dataset and can be reused for any task where aligned image–text embeddings are available.

---

## What This Tool Does

- Trains a **two-tower neural classifier** on CLIP image + text embeddings  
- Supports **modality dropout** to improve robustness  
- Provides **simple inference** from raw image + text inputs  
- Handles device selection (`cpu`, `cuda`, `mps`, or `auto`)

**Sentiment Labels**
negative -> 0
neutral -> 1
positive -> 2


---

## Main Functions

### 1. Train a Two-Tower Classifier

```python
from CLIP_ViT_Large_Task22_utils import train_two_tower_classifier_v2

model, metrics = train_two_tower_classifier_v2(
    embeddings_parquet="path/to/clip_embeddings.parquet",
    d_model=256,
    epochs=20,
    batch_size=128,
    lr=5e-4,
    weight_decay=1e-4,
    modality_dropout_p=0.15,
    head_dropout=0.25,
    device="auto",
    patience=10
)

Inputs:

embeddings_parquet: Parquet file containing CLIP image & text embeddings
d_model: Hidden dimension of projection layers
modality_dropout_p: Probability of dropping image or text tower
device: "cpu" | "cuda" | "mps" | "auto"

Outputs:
Trained PyTorch model
Dictionary of training & validation metrics

**Predict Sentiment From Raw Inputs:**

from CLIP_ViT_Large_Task22_utils import predict_sentiment_from_raw

result = predict_sentiment_from_raw(
    img_path="sample.jpg",
    text="What a great day!",
    model_ckpt="model.pt",
    device="auto"
)

print(result["label"], result["probs"])

Returns
{
  "label": "positive",
  "probs": [0.03, 0.12, 0.85]
}

**Expected Embedding Format**

The training parquet file must contain:
image_embedding: CLIP image embedding (vector)
text_embedding: CLIP text embedding (vector)
label: Sentiment label (negative, neutral, positive)

**Design Philosophy**

Separation of concerns: embeddings, model, and training are decoupled
Reusable across datasets and domains
Minimal assumptions about data source or task
Production-ready structure (device handling, early stopping)

**Notes**

This API does not include dataset-specific preprocessing
Project-specific experiments and analysis belong in example.ipynb
The API notebook should only demonstrate how to use the tool

**Dependencies**

PyTorch
NumPy
Pandas
CLIP (OpenAI)
torchvision, PIL
