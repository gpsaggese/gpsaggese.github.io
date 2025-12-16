# CLIP ViT-L/14 Multimodal Sentiment Analysis — Example Usage

This example demonstrates an **end-to-end workflow** for multimodal sentiment analysis using **CLIP ViT-L/14 embeddings** and a **two-tower neural classifier**.
The goal of this notebook is to show **how the API can be applied in practice** on a real dataset, including training, evaluation, and inference.

---

## Overview of the Pipeline

1. Load precomputed CLIP image and text embeddings  
2. Train a two-tower multimodal classifier  
3. Evaluate model performance  
4. Run inference on unseen image–text pairs  

All dataset-specific logic and experimentation are contained here, **not in the API**.

---

## Dataset

- **Dataset**: MVSA (Multimodal Visual Sentiment Analysis)
- Each sample contains:
  - A social media image
  - Associated text (tweet/post)
  - Sentiment label: `negative`, `neutral`, or `positive`

Preprocessing and CLIP embedding extraction are assumed to be completed beforehand.

---

## Training the Model

```python
from CLIP_ViT_Large_Task22_utils import train_two_tower_classifier_v2

model, metrics = train_two_tower_classifier_v2(
    embeddings_parquet="artifacts/clip/mvsa_vitl14_img_txt_embeddings.parquet",
    d_model=256,
    epochs=50,
    batch_size=64,
    lr=5e-4,
    weight_decay=3e-4,
    modality_dropout_p=0.15,
    head_dropout=0.25,
    device="auto",
    patience=16
)

**Key Design Choices**

Two-tower architecture: separate projections for image and text
Modality dropout: improves robustness when one modality is noisy
Early stopping: prevents overfitting

**Model Evaluation**
After training, validation metrics such as:
Accuracy
Precision / Recall
F1-score
Confusion matrix
are used to assess performance across sentiment classes.

These metrics help analyze how well the model balances visual and textual signals.

**Inference on New Samples**

from CLIP_ViT_Large_Task22_utils import predict_sentiment_from_raw

result = predict_sentiment_from_raw(
    img_path="data/raw/sample_infer_img/img1.jpg",
    text="What a beautiful day!",
    model_ckpt="artifacts/models/best_model.pt",
    device="auto"
)

print(result)

**Output**

{
  "label": "positive",
  "probs": [0.05, 0.10, 0.85]
}

**Example Use Case**

Analyze sentiment of social media posts
Detect disagreement between image and text sentiment
Study robustness when one modality dominates the prediction

**Notes**

This example is dataset-specific by design
All reusable logic lives in the API module
The notebook focuses on application, not implementation details

**Reproducibility**

Fixed random seeds
Deterministic train/validation split
All dependencies listed in requirements.txt
Docker support provided for consistent execution

**Summary**

This example illustrates how a general-purpose multimodal sentiment API can be applied to a real-world dataset using CLIP embeddings, highlighting the effectiveness of combining visual and textual information for sentiment prediction.

