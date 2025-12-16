# Multimodal Sentiment Analysis using CLIP ViT-L/14  
**Task 22 — MSML 610 (Fall 2025)**

This project implements an **end-to-end multimodal sentiment analysis system** using **CLIP ViT-L/14 image–text embeddings** and a **two-tower neural classifier**.

All core functionality (model definition, training, and inference) is contained in a **single utility module**, making the project easy to understand, reuse, and reproduce.

---

## Project Objectives

- Use **pretrained CLIP ViT-L/14** embeddings for multimodal understanding  
- Train a **two-tower classifier** for image and text features  
- Improve robustness via **modality dropout**  
- Provide a **simple, reusable API** for training and inference  
- Ensure **reproducibility via Docker**

---

## Dataset

- **Dataset**: MVSA (Multimodal Visual Sentiment Analysis)
- Each example contains:
  - An image
  - Associated social media text
  - Sentiment label: `negative`, `neutral`, or `positive`

 CLIP embedding extraction and dataset preprocessing are performed prior to training.

---

## Repository Structure

|── data/
 ├── raw/ # Raw images and text
│ └── processed/ # Cleaned labels and splits
│
├── artifacts/
│ ├── clip/ # CLIP image & text embeddings (parquet)
│ └── models/ # Trained model checkpoints
│
├── notebooks/
│ ├── CLIP_ViT_Large_Task22.API.ipynb
│ └── CLIP_ViT_Large_Task22.example.ipynb
│
├── docs/
│ ├── CLIP_ViT_Large_Task22.API.md
│ └── CLIP_ViT_Large_Task22.example.md
│
├── CLIP_ViT_Large_Task22_utils.py # Model, training, and inference API
│
├── Dockerfile
├── requirements.txt
└── README.md


---

## Methodology

### CLIP Embeddings
- Pretrained **CLIP ViT-L/14**
- Image and text embeddings are **frozen**
- Enables efficient training without fine-tuning CLIP

### Model Architecture
- **Two-tower neural network**
  - Image projection head
  - Text projection head
- Feature fusion via concatenation
- Fully connected classification head

### Training Strategy
- Cross-entropy loss
- Adam optimizer with weight decay
- Early stopping based on validation loss
- **Modality dropout** to improve robustness

---

## API Usage

### Train the Model


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

----

## Methodology

### CLIP Embeddings
- Pretrained **CLIP ViT-L/14**
- Image and text embeddings are **frozen**
- Enables efficient training without fine-tuning CLIP

### Model Architecture
- **Two-tower neural network**
  - Image projection head
  - Text projection head
- Feature fusion via concatenation
- Fully connected classification head

### Training Strategy
- Cross-entropy loss
- Adam optimizer with weight decay
- Early stopping based on validation loss
- **Modality dropout** to improve robustness

----

## API Usage

### Train the Model


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

from CLIP_ViT_Large_Task22_utils import predict_sentiment_from_raw

result = predict_sentiment_from_raw(
    img_path="data/raw/sample_infer_img/img1.jpg",
    text="What a beautiful day!",
    model_ckpt="artifacts/models/best_model.pt",
    device="auto"
)

print(result)

{
  "label": "positive",
  "probs": [0.05, 0.10, 0.85]
}

**Notebooks:**

## API.ipynb
Documents the utility module

Explains:
What the API provides
How to call training and inference functions
No dataset- or experiment-specific logic

## example.ipynb
Demonstrates end-to-end application

Includes:
Dataset loading
Model training
Evaluation
Inference on unseen samples

**Evaluation**

Model performance is assessed using:
Accuracy
Precision
Recall
F1-score
Confusion matrix

Evaluation focuses on balancing visual and textual signals.

**Reproducibility**

Fixed random seeds
Deterministic train/validation split
All dependencies specified in requirements.txt
Fully reproducible using Docker

docker build -t clip-multimodal-sentiment .
docker run --rm -it clip-multimodal-sentiment

**Design Decisions**

Single-file utility API for clarity
Clean separation of reusable code and experiments
Minimal assumptions about dataset structure
Emphasis on reproducibility and readability

**Limitations & Future Work**

CLIP model is not fine-tuned end-to-end
No explicit modeling of image–text disagreement

**Possible extensions:**
Attention-based fusion
Contrastive fine-tuning
Zero-shot evaluation

Authors

Ayush Gaur(UID: 121333117)

Priya Gutti(UID: 121339744)

University of Maryland — MS in Data Science
Course: MSML 610