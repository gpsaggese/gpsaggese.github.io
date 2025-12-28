# Twitter Airline Sentiment Analysis

Name : Charan Kankanala
UID : 120983184
Course : MSML 610

Sentiment classification on airline tweets using TF-IDF + Logistic Regression.

**Difficulty:** Hard | **Classes:** 3 (negative, neutral, positive) | **Samples:** 14,640 | **Tool:** YData Profiling

## Quick Start (Docker)

### Prerequisites
- Docker installed ([get it](https://www.docker.com/))
- This repository cloned

### Run

```bash
cd UmdTask38_Fall2025_YData-profiling_Sentiment_Analysis_on_Social_Media_Posts

# Build image (first time only)
bash docker_build.sh

# Start Jupyter in container
bash docker_jupyter.sh

# Open the URL in browser
# http://127.0.0.1:8888/?token=...
```

### Train & Evaluate

1. Open `sentiment.example.ipynb`
2. Cell → Run All
3. This trains the model and saves artifacts (takes ~3-5 min)

### Make Predictions

1. Open `sentiment.API.ipynb`  
2. Cell → Run All
3. This demonstrates the API (takes ~1 min)

## Project Structure

```
UmdTask38_Fall2025_YData-profiling_Sentiment_Analysis.../
├── data/
│   └── Tweets.csv                    # 14,640 tweets
├── sentiment_utils.py                # Core module
├── sentiment.example.ipynb           # Training notebook
├── sentiment.example.md              # Training docs
├── sentiment.API.ipynb               # Inference notebook
├── sentiment.API.md                  # API docs
├── tfidf_vectorizer.joblib          # Saved vectorizer (created by training)
├── logreg_sentiment_model.joblib    # Saved model (created by training)
├── airline_sentiment_profile.html   # EDA report (created by training)
├── requirements.txt                  # Dependencies
├── README.md                         # This file
├── Dockerfile                        # Container config
└── docker_*.sh                       # Docker scripts
```

## What It Does

### Training (`sentiment.example.ipynb`)

1. Load 14,640 tweets
2. YData profiling for EDA
3. Clean text (remove URLs, @mentions, special chars)
4. Create numeric labels (0=negative, 1=neutral, 2=positive)
5. Split: train (70%), val (10%), test (20%)
6. Vectorize with TF-IDF (20K features)
7. Train Logistic Regression
8. Evaluate with metrics and confusion matrix
9. Analyze errors and important features
10. Save vectorizer + model

### Inference (`sentiment.API.ipynb`)

1. Load pre-trained artifacts
2. Demonstrate SentimentAPI class
3. Single + batch predictions
4. Confidence scores
5. Real-world examples

## Model Details

**Vectorizer:** TF-IDF
- 20,000 features (vocabulary)
- Unigrams + bigrams
- Min document frequency: 5

**Classifier:** Logistic Regression
- Balanced class weights (handles imbalance)
- Max iterations: 1000
- L2 regularization

**Training Data:**
- 14,640 tweets
- Classes: negative (63%), neutral (21%), positive (16%)
- Class imbalance ratio: 3.9:1

## Results

**Test Set Performance:**
- Accuracy: ~0.776
- Precision (macro): ~0.725
- Recall (macro): ~0.744
- F1-Score (macro): ~0.731

**Per-Class:**
- Negative: Precision 0.83, Recall 0.82
- Neutral: Precision 0.75, Recall 0.68 (hardest)
- Positive: Precision 0.79, Recall 0.84

## Key Features

✓ **Class Imbalance Handling** - Balanced class weights + stratified sampling  
✓ **Comprehensive EDA** - YData profiling + visualizations  
✓ **Error Analysis** - Misclassification breakdown + examples  
✓ **Feature Analysis** - Top important words per class  
✓ **Clean API** - Simple methods for predictions  
✓ **Production-Ready** - Saved artifacts for deployment  
✓ **Well Documented** - Markdown docs + code comments  
✓ **Reproducible** - Fixed random seed + Docker environment  

## Why Neutral is Hard

Neutral tweets are:
- Only 21% of training data (vs 63% negative)
- Less distinctive (more factual, less emotional)
- Boundary cases (could be slightly negative or positive)
- Varied content types

Example neutral tweets:
```
"The flight arrived on time."
"We had a seat."
"Service was okay."
```

These lack strong sentiment words, making classification ambiguous.

## Class Imbalance Challenge

**The Problem:**
- 9,178 negative tweets (63%)
- 2,363 positive tweets (16%)
- 3.9:1 imbalance

**Solutions Implemented:**
1. Balanced class weights in LogisticRegression
2. Stratified sampling (maintains distribution in splits)
3. Macro-averaged metrics (equal weight per class)

**Result:** Fair performance across all classes despite imbalance

## API Usage

```python
# Load
api = SentimentAPI(vectorizer, model)

# Single prediction
sentiment = api.predict("Great flight!")
# → "positive"

# Batch prediction
sentiments = api.predict_batch([...])
# → ["positive", "negative", "neutral", ...]

# With confidence
probs = api.predict_with_confidence("Good but expensive")
# → {"negative": 0.3, "neutral": 0.5, "positive": 0.2}
```

## Docker Commands

```bash
# Build
bash docker_build.sh

# Run Jupyter
bash docker_jupyter.sh

# Clean up
bash docker_clean.sh

# Bash shell in container
bash docker_bash.sh
```

## Files Explanation

| File | Purpose |
|------|---------|
| `sentiment_utils.py` | Core functions (load, preprocess, train, predict) |
| `sentiment.example.ipynb` | Full training pipeline |
| `sentiment.example.md` | Training documentation |
| `sentiment.API.ipynb` | Inference demonstrations |
| `sentiment.API.md` | API documentation |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container configuration |
| `data/Tweets.csv` | Input dataset |
| `*.joblib` | Saved model artifacts |
| `*.html` | EDA report |

## Limitations

- **Domain:** Trained on airline tweets only
- **Language:** English only
- **Sarcasm:** May misclassify sarcastic text
- **Neutral:** Inherently harder (68% recall vs 84% for positive)
- **Context:** Each tweet treated independently
- **Spelling:** Sensitive to exact wording

## Improvements Could Include

- BERT/Transformers (likely 85%+ accuracy)
- Multi-domain training
- Sarcasm detection module
- Conversation context
- Active learning on hard examples
- Multilingual support

## Troubleshooting

**Docker won't build:**
```bash
# Check Docker running
docker ps

# Try again with fresh cache
bash docker_build.sh --no-cache
```

**Port 8888 in use:**
```bash
# Kill existing Jupyter
pkill -f jupyter
```

**Artifacts missing:**
```bash
# Run sentiment.example.ipynb first to create them
```

**Import errors:**
```bash
# In container, install deps
pip install -r requirements.txt
```

## Files to Submit

- `sentiment_utils.py`
- `sentiment.example.ipynb`
- `sentiment.example.md`
- `sentiment.API.ipynb`
- `sentiment.API.md`
- `README.md`
- `requirements.txt`
- `Dockerfile`
- `data/Tweets.csv`
- `tfidf_vectorizer.joblib`
- `logreg_sentiment_model.joblib`
- `airline_sentiment_profile.html`
- Docker scripts (`docker_*.sh`)

## Performance Notes

- **Training Time:** 3-5 minutes on modern laptop
- **Inference:** ~5ms per single prediction
- **Batch:** ~1.5ms per text when processing 1000+
- **Memory:** ~30MB for loaded models
- **Storage:** ~20MB for artifacts

## References

- TF-IDF: [Scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- Logistic Regression: [Scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- Class Imbalance: [Handling Imbalanced Data](https://imbalanced-learn.org/)

## Author Notes

This project demonstrates:
- Complete ML pipeline (data → training → evaluation → deployment)
- Handling imbalanced classification
- Text preprocessing and feature engineering
- Model evaluation and error analysis
- Production-ready code structure
- Docker containerization
- Clear documentation

Built for MSML610 Fall 2025.

## Getting Help

- `sentiment.example.md` - how training works
- `sentiment.API.md` - how to use predictions
- Code comments in `sentiment_utils.py`
- Output from running notebooks
