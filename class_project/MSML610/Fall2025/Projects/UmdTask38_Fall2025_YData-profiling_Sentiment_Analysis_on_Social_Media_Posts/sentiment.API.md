# API Documentation - sentiment.API.ipynb

Production-ready interface for sentiment predictions.

## Overview

This notebook provides a clean API for making predictions on new airline tweets.

**Key Components:**
- Load pre-trained vectorizer and model
- `SentimentAPI` class with prediction methods
- Examples and use cases

## Loading Models

```python
from joblib import load

vectorizer = load("tfidf_vectorizer.joblib")
model = load("logreg_sentiment_model.joblib")
```

These are created by `sentiment.example.ipynb` and should already exist.

## The API Class

```python
class SentimentAPI:
    def predict(text: str) -> str
    def predict_batch(texts: List[str]) -> List[str]
    def predict_with_confidence(text: str) -> Dict[str, float]
    def predict_batch_with_confidence(texts: List[str]) -> List[Dict]
```

### Method 1: `predict(text)`

Single text prediction.

```python
api.predict("I love this airline!")
# Returns: "positive"
```

**When to use:** Interactive apps, real-time feedback

### Method 2: `predict_batch(texts)`

Multiple texts at once (faster than calling predict() multiple times).

```python
api.predict_batch([
    "Terrible service!",
    "Flight on time.",
    "Great experience!"
])
# Returns: ["negative", "neutral", "positive"]
```

**When to use:** Batch processing, reports, bulk analysis

### Method 3: `predict_with_confidence(text)`

Prediction + probability distribution.

```python
api.predict_with_confidence("Not bad, not great")
# Returns: {
#     "negative": 0.45,
#     "neutral": 0.50,
#     "positive": 0.05
# }
```

**When to use:** Uncertainty quantification, threshold filtering

### Method 4: `predict_batch_with_confidence(texts)`

Batch + probabilities.

```python
api.predict_batch_with_confidence([...])
# Returns: List of Dict[str, float]
```

## Input/Output Specification

### Input
- **Type:** String (single tweet)
- **Length:** Typically 20-280 characters
- **Language:** English
- **Encoding:** UTF-8
- **Handling:** Text is cleaned automatically

Valid inputs:
```
"Great flight!" ✓
"@AirlineX Bad service http://t.co/xyz" ✓ (cleaned)
"AMAZING!!!" ✓ (all caps)
"Flight was okay-ish" ✓
```

Invalid inputs:
```
"" ✗ (empty)
None ✗ (null)
123 ✗ (not string)
```

### Output

**From `predict()`:**
```
String: "negative" | "neutral" | "positive"
```

**From `predict_with_confidence()`:**
```
Dict: {
    "negative": float (0.0-1.0),
    "neutral": float (0.0-1.0),
    "positive": float (0.0-1.0)
}
Values sum to 1.0
```

## Model Details

- **Vectorizer:** TF-IDF
  - 20,000 features
  - Unigrams + bigrams
  - min_df=5

- **Classifier:** Logistic Regression
  - Balanced class weights
  - 1000 iterations
  - Random state: 42

- **Training Data:** 14,640 airline tweets
- **Performance:** ~80% accuracy on test set

## Demonstrations in Notebook

**Demo 1:** Single predictions on example tweets

**Demo 2:** Batch processing multiple feedbacks

**Demo 3:** Confidence scores and probability interpretation

**Demo 4:** Real customer feedback analysis with summary statistics

## Use Cases

### Real-time Feedback Classification
```python
user_tweet = input("Enter tweet: ")
sentiment = api.predict(user_tweet)
print(f"Sentiment: {sentiment}")
```

### Batch Report Generation
```python
daily_tweets = load_from_db()
sentiments = api.predict_batch(daily_tweets)
report = create_report(sentiments)
```

### Confidence-based Filtering
```python
probs = api.predict_with_confidence(text)
if max(probs.values()) < 0.7:
    flag_for_manual_review(text)
```

### Dashboard Integration
```python
feedback_df['sentiment'] = api.predict_batch(feedback_df['text'])
sentiment_counts = feedback_df['sentiment'].value_counts()
# visualize...
```

## Interpreting Confidence Scores

```
0.9+ : Very confident - trust the prediction
0.7-0.9: Reasonably confident - acceptable
0.5-0.7: Low confidence - verify manually
<0.5  : Borderline - likely wrong or ambiguous
```

Example:
```python
probs = api.predict_with_confidence("The flight was okay")
# {"negative": 0.48, "neutral": 0.45, "positive": 0.07}
# Confidence: 0.48 (low) - uncertain which negative or neutral
```

## Limitations

**Works Well:**
- Airline-specific tweets
- English text
- Clear sentiment
- Typical tweet length

**Limitations:**
- Domain-specific (airline tweets)
- English only
- Sarcasm handling (may fail)
- Neutral class harder (21% of training)
- No conversation context

**When to Retrain:**
- New domain needed
- Model accuracy degrading
- New sentiment patterns emerging
- Language changes significantly

## Examples from Notebook

### Example 1: Positive
```
"I absolutely love flying with this airline! Best experience ever!"
→ positive (high confidence)
```

### Example 2: Negative
```
"The flight was cancelled with no explanation. Terrible service."
→ negative (high confidence)
```

### Example 3: Neutral
```
"The flight arrived on time. Standard service."
→ neutral (medium confidence)
```

### Example 4: Mixed/Borderline
```
"Good price but bad service"
→ ? (low confidence - mixed signals)
```

## Performance Notes

- **Inference speed:** ~5ms per prediction
- **Batch efficiency:** ~1.5ms per text (1000+ texts)
- **Memory:** ~30MB for loaded models

## Troubleshooting

**Problem:** "ModuleNotFoundError: No module named 'sentiment_utils'"
- Make sure `sentiment_utils.py` is in same directory

**Problem:** "FileNotFoundError: tfidf_vectorizer.joblib"
- Run `sentiment.example.ipynb` first to create artifacts

**Problem:** All predictions are same class
- Check artifacts loaded correctly
- Verify text isn't empty after cleaning

**Problem:** Confidence always low**
- Text may be outside training domain
- Try with actual airline-related text

## Next Steps

1. Run this notebook to test predictions
2. Integrate into your application
3. Monitor performance in production
4. Retrain with new data periodically

## See Also

- `sentiment.example.md` - Training details
- `sentiment.example.ipynb` - Full training pipeline
- `sentiment_utils.py` - Source code
- `README.md` - Project overview
