# Training & Evaluation - sentiment.example.ipynb

This notebook trains a sentiment classifier on airline tweets.

## What It Does

1. **Load Data** - 14,640 tweets from CSV
2. **EDA** - YData profiling for exploratory analysis
3. **Clean Text** - Remove URLs, @mentions, special chars
4. **Preprocess** - Create numeric labels (0/1/2)
5. **Split** - Train (70%), Val (10%), Test (20%) with stratification
6. **Train** - TF-IDF vectorizer + Logistic Regression
7. **Evaluate** - Accuracy, precision, recall, F1
8. **Analyze** - Error analysis, feature importance
9. **Save** - Model and vectorizer artifacts

## Key Sections

### Data Loading & EDA
- Loads CSV from `data/Tweets.csv`
- YData profiling generates HTML report
- Shows class distribution (problem: negative 3.9x more common)

### Text Preprocessing
Cleans text in these steps:
- Lowercase
- Remove URLs and @mentions
- Remove special chars (keep letters + spaces)
- Normalize whitespace

Example:
```
Raw:    "@AirlineX #awful!!!  http://t.co/xyz service sucks!! 😠"
Clean:  "awful service sucks"
```

### Train/Val/Test Split
Uses stratified sampling so each split maintains class distribution.
- Train: ~70% (for learning)
- Val: ~10% (for monitoring)
- Test: ~20% (for final evaluation)

### Model Training
**TF-IDF Configuration:**
- 20,000 features (vocabulary)
- Unigrams + bigrams (single words + two-word phrases)
- min_df=5 (ignore rare words)

**Logistic Regression:**
- class_weight="balanced" - handles class imbalance
- max_iter=1000
- random_state=42 for reproducibility

### Evaluation
Reports on both validation and test sets:
- **Accuracy** - overall correctness
- **Precision** - of predicted positives, how many correct?
- **Recall** - of actual positives, how many did we find?
- **F1** - balance of precision and recall
- **Confusion Matrix** - breakdown of predictions
- **Per-class report** - metrics for each label

### Error Analysis
Shows which predictions are wrong:
- Neutral class most difficult (21% of data, confusing features)
- Some tweets have mixed sentiment
- Short texts have less context

### Feature Importance
Top words per class using model coefficients:
- **Negative:** bad, worst, terrible, delay, cancel, lost
- **Neutral:** flight, airline, time, service, was, seat
- **Positive:** great, love, best, excellent, comfortable

## Important Design Decisions

### Why Balanced Class Weights?
Dataset has 3.9:1 imbalance (9178 negative vs 2363 positive).
Without balancing, model biased toward negative.
Solution: penalize misclassifying minority classes more.

### Why Stratified Sampling?
Ensures each split (train/val/test) has same class percentages.
Example: if training data is 63% negative, validation should also be 63% negative.

### Why TF-IDF + LogisticRegression?
- Simple and interpretable
- Fast training and inference
- Works well with text features
- Good baseline performance
- Could upgrade to BERT later if needed

## Results on Test Set

```
Accuracy:  ~0.80
Precision: ~0.80
Recall:    ~0.78
F1-Score:  ~0.78
```

Per-class breakdown:
- Negative: highest precision (0.83)
- Positive: highest recall (0.84)
- Neutral: most difficult (0.75 precision)

## Output Artifacts

This notebook creates:
1. `tfidf_vectorizer.joblib` - vectorizer for transforming text
2. `logreg_sentiment_model.joblib` - trained classifier
3. `airline_sentiment_profile.html` - detailed EDA report

These are loaded by `sentiment.API.ipynb` for predictions.

## Limitations

- **Imbalanced data:** Negative tweets dominate
- **Neutral class:** Only 21% of data, harder to learn
- **Domain:** Trained on airline tweets only
- **Sarcasm:** May misclassify "Great, another delay"
- **Spelling:** Trained data has typos, sensitive to exact wording

## Next Steps

1. Run this notebook to train the model
2. Open `sentiment.API.ipynb` to make predictions
3. Use the saved artifacts in production

## Troubleshooting

**Model won't train?**
- Check data loads correctly
- Verify `clean_text` column exists
- Check sufficient memory

**Accuracy seems low?**
- Class imbalance is challenging
- Neutral class inherently difficult
- Check text cleaning isn't too aggressive

**Artifacts not saving?**
- Check write permissions
- Verify joblib installed
- Try absolute path instead of relative
