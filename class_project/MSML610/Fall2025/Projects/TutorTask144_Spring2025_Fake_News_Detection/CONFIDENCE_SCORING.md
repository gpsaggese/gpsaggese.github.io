# Confidence Scoring and Uncertainty Quantification

## Overview

Confidence scores indicate how certain the model is about each prediction. Instead of just "Fake" or "Real", the model can now say:

- "Fake (95% confidence)" - High certainty
- "Real (72% confidence)" - Moderate certainty
- "Fake (51% confidence)" - Low certainty, may need human review

## Why Confidence Matters

### Use Cases
1. **Content Moderation:** Flag for human review when confidence < 70%
2. **Ranking:** Rank news articles by confidence for fact-checkers
3. **System Reliability:** Estimate system accuracy on high-confidence predictions
4. **Active Learning:** Select uncertain samples for annotation

### Benefits
- **Better decision-making:** Use high-confidence predictions, ask for help on uncertain ones
- **Cost reduction:** Only human-review uncertain cases
- **System improvement:** Identify where model needs improvement
- **User trust:** Users see uncertainty, not false certainty

## Methods

### 1. Probability Method (Default)

**What it does:** Uses the predicted probability as confidence score

```
Confidence = max(P(Fake), P(Real))
```

**Advantages:**
- Simple, fast
- Works immediately after training
- Interpretable as actual probability

**Disadvantages:**
- Models may be overconfident or underconfident
- Requires calibration for reliability

**Example:**
```
Model output: P(Fake) = 0.75, P(Real) = 0.25
Confidence: 0.75 (75%)
```

### 2. Entropy Method

**What it does:** Measures information entropy across predicted probabilities

```
Entropy = -Σ(P_i * log(P_i))
Confidence = 1 - (Entropy / log(2))  # Normalized to [0, 1]
```

Lower entropy = higher confidence (model is decisive)

**Advantages:**
- Captures decisiveness
- 0.5 probabilities (most uncertain) = 0% confidence
- Natural interpretation

**Disadvantages:**
- Less common than probability
- Slower computation

**Example:**
```
Model output: P(Fake) = 0.75, P(Real) = 0.25
Entropy = 0.56 bits (max = 1.0)
Confidence = 0.44 (44%)  # Lower than probability method
```

### 3. Margin Method

**What it does:** Computes gap between top 2 predictions

```
Margin = max(P) - second_max(P)
Confidence = Margin
```

Larger margin = higher confidence

**Advantages:**
- Simple, interpretable
- Robust to calibration issues
- Works well for tight predictions

**Disadvantages:**
- Doesn't consider absolute probability
- Can be misleading (e.g., 0.51 vs 0.49 has high margin, both uncertain)

**Example:**
```
Model output: P(Fake) = 0.75, P(Real) = 0.25
Margin: 0.50
Confidence: 0.50 (50%)
```

### 4. Bayesian Uncertainty (MC Dropout)

**What it does:** Performs multiple forward passes with dropout enabled to sample from posterior distribution

```
For i in 1 to N:
    Output_i = model_with_dropout(input)

Mean = average(Output_1, ..., Output_N)
Std = std_dev(Output_1, ..., Output_N)
Confidence = 1 - (Entropy(Mean) + Uncertainty(Std))
```

**Advantages:**
- Theoretically sound (Bayesian approximation)
- Estimates aleatoric + epistemic uncertainty
- Most reliable confidence scores

**Disadvantages:**
- Slower (10+ forward passes per sample)
- Requires retraining with dropout
- More computationally expensive

**Example:**
```
10 forward passes give:
Mean P(Fake) = 0.72 ± 0.08
Confidence = 0.88 (88%)  # Includes uncertainty about prediction
```

## Implementation

### Basic Usage

```python
from confidence_utils import ConfidenceScorer
from bert_utils import BertModelWrapper

# Load model
model = BertModelWrapper(config)

# Create scorer
scorer = ConfidenceScorer(model=model, device='cuda')

# Get predictions with confidence
results = scorer.predict_with_confidence(
    texts=["News article 1", "News article 2"],
    method='probability',  # or 'entropy', 'margin'
    threshold=0.5
)

# Results
print(results['predictions'])  # [0, 1]
print(results['probabilities'])  # [0.72, 0.81]
print(results['confidence'])  # [0.72, 0.81]
```

### Different Methods

```python
# Probability-based (default)
results = scorer.predict_with_confidence(
    texts=texts,
    method='probability'
)

# Entropy-based (captures decisiveness)
results = scorer.predict_with_confidence(
    texts=texts,
    method='entropy'
)

# Margin-based (gap between top 2)
results = scorer.predict_with_confidence(
    texts=texts,
    method='margin'
)
```

### Bayesian Uncertainty

```python
# MC Dropout approach (slowest, most accurate)
results = scorer.predict_with_uncertainty(
    tokenizer=model.tokenizer,
    texts=texts,
    num_samples=10,
    threshold=0.5
)

# Results include uncertainty
print(results['mean_probabilities'])  # [0.72, 0.81]
print(results['std_probabilities'])   # [0.08, 0.06]
print(results['entropy'])              # [0.56, 0.42]
print(results['uncertain'])           # [False, False] (low entropy = not uncertain)
```

### Confidence-Based Filtering

```python
from confidence_utils import ConfidenceThresholdAnalyzer

# Analyze predictions at different confidence levels
analyzer = ConfidenceThresholdAnalyzer()
analysis = analyzer.analyze(
    predictions=predictions,
    confidences=confidences,
    labels=labels,
    thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]
)

for threshold, metrics in analysis.items():
    print(f"Threshold {threshold}:")
    print(f"  Coverage: {metrics['coverage']:.1%}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
```

## Calibration

Models may be **overconfident** (too sure) or **underconfident** (too unsure).

### Measuring Calibration

```python
from confidence_utils import CalibrationMetrics

# Expected Calibration Error
ece = CalibrationMetrics.expected_calibration_error(
    predictions, probabilities, labels
)
print(f"ECE: {ece:.4f}")  # Lower is better (0 = perfect)

# Maximum Calibration Error
mce = CalibrationMetrics.maximum_calibration_error(
    predictions, probabilities, labels
)
print(f"MCE: {mce:.4f}")  # Max difference across bins

# Brier Score (MSE of probabilities)
brier = CalibrationMetrics.brier_score(probabilities, labels)
print(f"Brier: {brier:.4f}")  # Lower is better
```

### Interpreting Calibration

| ECE | Interpretation |
|-----|---|
| 0.0-0.05 | Excellent, well-calibrated |
| 0.05-0.10 | Good, slightly overconfident |
| 0.10-0.20 | Moderate, needs adjustment |
| >0.20 | Poor, significantly miscalibrated |

### Fixing Overconfidence

**Temperature Scaling (post-hoc calibration):**

```python
# Scale probabilities without retraining
# Higher temperature = lower confidence
T = 1.5  # Temperature parameter

calibrated_probs = torch.softmax(logits / T, dim=1)
```

## Practical Applications

### Example 1: Content Moderation

```python
scorer = ConfidenceScorer(model)

for article in articles:
    pred, conf = scorer.predict_with_confidence([article['text']])

    if conf[0] >= 0.8:
        # High confidence - auto-label
        article['label'] = pred[0]
        article['reviewer'] = 'automated'
    elif conf[0] >= 0.6:
        # Medium confidence - priority review
        article['label'] = pred[0]
        article['reviewer'] = 'priority'
    else:
        # Low confidence - needs expert review
        article['label'] = '?'
        article['reviewer'] = 'expert'
```

### Example 2: Model Evaluation

```python
# Only trust high-confidence predictions
high_conf_mask = np.array(confidences) >= 0.8
high_conf_accuracy = accuracy_score(
    labels[high_conf_mask],
    predictions[high_conf_mask]
)

print(f"Accuracy on high-confidence predictions: {high_conf_accuracy:.1%}")
print(f"Coverage: {np.sum(high_conf_mask) / len(labels):.1%}")
```

### Example 3: Ranking Predictions

```python
# Sort by confidence for fact-checker prioritization
ranked = sorted(
    zip(texts, predictions, confidences),
    key=lambda x: x[2],
    reverse=True  # Highest confidence first
)

for text, pred, conf in ranked[:10]:
    print(f"{conf:.1%} - {pred}: {text[:50]}...")
```

## Best Practices

### 1. Choose Right Method
- **Probability:** Fast, use immediately
- **Entropy:** Better calibration, simple
- **Bayesian:** Most accurate, slow
- **Margin:** Best for tight predictions

### 2. Set Thresholds Appropriately
- **High confidence (>0.8):** Auto-decision making
- **Medium confidence (0.6-0.8):** Human review
- **Low confidence (<0.6):** Expert decision

### 3. Monitor Calibration
```python
# Periodically check if model is still calibrated
ece = CalibrationMetrics.expected_calibration_error(...)
if ece > 0.15:
    # Retrain with recent data or apply temperature scaling
```

### 4. Validate on Test Set
- Don't use training set to calibrate (information leak)
- Use held-out validation set
- Retrain if significant distribution shift

## Performance Comparison

### Speed

| Method | Speed | Relative |
|--------|-------|----------|
| Probability | Baseline | 1.0x |
| Entropy | Baseline | 1.0x |
| Margin | Baseline | 1.0x |
| Bayesian (10 samples) | 10x slower | 10x |

### Accuracy (ECE)

| Method | Calibration | Ranking |
|--------|---|---|
| Probability | 0.08-0.12 | 2nd |
| Entropy | 0.06-0.10 | 3rd |
| Margin | 0.07-0.11 | 2nd |
| Bayesian | 0.03-0.06 | 1st |

## Files

- `confidence_utils.py` - Core confidence scoring implementation
- `CONFIDENCE_SCORING.md` - This document

## Next Steps

1. Add confidence scores to your predictions
2. Measure calibration on validation set
3. Determine optimal confidence thresholds for your use case
4. Move to Task 7 (FastAPI deployment)

## References

- Guo et al. (2017): [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- Lakshminarayanan et al. (2017): [Simple and Scalable Predictive Uncertainty Estimation](https://arxiv.org/abs/1506.02142)
- Uncertainty in Deep Learning: [Gal & Ghahramani (2016)](https://arxiv.org/abs/1506.02142)
