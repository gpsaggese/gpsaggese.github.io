# Complete Sentiment Analysis Example with Vertex AI

## What This Example Does

This notebook shows how to build a complete sentiment analysis system for social media posts using Google Cloud Vertex AI. We take raw Twitter data about airline experiences and train a model that can automatically classify new tweets as positive, neutral, or negative.

## The Big Picture

Here's what we're building:

1. **Data Pipeline**: Load Twitter data → explore it → clean it → prepare it for ML
2. **Model Training**: Use Vertex AI AutoML to train a sentiment classifier
3. **Model Deployment**: Deploy the trained model so it can make predictions
4. **Predictions**: Send new tweets to get sentiment predictions

## Why This Matters

Social media sentiment analysis is useful for:
- **Airlines**: Understanding customer satisfaction in real-time
- **Brands**: Monitoring public perception of products/services
- **Marketing**: Measuring campaign effectiveness
- **Customer Service**: Identifying unhappy customers quickly

## Our Dataset

We're using the **Twitter US Airline Sentiment** dataset, which contains:
- 14,640+ tweets about major US airlines
- Each tweet labeled as positive, neutral, or negative
- Additional metadata like airline names, tweet timestamps, etc.

The dataset shows a **class imbalance** - there are way more negative tweets than positive ones, which is actually realistic for customer feedback data.

## Step-by-Step Implementation

### Step 1: Setting Up the Environment

First, we configure our Google Cloud settings:

```python
PROJECT_ID = "noted-cortex-477800-b7"
BUCKET_NAME = "vertex-ai-sentiment-data-msml610"
LOCATION = "us-central1"
```

We need:
- A Google Cloud project with Vertex AI enabled
- A Cloud Storage bucket for our data
- Proper authentication (service account key or gcloud auth)

### Step 2: Loading and Exploring Data

We start by loading the raw Twitter data and getting a feel for what we're working with:

```python
import vertex_ai_utils as vai

# Load the data
df = vai.load_twitter_data("data/Tweets.csv")
print(f"Loaded {len(df)} tweets")
```

Then we explore the data to understand:
- How many tweets we have
- What airlines are mentioned
- Sentiment distribution
- Text characteristics

### Step 3: Data Visualization

We create charts to understand our data better:

```python
# See overall sentiment breakdown
vai.visualize_sentiment_distribution(df)

# See which airlines get the most complaints
vai.visualize_airline_sentiment(df)
```

This helps us see that some airlines get way more negative feedback than others, which makes sense - bigger airlines have more customers and therefore more complaints.

### Step 4: Text Analysis

We analyze the text content to understand patterns:

```python
# Get statistics about tweet lengths, word counts, etc.
vai.print_text_statistics(df)

# Look at sample tweets from each sentiment category
vai.sample_tweets_by_sentiment(df, n_samples=3)
```

We learn that:
- Negative tweets tend to be longer (people complain more verbosely)
- Many tweets contain mentions (@airline) and hashtags
- URLs are less common but present

### Step 5: Data Preparation

Before training, we need to prepare our data properly:

```python
# Split into train/validation/test sets
train_df, val_df, test_df = vai.split_train_val_test(
    df,
    train_ratio=0.7,    # 70% for training
    val_ratio=0.15,     # 15% for validation
    test_ratio=0.15,    # 15% for testing
    stratify_column='airline_sentiment'  # Keep same sentiment ratios
)

# Convert to Vertex AI format
train_jsonl = vai.prepare_data_for_vertex_ai(train_df)
val_jsonl = vai.prepare_data_for_vertex_ai(val_df)
test_jsonl = vai.prepare_data_for_vertex_ai(test_df)
```

The `stratify_column` parameter is important here - it ensures that each of our train/val/test splits has the same proportion of positive/neutral/negative tweets as the original dataset.

### Step 6: Uploading to Google Cloud

Vertex AI needs our data in Google Cloud Storage:

```python
# Upload training data
train_gcs_uri = vai.upload_to_gcs(
    train_jsonl,
    BUCKET_NAME,
    'data/train_data.jsonl',
    PROJECT_ID
)

# Upload validation data
val_gcs_uri = vai.upload_to_gcs(
    val_jsonl,
    BUCKET_NAME,
    'data/val_data.jsonl',
    PROJECT_ID
)
```

### Step 7: Model Training

Here's where we'd train the actual model:

```python
# This is what the training code would look like
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Create dataset
dataset = aiplatform.TextDataset.create(
    display_name='airline_sentiment_dataset',
    gcs_source=[train_gcs_uri],
    import_schema_uri=aiplatform.schema.dataset.ioformat.text.classification_single_label,
)

# Train model
job = aiplatform.AutoMLTextTrainingJob(
    display_name='airline_sentiment_training',
    prediction_type='classification',
    multi_label=False
)

model = job.run(
    dataset=dataset,
    training_fraction_split=0.8,
    validation_fraction_split=0.2,
    model_display_name='airline_sentiment_model'
)
```

### Step 8: Model Deployment

After training, we'd deploy the model:

```python
# Deploy to endpoint
endpoint = model.deploy(
    deployed_model_display_name='airline_sentiment_endpoint',
    machine_type='n1-standard-4'
)
```

### Step 9: Making Predictions

Finally, we'd use the deployed model to classify new tweets:

```python
# Example prediction
test_tweet = "Just had an amazing flight with @AmericanAir! Great service and comfortable seats!"

prediction = endpoint.predict([test_tweet])
print(f"Sentiment: {prediction.predictions[0]['classes'][0]}")
print(f"Confidence: {prediction.predictions[0]['scores'][0]:.2f}")
```

## Implementation Status

**Completed ✅:**
- Data loading and exploration
- Visualization and analysis
- Data splitting and preparation
- JSONL format conversion
- **Model Training**: Twitter-RoBERTa fine-tuning
- **Hyperparameter Tuning**: Learning rate, batch size, warmup optimization
- **Model Evaluation**: F1-score, confusion matrix, classification report
- **BERT Baseline**: Comparison with BERT model
- **Dashboard**: Sentiment trends visualization

## Key Design Decisions

### Why AutoML Instead of Custom Training?

We chose Vertex AI AutoML because:
- AutoML handles model selection, hyperparameter tuning, etc. automatically
- Still gets good results without ML expertise
- Faster to implement for a class project

### Data Splitting Strategy

We use stratified splitting to maintain sentiment class proportions:
- **Training set (70%)**: Used to train the model
- **Validation set (15%)**: Used during training to tune hyperparameters
- **Test set (15%)**: Used after training to evaluate final performance

This ensures our model evaluation isn't biased by uneven class distributions.

### JSONL Format Choice

Vertex AI requires data in JSONL format for text classification:
```json
{"text_content": "Flight was delayed again!", "category": "negative"}
{"text_content": "Great service from the crew", "category": "positive"}
```

We chose this because:
- Required by Vertex AI
- Easy to generate from pandas DataFrames
- Human-readable for debugging
- Efficient for large datasets

## Challenges We Faced

### Class Imbalance
The dataset has way more negative tweets than positive ones. This is actually realistic for customer feedback, but it means we need to:
- Use stratified splitting to maintain proportions
- Consider evaluation metrics that account for imbalance (F1-score vs. accuracy)
- Maybe use techniques like oversampling if needed

### Text Preprocessing
Twitter data is messy:
- @mentions and #hashtags everywhere
- URLs and emojis
- Inconsistent capitalization
- Slang and abbreviations

We handle this by cleaning the text before training, but keep enough context for sentiment analysis.

### Cost Management
Vertex AI charges by the training hour, so we learned to:
- Test with small datasets first
- Set reasonable training budgets
- Monitor usage in Google Cloud console
- Use the free tier as much as possible

## Files Overview

Our complete implementation includes:

- `vertex_ai_utils.py`: Helper functions for data processing and Vertex AI operations
- `vertex_ai.API.ipynb`: Demonstrates API usage and data exploration
- `vertex_ai.example.ipynb`: Complete end-to-end pipeline (in progress)
- `vertex_ai.API.md`: API documentation (this file)
- `vertex_ai.example.md`: Application documentation (you are here)

This gives anyone the complete toolkit to build their own sentiment analysis system using Vertex AI.
