# Vertex AI API Documentation

## What This Is About

This document explains how we use Google Cloud Vertex AI for sentiment analysis on social media posts. I've focused on the key APIs we actually use and built some helper functions to make them easier to work with.

## Vertex AI AutoML Natural Language

Vertex AI is Google's machine learning platform, and AutoML Natural Language is their tool for training text classification models without needing to be a machine learning expert.

**Key things we use:**
- **AutoML training**: Point it at your data and it figures out the model
- **Text classification**: Perfect for our sentiment analysis (positive/neutral/negative)
- **REST APIs**: We can call it from Python code

## Core Components We Work With

### 1. Datasets in Vertex AI
When you want to train a model, you first need to create a "dataset" in Vertex AI. Think of it like uploading your training data to Google's cloud.

**What we do:**
- Create datasets from our JSONL files
- Tell Vertex AI which column has the text and which has the labels
- Let it import the data from Google Cloud Storage

### 2. Training Jobs
Once you have a dataset, you create a "training job" that actually builds the model.

**Our approach:**
- Set a training budget
- Let Vertex AI pick the best model architecture

### 3. Model Endpoints
After training, you deploy the model to an "endpoint" so you can make predictions.

**How we use it:**
- Deploy models so we can send new tweets and get sentiment predictions
- Use the REST API to make predictions from our Python code
- Each prediction returns the sentiment label plus confidence scores

## Our Helper Functions

We built some wrapper functions in `vertex_ai_utils.py` to make working with Vertex AI easier. Here are the main ones:

### Data Loading & Exploration

```python
def load_twitter_data(file_path: str) -> pd.DataFrame
```
Loads the Twitter dataset CSV file. Pretty straightforward - just reads the CSV and gives you a pandas DataFrame.

```python
def get_dataset_info(df: pd.DataFrame) -> Dict
```
Analyzes your DataFrame and returns stats like:
- How many rows/columns you have
- Sentiment distribution (how many positive/neutral/negative tweets)
- Which airlines are mentioned
- Memory usage

### Data Visualization

```python
def visualize_sentiment_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> None
```
Makes charts showing how sentiments are distributed. Creates both a bar chart and pie chart.

```python
def visualize_airline_sentiment(df: pd.DataFrame, save_path: Optional[str] = None) -> None
```
Shows sentiment patterns by airline. Helps see which airlines get more negative feedback.

### Text Analysis

```python
def analyze_text_statistics(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame
```
Adds columns to your DataFrame with text stats like:
- How long each tweet is (character count)
- How many words
- Whether it contains URLs, mentions, or hashtags

```python
def print_text_statistics(df: pd.DataFrame, text_column: str = 'text') -> None
```
Prints a nice summary of text statistics, including breakdowns by sentiment type.

### Data Preparation

```python
def split_train_val_test(df: pd.DataFrame, train_ratio: float = 0.7,
                        val_ratio: float = 0.15, test_ratio: float = 0.15,
                        random_state: int = 42, stratify_column: Optional[str] = None)
```
Splits your data into training, validation, and test sets. The `stratify_column` parameter ensures each split has the same proportion of positive/neutral/negative tweets as the original data.

```python
def prepare_data_for_vertex_ai(df: pd.DataFrame, text_column: str = 'text',
                              label_column: str = 'airline_sentiment',
                              output_path: str = 'data/processed/sentiment_data.jsonl') -> str
```
Converts your pandas DataFrame to the JSONL format that Vertex AI expects. Each line looks like:
```json
{"text_content": "This flight was amazing!", "category": "positive"}
```

### Cloud Storage Helpers

```python
def upload_to_gcs(local_file_path: str, bucket_name: str,
                 blob_name: str, project_id: str) -> str
```
Uploads a file from your computer to Google Cloud Storage. Returns the GCS path (like `gs://bucket-name/file.jsonl`).

```python
def create_gcs_bucket(bucket_name: str, project_id: str, location: str = "us-central1") -> None
```
Creates a new Google Cloud Storage bucket for storing your data.

## Why We Built These Wrappers

Working directly with Vertex AI APIs can be complex - you need to handle authentication, error checking, data formats, etc. Our wrappers:

- **Handle authentication**: Set up Google Cloud credentials automatically
- **Add error checking**: Give helpful error messages instead of cryptic API errors
- **Format conversion**: Convert between pandas DataFrames and Vertex AI formats
- **Progress updates**: Print status messages so you know what's happening
- **Resource cleanup**: Close connections and clean up temporary files

## How We Use This in Practice

A typical workflow looks like:

```python
import vertex_ai_utils as vai

# Load and explore data
df = vai.load_twitter_data('data/Tweets.csv')
vai.print_dataset_summary(df)

# Prepare for Vertex AI
train_df, val_df, test_df = vai.split_train_val_test(df, stratify_column='airline_sentiment')
train_jsonl = vai.prepare_data_for_vertex_ai(train_df)

# Upload to Google Cloud
gcs_uri = vai.upload_to_gcs(train_jsonl, 'my-bucket', 'data/train.jsonl', 'my-project')
```

## Design Decisions We Made

### Keeping It Simple
We chose AutoML over custom training because:
- We're learning Vertex AI, not building novel ML architectures
- AutoML handles model selection, hyperparameter tuning, etc.
- Easier to get started and still gets good results

### Error Handling
Added lots of error checking because:
- Cloud APIs can fail in mysterious ways
- File operations can fail if paths are wrong
- Network issues happen
- Better to give clear error messages than cryptic API responses

### Data Format Choices
Used JSONL format because:
- Vertex AI requires it for text classification
- Easy to generate from pandas DataFrames
- Human-readable for debugging
- Efficient for large datasets

## Challenges We Solved

### Authentication Headaches
Setting up Google Cloud authentication was tricky at first. We ended up using service account keys stored as environment variables.

### Data Format Conversion
Getting pandas DataFrames into the right JSONL format took some trial and error. The format needs to be exactly right or Vertex AI rejects it.

### Cost Management
Vertex AI charges by the hour for training. We learned to:
- Start with small datasets for testing
- Set reasonable training budgets
- Monitor costs in the Google Cloud console

## What We Learned

### Vertex AI Strengths
- Really easy to get started with AutoML
- Good documentation and examples
- Scales well for production use
- Integrates nicely with other Google Cloud services

### Areas That Could Be Better
- Error messages could be clearer sometimes
- Pricing can be confusing (training hours vs. prediction costs)
- The web interface is good but the APIs feel a bit clunky

### Tips for Others
- Start with the web interface to understand the concepts, then move to APIs
- Always test with small datasets first to avoid wasting money
- Use the `gcloud` command-line tool for debugging authentication issues
- Keep track of your resource usage to avoid surprise bills
