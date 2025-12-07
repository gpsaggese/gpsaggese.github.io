# Vertex AI API Documentation

## Overview

This document provides comprehensive documentation for using Google Cloud Vertex AI for sentiment analysis on social media posts. It is structured according to the course submission guidelines, clearly separating the **native Vertex AI SDK** from our **wrapper layer functions**.

---

## Section 1: Native Vertex AI SDK

### What is Vertex AI?

Vertex AI is Google's unified machine learning platform that provides:

- **AutoML**: Automated model training without ML expertise
- **Custom Training**: GPU-accelerated training with custom scripts
- **Model Deployment**: Production-ready endpoints with auto-scaling
- **MLOps**: End-to-end ML pipeline management

### Core SDK Components

#### 1. Authentication (`google.auth`)
```python
from google.auth import default
credentials, project = default()
```

#### 2. Vertex AI Initialization (`google.cloud.aiplatform`)
```python
import google.cloud.aiplatform as aiplatform

aiplatform.init(
    project="your-project-id",
    location="us-central1",
    credentials=credentials
)
```

#### 3. Dataset Management
```python
# Create text dataset
dataset = aiplatform.TextDataset.create(
    display_name="sentiment-dataset",
    gcs_source=["gs://bucket/data.jsonl"],
    import_schema_uri=aiplatform.schema.dataset.ioformat.text.single_label_classification
)
```

#### 4. Custom Training Jobs
```python
# Custom training job
job = aiplatform.CustomTrainingJob(
    display_name="sentiment-training",
    script_path="vertex_ai_training.py",
    container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-8:latest",
    requirements=["transformers", "datasets", "torch"],
    model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-8:latest"
)

# Run training
model = job.run(
    dataset=dataset,
    model_display_name="sentiment-model",
    args=["--epochs", "3", "--batch_size", "16"]
)
```

#### 5. Model Deployment
```python
# Deploy to endpoint
endpoint = model.deploy(
    deployed_model_display_name="sentiment-endpoint",
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3
)
```

#### 6. Predictions
```python
# Make predictions
predictions = endpoint.predict([
    {"content": "Great flight experience!"},
    {"content": "Terrible customer service"}
])

for pred in predictions.predictions:
    print(f"Sentiment: {pred['displayNames'][0]}")
    print(f"Confidence: {pred['confidences'][0]}")
```

### Advanced Features

#### Hyperparameter Tuning
```python
# Hyperparameter tuning job
tuning_job = aiplatform.HyperparameterTuningJob(
    display_name="sentiment-hp-tuning",
    custom_job=job,
    metric_spec={"accuracy": "maximize"},
    parameter_spec={
        "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(min=1e-5, max=1e-3, scale="log"),
        "batch_size": aiplatform.hyperparameter_tuning.DiscreteParameterSpec(values=[8, 16, 32])
    },
    max_trial_count=20
)
```

#### Model Evaluation
```python
# Get evaluation metrics
evaluations = model.list_model_evaluations()
metrics = evaluations[0].metrics

print("Model Performance:")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value}")
```

---

## Section 2: Wrapper Functions (`vertex_ai_utils.py`)

### Design Philosophy

Our wrapper functions provide:
- **Simplified API**: Hide complexity of Vertex AI SDK
- **Error Handling**: Clear error messages and recovery
- **Cost Management**: Commented expensive operations
- **Educational Value**: Show Vertex AI best practices

### Data Processing Wrappers

#### Data Loading & Exploration
```python
def load_twitter_data(file_path: str) -> pd.DataFrame
```
**Purpose**: Load Twitter sentiment dataset CSV
**Parameters**: file_path (str) - Path to CSV file
**Returns**: pandas DataFrame with tweet data
**Usage**: `df = load_twitter_data("Data/Tweets.csv")`

#### Data Visualization
```python
def visualize_sentiment_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> None
```
**Purpose**: Create sentiment distribution charts
**Parameters**:
- df: DataFrame with sentiment data
- save_path: Optional path to save chart
**Returns**: None (displays/saves chart)

#### Data Preparation
```python
def prepare_data_for_vertex_ai(df: pd.DataFrame, text_column: str = 'text',
                              label_column: str = 'airline_sentiment',
                              output_path: str = 'data/processed/sentiment_data.jsonl') -> str
```
**Purpose**: Convert DataFrame to Vertex AI JSONL format
**Parameters**:
- df: Input DataFrame
- text_column: Column containing text data
- label_column: Column containing labels
- output_path: Path for JSONL output
**Returns**: Path to created JSONL file

### Cloud Integration Wrappers

#### Vertex AI Initialization
```python
def initialize_vertex_ai(project_id: str, location: str,
                        credentials_path: str = 'vertex-ai-key.json') -> None
```
**Purpose**: Set up Vertex AI connection and authentication
**Parameters**:
- project_id: Google Cloud Project ID
- location: GCP region (e.g., 'us-central1')
- credentials_path: Path to service account key
**Returns**: None

#### Cloud Storage Operations
```python
def upload_to_gcs(bucket_name: str, source_file_path: str,
                 destination_blob_name: str, project_id: str) -> str
```
**Purpose**: Upload files to Google Cloud Storage
**Parameters**:
- bucket_name: GCS bucket name
- source_file_path: Local file path
- destination_blob_name: Destination path in bucket
- project_id: GCP project ID
**Returns**: GCS URI (gs://bucket/path)

### Model Training Wrappers

#### Custom Training Job Creation
```python
def create_custom_roberta_training_job(project_id: str, location: str,
                                      job_name: str, model_display_name: str,
                                      train_data_uri: str, val_data_uri: str) -> aiplatform.CustomTrainingJob
```
**Purpose**: Create Vertex AI custom training job for RoBERTa
**Parameters**:
- project_id, location: GCP settings
- job_name: Unique job identifier
- model_display_name: Name for trained model
- train_data_uri, val_data_uri: GCS paths to data
**Returns**: CustomTrainingJob object

#### Model Deployment
```python
def deploy_model_to_endpoint(model: aiplatform.Model, endpoint_display_name: str,
                           machine_type: str = "n1-standard-2",
                           min_replica_count: int = 1, max_replica_count: int = 1) -> aiplatform.Endpoint
```
**Purpose**: Deploy trained model to Vertex AI endpoint
**Parameters**:
- model: Trained Model object
- endpoint_display_name: Name for endpoint
- machine_type: VM type for serving
- min/max_replica_count: Auto-scaling settings
**Returns**: Endpoint object

### Prediction & Evaluation Wrappers

#### Batch Predictions
```python
def predict_with_vertex_ai_endpoint(endpoint: aiplatform.Endpoint,
                                   text_instances: List[str]) -> List[Dict]
```
**Purpose**: Make sentiment predictions using deployed endpoint
**Parameters**:
- endpoint: Deployed Endpoint object
- text_instances: List of text strings to classify
**Returns**: List of prediction results with sentiment and confidence

#### Model Evaluation
```python
def evaluate_vertex_ai_model_predictions(endpoint: aiplatform.Endpoint,
                                       test_texts: List[str],
                                       true_labels: List[str]) -> Dict
```
**Purpose**: Evaluate model performance with F1-score and confusion matrix
**Parameters**:
- endpoint: Deployed model endpoint
- test_texts: Test text samples
- true_labels: Ground truth labels
**Returns**: Dictionary with evaluation metrics

### Resource Management Wrappers

#### Cleanup Functions
```python
def cleanup_vertex_ai_resources(endpoints: Optional[List[aiplatform.Endpoint]] = None,
                              models: Optional[List[aiplatform.Model]] = None,
                              datasets: Optional[List[aiplatform.Dataset]] = None) -> None
```
**Purpose**: Clean up Vertex AI resources to prevent charges
**Parameters**:
- endpoints, models, datasets: Lists of resources to delete
**Returns**: None (prints cleanup status)

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
