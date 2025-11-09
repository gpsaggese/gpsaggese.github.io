# Fake News Detection with Model Context Protocol (MCP)

## Project Overview

This project implements a **Fake News Detection system** using the **Model Context Protocol (MCP)**, a framework for managing machine learning models in context-aware scenarios.

### Key Features

- **Data Preprocessing**: Text cleaning, tokenization, stopword removal, and stemming
- **Feature Extraction**: TF-IDF vectorization for text representation
- **Multiple Models**: Logistic Regression, Random Forest, Gradient Boosting classifiers
- **Model Context Management**: Track model metadata and validate deployment context
- **Comprehensive Evaluation**: ROC-AUC, Precision, Recall, F1-score metrics
- **Cross-Validation**: K-fold validation for robust performance estimation

### Project Structure

```
TutorTask144_Spring2025_Fake_News_Detection/
├── utils_data_preprocessing.py       # Text processing and feature extraction
├── utils_model_training.py           # Model training with MCP context management
├── MCP.API.md                        # Model Context Protocol documentation
├── MCP.API.ipynb                     # MCP tutorial notebook (to be created)
├── FakeNewsDetection.example.md      # Fake news detection documentation (to be created)
├── FakeNewsDetection.example.ipynb   # Complete implementation notebook (to be created)
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker container setup
├── docker_build.sh                   # Script to build Docker image
├── docker_jupyter.sh                 # Script to run Jupyter in container
├── README.md                         # This file
└── IMPLEMENTATION_GUIDE.md           # Detailed implementation instructions
```

## Getting Started

### Prerequisites

- Docker installed on your machine
- Kaggle Fake News Detection dataset (download from https://www.kaggle.com/c/fake-news/data)
- Basic understanding of NLP and ML concepts

### Quick Start with Docker

1. **Navigate to project directory**
   ```bash
   cd class_project/MSML610/Fall2025/Projects/TutorTask144_Spring2025_Fake_News_Detection
   ```

2. **Build the Docker image**
   ```bash
   ./docker_build.sh
   ```

   Expected output:
   ```
   Building Docker image for TutorTask144_Spring2025_Fake_News_Detection...
   Successfully built [image_hash]
   ```

3. **Start Jupyter server**
   ```bash
   ./docker_jupyter.sh
   ```

   Expected output:
   ```
   Starting Jupyter server...
   Jupyter is running at http://localhost:8888
   Token: [your_token_here]
   ```

4. **Open in browser**
   - Go to [http://localhost:8888](http://localhost:8888)
   - Enter the token if prompted
   - Navigate to the project notebooks

## Project Deliverables

### 1. MCP.API.md
Comprehensive documentation of the Model Context Protocol:
- Core concepts: Model Context, Context Manager, Compatibility Checking
- API architecture with code examples
- Design decisions and rationale
- Workflow examples

### 2. MCP.API.ipynb
Interactive tutorial demonstrating MCP:
- Creating `ModelContext` instances
- Using `ModelContextManager` for registration
- Context compatibility validation
- Saving and loading contexts

### 3. FakeNewsDetection.example.md
Complete guide to fake news detection:
- Dataset overview and statistics
- Data preprocessing pipeline
- Feature extraction with TF-IDF
- Model selection and training process
- Evaluation metrics and results
- Integration with MCP

### 4. FakeNewsDetection.example.ipynb
Full implementation showing:
- Loading and exploring the dataset
- Text preprocessing pipeline
- Feature extraction
- Training multiple models
- Model comparison and evaluation
- Cross-validation analysis
- Using MCP for context management
- Visualizations (ROC curves, confusion matrices)

## Dataset Information

**Kaggle Fake News Detection Dataset**
- Download: https://www.kaggle.com/c/fake-news/data
- Total samples: ~20,000 news articles
- Features: id, title, author, text, date, label
- Target: Binary (0 = Fake, 1 = Real)
- Relatively balanced class distribution

## Data Preprocessing Pipeline

The preprocessing pipeline includes:

1. **Text Cleaning**
   - Remove URLs and email addresses
   - Convert to lowercase
   - Remove special characters
   - Normalize whitespace

2. **Tokenization**
   - Split text into individual tokens
   - Use NLTK word tokenizer

3. **Stopword Removal**
   - Filter common English stopwords
   - Remove short tokens (< 3 characters)

4. **Stemming/Lemmatization**
   - PorterStemmer for word stemming
   - WordNetLemmatizer for lemmatization

5. **Feature Extraction**
   - TF-IDF vectorization
   - Configurable max features (default: 5000)
   - Unigrams and bigrams

## Model Training and Evaluation

### Supported Models

1. **Logistic Regression**
   - Fast, interpretable baseline
   - Default: max_iter=1000

2. **Random Forest**
   - Ensemble method with feature importance
   - Default: n_estimators=100, max_depth=20

3. **Gradient Boosting**
   - Sequential tree ensemble
   - Default: n_estimators=100, learning_rate=0.1

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: False positive rate control
- **Recall**: Fake news detection rate
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall classifier performance
- **Confusion Matrix**: Per-class performance

## Model Context Protocol (MCP)

MCP is a framework for managing ML models in context-aware scenarios:

### Key Concepts

1. **ModelContext**: Dataclass containing:
   - Model identity and type
   - Feature type and preprocessing state
   - Training/validation/test set sizes
   - Hyperparameters
   - Performance metrics

2. **ModelContextManager**: Registry for model contexts
   - Register and retrieve contexts
   - Validate context compatibility
   - Support for context evolution

3. **Context Compatibility**: Ensures models are used appropriately
   - Feature type must match
   - Preprocessing state must match
   - Prevents errors from context mismatch

### Example Usage

```python
from utils_model_training import ModelContextManager, ModelContext

# Create context
context = ModelContext(
    model_id='news_detector_v1',
    model_name='Fake News Detector',
    model_type='random_forest',
    feature_type='tfidf',
    created_at='2025-01-20',
    training_samples=15000,
    validation_samples=2000,
    test_samples=3000,
    preprocessed=True,
    hyperparameters={'n_estimators': 100}
)

# Manage context
manager = ModelContextManager()
manager.register_context(context)

# Validate before inference
if manager.is_context_compatible('news_detector_v1', 'tfidf', True):
    predictions = model.predict(features)
```

## Usage Examples

### Basic Preprocessing

```python
from utils_data_preprocessing import preprocess_text, extract_tfidf_features

# Preprocess a single text
cleaned = preprocess_text("This is a news article...")

# Extract TF-IDF features from multiple texts
features, vectorizer = extract_tfidf_features(texts_list)
```

### Model Training

```python
from utils_model_training import create_model, train_model, evaluate_model

# Create and train
model = create_model('random_forest', {'n_estimators': 100})
train_model(model, X_train, y_train)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### Model Comparison

```python
from utils_model_training import compare_models

results = compare_models(
    {
        'logistic_regression': {'model_type': 'logistic_regression'},
        'random_forest': {'model_type': 'random_forest'},
        'gradient_boosting': {'model_type': 'gradient_boosting'}
    },
    X_train, y_train, X_test, y_test
)
print(results)  # DataFrame with all model results
```

## Docker Commands

### Build Image
```bash
./docker_build.sh
```

### Run Jupyter
```bash
./docker_jupyter.sh
```

### Run Bash
```bash
./docker_bash.sh
```

### Clean Up
```bash
./docker_clean.sh
```

### Stop Container
```bash
./docker_exec.sh docker stop [container_name]
```

## Troubleshooting

### Issue: NLTK Data Not Found
**Solution**: The utilities module automatically downloads required NLTK data on first run.

### Issue: Memory Error with Large Dataset
**Solution**: Reduce max_features in TF-IDF extraction or process data in batches.

### Issue: Docker Build Fails
**Solution**: Update requirements.txt or check Docker installation.

### Issue: Jupyter Connection Refused
**Solution**: Check if port 8888 is available; use `docker_bash.sh` to access container.

## References

### MCP Concepts
- [Model Context Protocol Paper](https://example.com)
- Context-aware ML deployment
- Model versioning best practices

### NLP and Feature Extraction
- TF-IDF: Term Frequency-Inverse Document Frequency
- NLTK: Natural Language Toolkit
- Stemming vs Lemmatization

### Fake News Detection
- Kaggle Competition: https://www.kaggle.com/c/fake-news
- Misinformation detection techniques
- Fake news characteristics

## Learning Outcomes

After completing this project, you will understand:

1. **Text Processing Pipeline**: From raw text to ML-ready features
2. **Feature Engineering**: TF-IDF and other text representations
3. **Model Selection**: Choosing appropriate algorithms for text classification
4. **Model Evaluation**: Comprehensive metrics beyond accuracy
5. **Model Context Management**: Using MCP for reproducible ML
6. **Deployment Considerations**: Context validation and compatibility checking

## Next Steps

1. Download the Kaggle dataset
2. Complete the notebook implementations
3. Experiment with different preprocessing techniques
4. Try additional models (SVM, Neural Networks)
5. Explore adversarial training for robustness
6. Create a user interface for real-time classification

## Credits

This project was created as part of MSML610 Fall 2025 coursework at University of Maryland.

**Issue #144**: Spring2025_Fake_News_Detection
**Branch**: TutorTask144_Spring2025_Fake_News_Detection