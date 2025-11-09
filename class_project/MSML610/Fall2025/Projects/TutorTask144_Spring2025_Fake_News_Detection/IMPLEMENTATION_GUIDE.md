# Implementation Guide for Fake News Detection with MCP

## Project Status

### Completed ✓
1. **Branch Created**: `TutorTask144_Spring2025_Fake_News_Detection`
2. **Directory Structure**: Set up in `/class_project/MSML610/Fall2025/Projects/TutorTask144_Spring2025_Fake_News_Detection/`
3. **Utility Modules Created**:
   - `utils_data_preprocessing.py`: Text cleaning, tokenization, stopword removal, stemming, TF-IDF extraction
   - `utils_model_training.py`: Model training with MCP context management, evaluation, cross-validation
4. **API Documentation**:
   - `MCP.API.md`: Complete documentation of Model Context Protocol
5. **Requirements**: Updated `requirements.txt` with all dependencies

### Remaining Tasks

#### 1. MCP.API.ipynb
Create a Jupyter notebook demonstrating MCP concepts:
- Import and use `ModelContext` dataclass
- Create `ModelContextManager` instance
- Register model contexts
- Test context compatibility checking
- Save/load contexts from JSON
- Show how to use MCP with actual models

#### 2. FakeNewsDetection.example.md
Document the fake news detection implementation:
- Dataset overview (Kaggle Fake News Detection)
- Data preprocessing pipeline
- Feature extraction (TF-IDF)
- Model selection and training
- Evaluation metrics (ROC-AUC, precision, recall)
- Results and findings
- How MCP is applied in this scenario

#### 3. FakeNewsDetection.example.ipynb
Implement the complete fake news detection system:
- Load Kaggle dataset
- Preprocess text using `utils_data_preprocessing.py`
- Extract TF-IDF features
- Train multiple models (Logistic Regression, Random Forest, Gradient Boosting)
- Evaluate using `utils_model_training.py`
- Use `ModelContextManager` to track models
- Create cross-validation comparisons
- Visualize results

#### 4. Update Dockerfile
Add ML-specific dependencies:
```dockerfile
RUN pip install -r requirements.txt
```

#### 5. Update README.md
Include:
- Project overview
- Dataset download instructions
- Docker build and run commands
- How to run notebooks
- MCP and fake news detection concepts
- Results and findings

## Key Implementation Tips

### Data Loading
- Download from: https://www.kaggle.com/c/fake-news/data
- Expected columns: id, title, author, text, label
- Use `utils_data_preprocessing.load_fake_news_data()` and `validate_data()`

### Preprocessing Pipeline
```python
from utils_data_preprocessing import preprocess_text, extract_tfidf_features, prepare_dataset

# Load and prepare data
df = pd.read_csv('fake_news.csv')
X_train, X_test, y_train, y_test = prepare_dataset(df)

# Extract TF-IDF
tfidf_train, vectorizer = extract_tfidf_features(X_train['text'].tolist())
tfidf_test = vectorizer.transform(X_test['text']).toarray()
```

### Model Training with MCP
```python
from utils_model_training import (
    create_model, train_model, evaluate_model,
    ModelContextManager, ModelContext
)

# Create and train model
model = create_model('random_forest', {'n_estimators': 100})
train_model(model, tfidf_train, y_train)

# Create context
context = ModelContext(
    model_id='fake_news_rf_v1',
    model_name='Random Forest Fake News Detector',
    model_type='random_forest',
    feature_type='tfidf',
    created_at=datetime.now().isoformat(),
    training_samples=len(X_train),
    validation_samples=0,
    test_samples=len(X_test),
    preprocessed=True,
    hyperparameters={'n_estimators': 100}
)

# Register and validate
manager = ModelContextManager()
manager.register_context(context)

# Check compatibility before inference
if manager.is_context_compatible('fake_news_rf_v1', 'tfidf', True):
    predictions = model.predict(tfidf_test)
    metrics = evaluate_model(model, tfidf_test, y_test)
```

### Model Comparison
```python
from utils_model_training import compare_models

models_config = {
    'logistic_regression': {
        'model_type': 'logistic_regression',
        'hyperparameters': {'max_iter': 1000}
    },
    'random_forest': {
        'model_type': 'random_forest',
        'hyperparameters': {'n_estimators': 100, 'max_depth': 20}
    },
    'gradient_boosting': {
        'model_type': 'gradient_boosting',
        'hyperparameters': {'n_estimators': 100, 'learning_rate': 0.1}
    }
}

results_df = compare_models(
    models_config, tfidf_train, y_train, tfidf_test, y_test
)
```

### Cross-Validation
```python
from utils_model_training import cross_validate_model

model = create_model('random_forest')
cv_results = cross_validate_model(model, tfidf_train, y_train, cv=5)
```

## Notebook Structure

### MCP.API.ipynb
- Cell 1: Import and setup
- Cell 2: Explain ModelContext concept
- Cell 3: Create ModelContext instance
- Cell 4: Initialize ModelContextManager
- Cell 5: Register contexts
- Cell 6: Retrieve and inspect contexts
- Cell 7: Test compatibility checking
- Cell 8: Save and load contexts as JSON
- Cell 9: Summary and best practices

### FakeNewsDetection.example.ipynb
- Cell 1: Imports and setup
- Cell 2: Load dataset
- Cell 3: Explore data (shape, samples, class distribution)
- Cell 4: Text preprocessing examples
- Cell 5: Data preparation (train/test split)
- Cell 6: TF-IDF feature extraction
- Cell 7: Create ModelContextManager
- Cell 8: Train Logistic Regression with MCP context
- Cell 9: Train Random Forest with MCP context
- Cell 10: Train Gradient Boosting with MCP context
- Cell 11: Compare models and results
- Cell 12: Cross-validation analysis
- Cell 13: Confusion matrix and classification reports
- Cell 14: Feature importance (for tree models)
- Cell 15: Visualizations (ROC curves, confusion matrices)
- Cell 16: Save models and contexts
- Cell 17: Demonstrate context compatibility checking
- Cell 18: Conclusions and findings

## Next Steps

1. Download the Kaggle fake news dataset
2. Create the `.ipynb` files following the structure above
3. Test all notebooks with "Restart and Run All"
4. Verify Docker builds successfully
5. Create PR on GitHub and request review

## Dataset Information

The Kaggle Fake News Detection dataset contains:
- **Total samples**: ~20,000 news articles
- **Target**: Binary classification (0=fake, 1=real)
- **Features**: title, author, text, date, label
- **Class distribution**: Relatively balanced (~9800 fake, ~10000 real)

Download: https://www.kaggle.com/c/fake-news/data

## Common Issues and Solutions

### Issue: NLTK data not found
**Solution**: Ensure `utils_data_preprocessing.py` has already downloaded required data

### Issue: Out of memory with large datasets
**Solution**: Use `batch_size` parameter or reduce `max_features` in TF-IDF

### Issue: Model context mismatch
**Solution**: Always check `manager.is_context_compatible()` before inference

### Issue: Docker build fails
**Solution**: Check requirements.txt dependencies, may need to adjust versions
