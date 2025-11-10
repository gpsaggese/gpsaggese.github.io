# Vertex AI Sentiment Analysis on Social Media Posts

**MSML610 Fall 2025 - Class Project**
**Author:** Balamurugan Manickaraj, Abhinav Kumar, Adwaith Santhosh
**Project Difficulty:** Level 3 (Hard)

---

## Project Overview

This project implements a comprehensive sentiment analysis system for social media posts (tweets) using Google Cloud Vertex AI. The system classifies airline-related tweets into three sentiment categories: positive, neutral, and negative.

### Objectives

- Develop an NLP model to analyze sentiments expressed in social media posts
- Learn and demonstrate Google Cloud Vertex AI capabilities
- Create a tutorial-style implementation for educational purposes
- Deploy a production-ready sentiment classification system

### Technologies Used

- **Google Cloud Vertex AI**: ML platform for training and deployment
- **Python**: Primary programming language
- **Pandas/NumPy**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **scikit-learn**: Model evaluation and metrics
- **NLTK/spaCy**: Text preprocessing (planned)
- **Transformers**: Pre-trained models like BERT (optional bonus)

---

## Dataset

**Twitter US Airline Sentiment**
- **Source**: [Kaggle - Twitter Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Size**: 14,640+ tweets
- **Features**: 15 columns including text, sentiment labels, airline names, confidence scores
- **Classes**: 3 sentiment categories (positive, neutral, negative)
- **Note**: Dataset exhibits class imbalance (more negative tweets)

### Dataset Characteristics
- Text length: Varies from 10-280 characters (typical Twitter length)
- Airlines covered: Virgin America, United, Southwest, Delta, US Airways, American
- Contains metadata: timestamps, user timezone, retweet counts, etc.

---

## Implementation Phases

### Phase 1: Data Ingestion

**Components:**
- Data loading from CSV
- Exploratory Data Analysis (EDA)
- Dataset statistics and visualization
- Text statistics analysis
- Data splitting (train/val/test: 70/15/15)
- JSONL format preparation for Vertex AI

### Phase 2: Text Preprocessing

**Tasks:**
- Text cleaning (remove URLs, mentions, special characters)
- Tokenization
- Stop words removal
- Text normalization (lowercase, lemmatization)
- Handle Twitter-specific elements (@mentions, hashtags, emojis)
- Feature engineering

### Phase 3: Model Training

**Approach 1: AutoML Natural Language**
- Create Vertex AI dataset
- Configure AutoML training job
- Train text classification model
- Monitor training progress

**Approach 2: Custom Training with BERT (Bonus)**
- Load pre-trained BERT model
- Fine-tune on airline sentiment data
- Custom training job on Vertex AI
- GPU/TPU acceleration

**Key Components:**
- Model training scripts
- Hyperparameter configuration
- Training job submission
- Model versioning

### Phase 4: Hyperparameter Tuning

**Tasks:**
- Define hyperparameter search space
- Configure Vertex AI tuning job
- Run parallel trials
- Select best model

### Phase 5: Model Evaluation

**Metrics:**
- F1-Score (macro and weighted)
- Confusion Matrix
- Precision, Recall, Accuracy
- Per-class performance
- ROC curves

### Phase 6: Model Deployment

**Tasks:**
- Deploy model to Vertex AI endpoint
- Configure auto-scaling
- Test endpoint with sample predictions
- Set up monitoring

### Phase 7: Bonus Features

**Optional Enhancements:**
- Dashboard for sentiment trends (Google Data Studio)
- BERT comparison study
- Real-time sentiment monitoring
- Aspect-based sentiment analysis
- Explainable AI integration

## Usage

**Using notebooks**
   - `vertex_ai.API.ipynb`: Learn Vertex AI APIs and utilities
   - `vertex_ai.example.ipynb`: Run complete end-to-end pipeline
