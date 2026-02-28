# Libact

## Description
- **Active Learning Framework**: libact is a Python library designed for active
  learning, which allows models to identify the most informative data points to
  label, thereby improving efficiency in training.
- **Model Agnostic**: It supports various machine learning models, making it
  versatile for different tasks and allowing students to experiment with
  different algorithms.
- **Interactive Learning**: The library provides tools to interactively select
  data points for labeling, helping students understand the implications of
  their choices in model performance.
- **Comprehensive Toolkit**: Includes a range of active learning strategies,
  such as uncertainty sampling, query-by-committee, and more, enabling
  exploration of different approaches to data selection.
- **Integration with Scikit-learn**: Easily integrates with scikit-learn,
  allowing students to leverage familiar tools while exploring active learning
  concepts.

## Project Objective
The goal of this project is to build a machine learning model that classifies
text data (e.g., movie reviews) into positive or negative sentiments using
active learning. The project will optimize the model's accuracy while minimizing
the number of labeled examples required for training.

## Dataset Suggestions
1. **IMDb Movie Reviews**
   - **Source**: Kaggle
   - **URL**:
     [IMDb Movie Reviews Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - **Data Contains**: 50,000 movie reviews labeled as positive or negative.
   - **Access Requirements**: Free to use after signing up for a Kaggle account.

2. **Twitter Sentiment Analysis**
   - **Source**: Kaggle
   - **URL**:
     [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/kazanova/sentiment140)
   - **Data Contains**: 1.6 million tweets labeled with sentiments (positive,
     negative, neutral).
   - **Access Requirements**: Free to use after signing up for a Kaggle account.

3. **Sentiment140**
   - **Source**: Sentiment140
   - **URL**: [Sentiment140 Dataset](http://help.sentiment140.com/for-students)
   - **Data Contains**: 1.6 million tweets with sentiment labels.
   - **Access Requirements**: Open access, no authentication required.

## Tasks
- **Data Preprocessing**: Clean and preprocess the text data, including
  tokenization and vectorization using techniques like TF-IDF.
- **Initial Model Training**: Train a baseline sentiment analysis model (e.g.,
  logistic regression or SVM) on a small labeled dataset.
- **Active Learning Loop**: Implement an active learning loop using libact to
  iteratively select the most informative samples for labeling and retrain the
  model.
- **Model Evaluation**: Evaluate the model's performance using metrics such as
  accuracy, precision, recall, and F1-score as more data points are labeled.
- **Comparison with Random Sampling**: Compare the active learning approach
  against a random sampling method to highlight efficiency in model training.

## Bonus Ideas
- **Experiment with Different Models**: Test various models (e.g., Random
  Forest, Neural Networks) to see how they perform with active learning.
- **Visualizations**: Create visualizations to demonstrate how the active
  learning process selects data points and its impact on model performance.
- **Hyperparameter Tuning**: Explore hyperparameter tuning for the selected
  models to optimize performance further.
- **Real-time Feedback**: Develop a simple user interface to simulate real-time
  feedback on selected data points for labeling.

## Useful Resources
- [libact Documentation](https://libact.readthedocs.io/en/latest/)
- [Kaggle - IMDb Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Kaggle - Twitter Sentiment Analysis](https://www.kaggle.com/kazanova/sentiment140)
- [Sentiment140 Dataset](http://help.sentiment140.com/for-students)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
