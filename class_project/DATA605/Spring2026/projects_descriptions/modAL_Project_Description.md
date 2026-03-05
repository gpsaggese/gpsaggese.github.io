# modAL

## Description
- ModAL is a Python library designed for active learning, allowing users to
  efficiently query and label data points for machine learning models.
- It provides a flexible framework that integrates seamlessly with popular
  machine learning libraries like Scikit-learn.
- Key features include various query strategies (e.g., uncertainty sampling,
  query-by-committee) to select the most informative data points for labeling.
- ModAL supports both supervised and semi-supervised learning, enabling users to
  improve model performance with fewer labeled instances.
- The library is easy to install and use, making it accessible for both
  beginners and experienced data scientists.

## Project Objective
The goal of this project is to build an active learning model that predicts the
sentiment of movie reviews. The project will focus on optimizing the model's
accuracy while minimizing the number of labeled data points required for
training.

## Dataset Suggestions
1. **IMDb Movie Reviews**
   - **Source Name**: Kaggle
   - **URL**:
     [IMDb Movie Reviews Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - **Data Contains**: 50,000 movie reviews labeled as positive or negative.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **Sentiment140**
   - **Source Name**: Sentiment140
   - **URL**: [Sentiment140 Dataset](http://help.sentiment140.com/for-students/)
   - **Data Contains**: 1.6 million tweets labeled for sentiment (positive,
     negative, neutral).
   - **Access Requirements**: Publicly available, no authentication required.

3. **Twitter US Airline Sentiment**
   - **Source Name**: Kaggle
   - **URL**:
     [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
   - **Data Contains**: 14,000 tweets about US airlines, labeled as positive,
     negative, or neutral.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

4. **Stanford Sentiment Treebank**
   - **Source Name**: Stanford University
   - **URL**:
     [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html)
   - **Data Contains**: Sentiment labels for 11,855 movie reviews with
     fine-grained sentiment scores.
   - **Access Requirements**: Publicly available, no authentication required.

## Tasks
- **Data Preparation**: Load the chosen dataset, clean the text data, and
  preprocess it for sentiment analysis.
- **Model Selection**: Choose an appropriate machine learning model (e.g.,
  Logistic Regression, Random Forest) for sentiment classification.
- **Active Learning Setup**: Implement modAL to create an active learning loop,
  including the selection of the most informative samples for labeling.
- **Model Training and Evaluation**: Train the model using both labeled and
  unlabeled data, evaluate its performance using metrics such as accuracy and F1
  score.
- **Analysis and Reporting**: Analyze the results, discuss the impact of active
  learning on model performance, and prepare a report summarizing findings.

## Bonus Ideas
- Experiment with different query strategies in modAL (e.g., uncertainty
  sampling vs. query-by-committee) and compare their effectiveness.
- Implement a user interface that allows users to label data points dynamically
  during the active learning process.
- Investigate the impact of varying the initial size of the labeled dataset on
  model performance.
- Extend the project to include multi-class sentiment classification (e.g.,
  positive, negative, neutral).

## Useful Resources
- [modAL Documentation](https://modal-python.readthedocs.io/en/latest/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Natural Language Processing with Python](https://www.nltk.org/book/) - A
  great resource for text preprocessing and sentiment analysis techniques.
