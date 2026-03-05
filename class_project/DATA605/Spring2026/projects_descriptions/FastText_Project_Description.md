# FastText

## Description
- FastText is an open-source library developed by Facebook AI Research for
  efficient text classification and representation.
- It allows users to create word embeddings and perform supervised learning
  tasks such as text classification with high accuracy and speed.
- FastText can handle large datasets and provides pre-trained models for various
  languages, making it accessible for multilingual applications.
- The tool supports subword information, which allows it to generate embeddings
  for out-of-vocabulary words, improving its robustness in natural language
  processing tasks.
- FastText is designed to be easy to use, with a command-line interface and
  Python bindings, making it suitable for both beginners and advanced users.

## Project Objective
The goal of the project is to build a text classification model that predicts
the category of news articles based on their content. Students will optimize the
model for accuracy and efficiency in classifying articles into predefined
categories.

## Dataset Suggestions
1. **AG News Dataset**
   - **Source**: Kaggle
   - **URL**:
     [AG News Dataset](https://www.kaggle.com/amananandrai/ag-news-classification-dataset)
   - **Data Contains**: News articles labeled into four categories: World,
     Sports, Business, Science/Technology.
   - **Access Requirements**: Free to use; requires a Kaggle account for
     download.

2. **20 Newsgroups Dataset**
   - **Source**: Scikit-learn
   - **URL**: [20 Newsgroups Dataset](http://qwone.com/~jason/20Newsgroups/)
   - **Data Contains**: Approximately 20,000 newsgroup documents, partitioned
     across 20 different newsgroups.
   - **Access Requirements**: Publicly accessible; can be loaded directly via
     Scikit-learn.

3. **BBC News Classification Dataset**
   - **Source**: Kaggle
   - **URL**:
     [BBC News Dataset](https://www.kaggle.com/datasets/rtatman/bbc-news-dataset)
   - **Data Contains**: News articles categorized into five topics: business,
     entertainment, politics, sport, and tech.
   - **Access Requirements**: Free to use; requires a Kaggle account for
     download.

4. **Sentiment140 Dataset**
   - **Source**: Kaggle
   - **URL**: [Sentiment140](https://www.kaggle.com/kazanova/sentiment140)
   - **Data Contains**: 1.6 million tweets labeled for sentiment (positive or
     negative).
   - **Access Requirements**: Free to use; requires a Kaggle account for
     download.

## Tasks
- **Data Preprocessing**: Clean and preprocess the text data by removing noise,
  normalizing text, and splitting into training and test sets.
- **Model Training**: Use FastText to train a text classification model on the
  selected dataset, utilizing pre-trained word embeddings if available.
- **Model Evaluation**: Evaluate the model's performance using metrics such as
  accuracy, precision, recall, and F1-score on the test set.
- **Hyperparameter Tuning**: Experiment with different hyperparameters (e.g.,
  learning rate, epoch count) to optimize model performance.
- **Result Analysis**: Analyze the results, including confusion matrix
  visualization and error analysis to understand misclassifications.

## Bonus Ideas
- Implement a multi-label classification task by predicting multiple categories
  for each article.
- Compare the performance of FastText with other text classification models
  (e.g., BERT, traditional ML models) to evaluate its effectiveness.
- Extend the project by building a web application that allows users to input
  text and receive category predictions in real-time.

## Useful Resources
- [FastText Official Documentation](https://fasttext.cc/docs/en/crawl.html)
- [FastText GitHub Repository](https://github.com/facebookresearch/fastText)
- [Kaggle FastText Tutorial](https://www.kaggle.com/code/siddharthbansal/fasttext-for-text-classification)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)
