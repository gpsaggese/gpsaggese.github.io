**Description**

In this project, students will utilize fastText, a library developed by Facebook's AI Research (FAIR) for efficient text classification and representation learning. fastText is designed to handle large datasets and can train models quickly, making it ideal for various NLP tasks. It supports word embeddings and supervised learning for text classification, providing a robust framework for working with textual data.

Technologies Used
fastText

- Fast and efficient text classification and representation learning.
- Supports both supervised and unsupervised learning tasks.
- Handles out-of-vocabulary words by using subword information.

---

### Project 1: Text Classification of News Articles
**Difficulty**: 1 (Easy)

**Project Objective**: Build a text classification model to categorize news articles into predefined topics (e.g., politics, sports, technology) using fastText.

**Dataset Suggestions**: Use the "AG News" dataset available on Kaggle (https://www.kaggle.com/amananandrai/ag-news-classification-dataset).

**Tasks**:
- Data Preprocessing:
  - Load the AG News dataset and clean the text data (remove HTML tags, special characters).
- FastText Model Training:
  - Train a fastText model for text classification using the processed dataset.
- Model Evaluation:
  - Evaluate the model's performance using accuracy and F1-score metrics.
- Results Visualization:
  - Visualize the classification results using confusion matrices and bar plots.

**Bonus Ideas (Optional)**:
- Experiment with different hyperparameters for the fastText model.
- Compare the performance of fastText with other text classification models like TF-IDF + Logistic Regression.

---

### Project 2: Sentiment Analysis on Movie Reviews
**Difficulty**: 2 (Medium)

**Project Objective**: Develop a sentiment analysis model to classify movie reviews as positive or negative using fastText.

**Dataset Suggestions**: Use the "IMDb Movie Reviews" dataset available on Kaggle (https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

**Tasks**:
- Data Preparation:
  - Load the IMDb dataset and preprocess the text (tokenization, lowercasing).
- FastText Model Training:
  - Train a fastText model for binary sentiment classification.
- Hyperparameter Tuning:
  - Perform grid search to find optimal hyperparameters for the model.
- Performance Evaluation:
  - Evaluate the model using ROC-AUC and precision-recall curves.
- Visualization:
  - Create word clouds for positive and negative reviews.

**Bonus Ideas (Optional)**:
- Implement a multi-class sentiment analysis (e.g., positive, neutral, negative).
- Use pre-trained word vectors from fastText to enhance model performance.

---

### Project 3: Topic Modeling on Reddit Comments
**Difficulty**: 3 (Hard)

**Project Objective**: Create a topic modeling pipeline to identify and visualize key topics discussed in Reddit comments using fastText embeddings.

**Dataset Suggestions**: Use the "Reddit Comments" dataset from the Pushshift API (https://files.pushshift.io/reddit/comments/).

**Tasks**:
- Data Collection:
  - Use the Pushshift API to collect Reddit comments from a specific subreddit (e.g., r/technology).
- Text Preprocessing:
  - Clean and preprocess the comments (remove stopwords, punctuation).
- FastText Embedding Generation:
  - Generate word embeddings for the comments using fastText.
- Topic Modeling:
  - Apply clustering algorithms (e.g., K-Means) on the embeddings to identify topics.
- Visualization:
  - Visualize the identified topics using t-SNE or PCA for dimensionality reduction.

**Bonus Ideas (Optional)**:
- Implement a dynamic topic modeling approach to analyze how topics evolve over time.
- Compare the clustering results with traditional LDA topic modeling for insights.

