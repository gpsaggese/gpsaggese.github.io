# Gensim

## Description
- Gensim is an open-source Python library designed for topic modeling and
  document similarity analysis, primarily used in natural language processing
  (NLP).
- It excels in unsupervised learning tasks, allowing users to extract insights
  from large text corpora through efficient algorithms.
- Key features include support for Word2Vec, FastText, and Latent Dirichlet
  Allocation (LDA), enabling the creation of word embeddings and topic modeling.
- Gensim is optimized for handling large datasets, streaming data, and
  incremental learning, making it suitable for real-world applications.
- The library provides easy-to-use interfaces for preprocessing text, training
  models, and evaluating results, which helps streamline the data science
  workflow.

## Project Objective
The goal of the project is to build a topic modeling system that can
automatically identify and categorize topics from a collection of news articles.
Students will optimize the model to accurately detect and label these topics
using LDA, while evaluating the coherence and relevance of the topics generated.

## Dataset Suggestions
1. **Kaggle News Articles Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Kaggle News Articles](https://www.kaggle.com/datasets/sunnysai12345/news20)
   - **Data Contains**: A collection of 20,000 news articles across various
     categories.
   - **Access Requirements**: Free access with a Kaggle account.

2. **BBC News Classification Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [BBC News Dataset](https://archive.ics.uci.edu/ml/datasets/BBC+News+Articles)
   - **Data Contains**: 2,225 articles from BBC categorized into five topics:
     business, entertainment, politics, sport, and tech.
   - **Access Requirements**: Publicly available without authentication.

3. **20 Newsgroups Dataset**
   - **Source**: Scikit-learn
   - **URL**: [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)
   - **Data Contains**: A collection of approximately 20,000 newsgroup
     documents, partitioned across 20 different newsgroups.
   - **Access Requirements**: Accessible via Scikit-learn's built-in dataset
     loader.

4. **Reuters News Dataset**
   - **Source**: NLTK
   - **URL**: [Reuters Dataset](https://www.nltk.org/nltk_data/)
   - **Data Contains**: A collection of Reuters news articles categorized into
     different topics.
   - **Access Requirements**: Available through the NLTK library, which can be
     installed via pip.

## Tasks
- **Data Preprocessing**: Clean and preprocess the text data, including
  tokenization, stopword removal, and lemmatization.
- **Model Training**: Implement LDA using Gensim to train the topic model on the
  preprocessed dataset.
- **Topic Evaluation**: Use coherence score metrics to evaluate the quality of
  the generated topics and make necessary adjustments.
- **Visualization**: Create visualizations of the topics and their distributions
  using tools like pyLDAvis to interpret the results effectively.
- **Reporting**: Document the findings, including insights on the identified
  topics and their implications, along with model performance metrics.

## Bonus Ideas
- Extend the project by implementing a supervised classification model to
  predict the category of articles based on the identified topics.
- Compare the results of LDA with other topic modeling techniques available in
  Gensim, such as Hierarchical Dirichlet Process (HDP).
- Challenge students to incorporate sentiment analysis on the topics identified
  to explore the emotional tone of articles within each topic.

## Useful Resources
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Gensim GitHub Repository](https://github.com/RaRe-Technologies/gensim)
- [NLTK Documentation](https://www.nltk.org/)
- [pyLDAvis Documentation](https://pyldavis.readthedocs.io/en/latest/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
