# Description

Gensim is an open-source Python library for unsupervised text modeling: word
embeddings, topic models, and document similarity, all streamed so that the corpus
does not need to fit in memory. It solves the problem of turning raw text into
vectors and topics without labels. It is worth a 60-minute tutorial because one small
API covers the classical representations of text, from TF-IDF to LDA to Word2Vec, and
they can be compared on the same task.

## Technologies Used

Gensim

- Word embeddings with `Word2Vec` and `FastText`, and `KeyedVectors` for queries
- `Dictionary` and `TfidfModel` to build bag-of-words and TF-IDF corpora
- Topic modeling with `LdaModel` and `CoherenceModel`
- Similarity search over documents with `MatrixSimilarity`

# Tutorial

- Implement the tutorial "Learn Gensim in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of `.claude/skills/tutorial_in_60_mins.rules.md`
  - Read `class_project/project_descriptions/DATA605/Gensim_Project_Description.md`
  - Read the `README.md` of the two DATA605 Gensim projects, and reuse what is good
    - `class_project/data605/Spring2025/projects/TutorTask96_Spring2025_Real-Time_Bitcoin_Data_Processing_with_Gensim/`
    - `class_project/data605/Spring2026/projects/UmdTask420_DATA605_Spring2026_Gensim_topic_modeling/`
  - Read the `README.md` of the Fall2025 SBert project for the neural embeddings
    - `class_project/msml610/Fall2025/projects/Fall2025_SBert_Sentiment_Analysis_with_Sentence_Embeddings/`
- Create `tutorials/Gensim/`, since it does not exist yet
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with NLTK and SBert from the text representation point of view,
  e.g., the quality of the embeddings for the same classification task
- Deliverables:
  - `gensim_utils.py`
  - `gensim.API.ipynb`
  - `gensim.example.ipynb`

# Project

## Project 1: Sentiment Classification of Movie Reviews

- **Difficulty**: 1 (Easy)
- **Project Objective**: Classify movie reviews as positive or negative from Word2Vec
  document vectors, and check if they beat a bag-of-words baseline
- **Dataset Suggestions**:
  [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Tasks**:
  - **Preprocess the Text**: Tokenize the reviews with `simple_preprocess`, remove
    the stop words, and split them into train and test sets
  - **Define the Problem**: Predict the sentiment as a binary label, and fit a TF-IDF
    plus logistic regression baseline
  - **Train Word Embeddings**: Train `Word2Vec` on the training reviews, and inspect
    the `most_similar` words of five sentiment words
  - **Build Document Features**: Represent each review as the mean of its word
    vectors, and fit a logistic regression on them
  - **Evaluate the Models**: Report accuracy, precision, and recall of both models on
    the test set, with a confusion matrix
  - **Visualize the Embeddings**: Project the 200 most frequent words to 2D with PCA
    and label the sentiment words
- **Bonus Ideas (Optional)**: Compare logistic regression with SVM and Random Forest;
  test the effect of the `vector_size` and `window` parameters

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Text
  - Result: `tutorials/Gensim/` container running, and a table with the vocabulary
    size and the tokens per review
- Milestone 2: API notebook
  - Project tasks: Train Word Embeddings, Build Document Features
  - Result: `gensim.API.ipynb` covering `Word2Vec`, `FastText`, `KeyedVectors`,
    `Dictionary`, `TfidfModel`, `LdaModel` with `CoherenceModel`, and
    `MatrixSimilarity`
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Train Word Embeddings, Build Document
    Features, Evaluate the Models, Visualize the Embeddings
  - Result: `gensim.example.ipynb` running end to end

## Project 2: Topic Modeling and Classification of News Articles

- **Difficulty**: 2 (Medium)
- **Project Objective**: Find the main topics of BBC news articles, and check how
  well they align with the category labels used by a supervised classifier
- **Dataset Suggestions**:
  [BBC News](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category),
  with about 2,200 articles in 5 categories
- **Tasks**:
  - **Preprocess the Text**: Lowercase, tokenize, and remove the stop words, then
    build a `Dictionary` and a bag-of-words corpus
  - **Fit the Topic Models**: Fit `LdaModel` for 3 to 12 topics, and choose the
    number of topics with the `CoherenceModel` c_v score
  - **Visualize the Topics**: Show the topics and their distances with `pyLDAvis`
  - **Classify the Articles**: Train `Word2Vec` or `FastText`, average the vectors of
    each article, and fit a logistic regression and an SVM
  - **Evaluate the Classifiers**: Report accuracy, precision, recall, and F1 for each
    category on the test set
  - **Interpret the Topics**: Plot a heatmap of the topic weights per category, and
    describe which topics match which category
- **Bonus Ideas (Optional)**: Compare `LdaModel` with the `Nmf` model of Gensim; find
  the categories that share topics; use a feed-forward network on FastText vectors

## Project 3: Similarity and Clustering of Research Papers

- **Difficulty**: 3 (Hard)
- **Project Objective**: Retrieve the papers similar to a given abstract and group
  the papers by research area, comparing two embedding models
- **Dataset Suggestions**:
  [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv), using a
  sample of 20,000 abstracts
- **Tasks**:
  - **Preprocess the Abstracts**: Sample abstracts from categories such as `cs.AI`,
    `cs.LG`, and `stat.ML`, then clean and tokenize them
  - **Train the Embeddings**: Train `FastText` and `Word2Vec` on the abstracts, and
    compare their vectors for out-of-vocabulary words
  - **Build Document Vectors**: Represent each abstract as the average of its word
    vectors, for each of the two models
  - **Rank Similar Papers**: Retrieve the ten nearest abstracts with
    `MatrixSimilarity`, and judge the topical relevance of 20 queries by hand
  - **Cluster the Papers**: Apply K-means and hierarchical clustering to the vectors,
    and report the Silhouette Score and the Davies-Bouldin Index
  - **Evaluate against the Categories**: Compare the clusters with the arXiv
    categories using the adjusted Rand index, for both embedding models
- **Bonus Ideas (Optional)**: Label each cluster with its top keywords; build a
  recommender that combines the similarity search with the cluster of the query
