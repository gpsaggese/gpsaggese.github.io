# Description

NLTK (Natural Language Toolkit) is a Python library for natural language processing
with tools for tokenization, stemming, tagging, parsing, and semantic reasoning, and
access to corpora and lexical resources like WordNet. It solves the problem of
building a text-processing pipeline step by step, with transparent and inspectable
components. It is worth a 60-minute tutorial because it teaches the classic NLP
pipeline (tokenize, normalize, featurize, classify) that modern models build on.

## Technologies Used

NLTK

- Comprehensive suite of libraries for text processing
- Easy-to-use interfaces for common NLP tasks like tokenization and stemming
- Access to a large collection of corpora and lexical resources like WordNet
- Includes a lexicon-based sentiment analyzer (VADER) and trainable classifiers

# Tutorial

- Implement the tutorial "Learn NLTK in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Read the `README.md` of the earlier NLTK projects, and reuse what is good
    - `class_project/msml610/Fall2025/projects/UmdTask66_Fall2025_NLTK_Named_Entity_Recognition_in_Scientific_Publications/`
    - `class_project/data605/Spring2025/projects/TutorTask98_Spring2025_Real-Time_Bitcoin_Sentiment_Analysis_Using_NLTK_and_Selenium/`
- Create `tutorials/NLTK/`, since it does not exist yet
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `nltk_utils.py`
  - `nltk.API.ipynb`
  - `nltk.example.ipynb`

# Project

## Project 1: Sentiment Analysis of Financial News

- **Difficulty**: 2 (Medium)
- **Project Objective**: Classify financial news headlines as negative, neutral, or
  positive, and compare a lexicon-based analyzer with a trained classifier by macro
  F1
- **Dataset Suggestions**:
  [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- **Tasks**:
  - **Preprocess the Headlines**: Load the headlines, keep the three sentiment
    labels, and split them into train and test sets
  - **Define the Problem**: Fix the three-class target and compute a majority-class
    baseline on the test set
  - **Tokenize and Featurize**: Use `word_tokenize`, remove `stopwords`, normalize
    with `WordNetLemmatizer`, and build bag-of-words features
  - **Classify with NLTK**: Score the headlines with `SentimentIntensityAnalyzer`
    (VADER) and train a `NaiveBayesClassifier` on the bag-of-words features
  - **Evaluate the Classifiers**: Compute accuracy, macro F1, and the confusion
    matrix of the VADER analyzer, the Naive Bayes classifier, and the baseline
  - **Visualize the Results**: Plot the `FreqDist` of the top words per sentiment and
    the most informative features of the classifier
- **Bonus Ideas (Optional)**: Add a finance lexicon (e.g., Loughran-McDonald) to
  improve VADER on financial text; compare with a pre-trained transformer

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Headlines
  - Result: `tutorials/NLTK/` container running with the NLTK data downloaded, and
    train and test tables of labeled headlines
- Milestone 2: API notebook
  - Project tasks: Tokenize and Featurize, Classify with NLTK
  - Result: `nltk.API.ipynb` covering tokenization, stop words, stemming and
    lemmatization, `FreqDist`, VADER, and `NaiveBayesClassifier` on a few sentences
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Tokenize and Featurize, Classify with NLTK,
    Evaluate the Classifiers, Visualize the Results
  - Result: `nltk.example.ipynb` running end to end

## Project 2: Topic Modeling of News Articles

- **Difficulty**: 2 (Medium)
- **Project Objective**: Identify underlying topics in a collection of news articles,
  optimizing for coherent topic representation
- **Dataset Suggestions**: [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/),
  loaded with `sklearn.datasets.fetch_20newsgroups`
- **Tasks**:
  - **Ingest the Data**: Load the 20 Newsgroups dataset with scikit-learn
  - **Preprocess the Text**: Clean the text with NLTK (remove stop words, stemming)
  - **Vectorize the Text**: Convert the text into a document-term matrix using TF-IDF
  - **Model the Topics**: Implement Latent Dirichlet Allocation (LDA) to extract
    topics from the text
  - **Visualize the Topics**: Visualize the topics using pyLDAvis to interpret and
    analyze the results
- **Bonus Ideas (Optional)**: Experiment with different numbers of topics and
  evaluate coherence scores; compare LDA results with Non-negative Matrix
  Factorization (NMF)

## Project 3: Named Entity Recognition in Scientific Publications

- **Difficulty**: 3 (Hard)
- **Project Objective**: Extract named entities (e.g., authors, institutions, and
  research topics) from a set of scientific papers, optimizing for recall and
  precision in entity extraction
- **Dataset Suggestions**:
  [CORD-19](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)
- **Tasks**:
  - **Acquire the Data**: Download and preprocess the CORD-19 dataset
  - **Clean the Text**: Remove unnecessary elements (e.g., references, figures)
  - **Tokenize and Tag**: Use NLTK to tokenize the text and apply part-of-speech
    tagging
  - **Recognize Entities**: Start with NLTK's built-in NER tools (`ne_chunk`), then
    compare results with pre-trained models (e.g., spaCy or Hugging Face
    Transformers)
  - **Evaluate the Recognizers**: Measure the performance of the NER systems using
    standard metrics like F1-score
- **Bonus Ideas (Optional)**: Integrate additional datasets to improve entity
  recognition performance; explore the use of deep learning models for NER and
  compare results with traditional methods
