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
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `nltk_utils.py`
  - `nltk.API.ipynb`
  - `nltk.example.ipynb`

# Project

## Project 1 (Fall2026): SEC 10-K Risk Language and Next-Year Stock Volatility

- **Project Objective**: Test whether the language of the risk factors in 10-K
  filings predicts high stock volatility in the next year beyond the volatility of
  the current year, by comparing a text classifier with a persistence baseline by
  ROC AUC on later years
- **Dataset Suggestions**:
  - [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus) with 10-K
    filings split by item, in yearly `train.jsonl`, `validate.jsonl`, and
    `test.jsonl` shards
    - Stream each shard line by line and keep only `section_1A` (risk factors) of
      the S&P 500 companies
    - Start with the `validate.jsonl` and `test.jsonl` shards (about 180 MB each),
      and add `train.jsonl` (about 1.4 GB) only if the sample is too small
  - [S&P 500 Companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
    with the tickers and the CIK, to match EDGAR-CORPUS and the prices
    - Today's constituents introduce survivorship bias, so state this limit in the
      analysis
  - Daily adjusted prices from [`yfinance`](https://github.com/ranaroussi/yfinance)
  - [Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)
    for the finance word lists (negative, uncertainty, litigious)
- **Tasks**:
  - **Acquire the Filings**: Stream the yearly shards for filing years 2010-2020,
    keep the risk factors of the S&P 500 companies, and save a table with the CIK,
    the ticker, the filing year, and the text
  - **Build the Volatility Labels**: Compute the realized volatility of each
    calendar year from daily returns, and label a filing as high risk if the
    volatility of the year after the filing year is above the median of the
    companies in that year
  - **Tokenize and Featurize**: Use `sent_tokenize` and `word_tokenize`, remove
    `stopwords`, stem with `SnowballStemmer`, build a `FreqDist` per filing, and add
    the fractions of Loughran-McDonald negative, uncertainty, and litigious words
  - **Classify with NLTK**: Train a `NaiveBayesClassifier` on filings through 2016,
    with the presence of the most frequent stemmed words and the binned
    Loughran-McDonald fractions as features, and compare it with the persistence
    baseline that predicts high risk when the volatility of the filing year was
    above the median
  - **Evaluate the Classifiers**: Compute ROC AUC, accuracy, and macro F1 on the
    filings from 2017 on, give a bootstrap 95% confidence interval over the
    filings, and report the filings of 2019 (target year 2020, with the COVID-19
    volatility) separately
  - **Visualize the Results**: Plot the most informative stems, the top collocations
    from `BigramCollocationFinder`, and the ROC curves of the text model and the
    baseline
- **Bonus Ideas (Optional)**: Combine the text score and the past volatility with a
  logistic regression; compare the risk factors of different GICS sectors

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Acquire the Filings, Build the Volatility Labels
  - Result: project dir created and container running with the NLTK data downloaded,
    and the table of risk factors with the volatility labels, with the number of
    filings and the share of high-risk labels per year
- Milestone 2: API notebook
  - Project tasks: Tokenize and Featurize, Classify with NLTK
  - Result: `nltk.API.ipynb` covering `sent_tokenize` and `word_tokenize`,
    `stopwords`, `SnowballStemmer`, `FreqDist`, `BigramCollocationFinder`, and
    `NaiveBayesClassifier` on a few sentences
- Milestone 3: Example notebook
  - Project tasks: Build the Volatility Labels, Tokenize and Featurize, Classify with
    NLTK, Evaluate the Classifiers, Visualize the Results
  - Result: `nltk.example.ipynb` running end to end

## Project 2: Sentiment Analysis of Financial News

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

## Project 3: Hawkish or Dovish Central Bank Sentences

- **Project Objective**: Classify sentences of Federal Reserve (FOMC) communications as
  dovish, neutral, or hawkish, and compare a WordNet-expanded stance lexicon, VADER
  sentiment, and trained classifiers by macro F1 on later years
- **Dataset Suggestions**:
  - [FOMC Communication](https://huggingface.co/datasets/gtfintechlab/fomc_communication)
    with about 2,500 labeled sentences from 1996-2022 and a `year` column
    (label 0 is dovish, 1 is hawkish, 2 is neutral)
  - [Federal Funds Effective Rate](https://fred.stlouisfed.org/series/FEDFUNDS) from
    FRED, for the context plot
- **Tasks**:
  - **Preprocess the Sentences**: Load the labeled sentences, map the labels to
    dovish, hawkish, and neutral, and split by the `year` column: train through 2016
    and test from 2017 on, so the test period is later than the training period
  - **Tokenize and Featurize**: Use `word_tokenize`, lowercase the tokens, remove
    `stopwords` but keep the negations, normalize with `WordNetLemmatizer`, and add
    bigram features with `nltk.bigrams`
  - **Expand a Stance Lexicon**: Write seed lists of hawkish and dovish words (e.g.,
    `tighten` and `accommodative`), expand them with the WordNet `synsets` and
    `lemma_names`, and score each sentence by the number of hawkish words minus the
    number of dovish words
  - **Classify with NLTK**: Train a `NaiveBayesClassifier` and a `MaxentClassifier` on
    unigram and bigram features, and score the sentences with
    `SentimentIntensityAnalyzer` (VADER) as a sentiment-based comparison
  - **Evaluate the Classifiers**: Compute accuracy, macro F1, and the confusion matrix
    on the test years for the majority-class baseline (neutral), VADER, the lexicon,
    Naive Bayes, and MaxEnt, and list sentences where the VADER sentiment and the
    stance disagree
  - **Visualize the Results**: Plot the most informative features of the best
    classifier, and the yearly share of hawkish minus dovish sentences next to the
    federal funds rate
- **Bonus Ideas (Optional)**: Tune the seed lists on a validation period; apply the
  best classifier to the statements on the
  [FOMC calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
  page and compare the stance with the changes of the federal funds rate
