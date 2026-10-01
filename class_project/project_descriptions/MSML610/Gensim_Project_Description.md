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
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
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

## Project 1 (Fall2026): Sentiment Classification of Financial News

- **Project Objective**: Classify financial news sentences as negative, neutral, or
  positive from Word2Vec sentence vectors, and check if they beat a TF-IDF baseline
- **Dataset Suggestions**:
  [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank),
  with 4,846 sentences from company news labeled by finance experts (use the file
  `Sentences_50Agree.txt` in the zip of the dataset repo)
- **Tasks**:
  - **Preprocess the Text**: Tokenize the sentences with `simple_preprocess`, remove
    the stop words, and make a stratified train and test split
  - **Define the Problem**: Predict the sentiment as a 3-class label, and fit a TF-IDF
    plus logistic regression baseline and a majority-class baseline
  - **Train Word Embeddings**: Train `Word2Vec` on the training sentences, and inspect
    the `most_similar` words of five finance terms, e.g., `profit`, `loss`, and `debt`
  - **Build Sentence Features**: Represent each sentence as the mean of its word
    vectors, and fit a logistic regression on them
  - **Evaluate the Models**: Report accuracy and macro-F1 of all models on the test
    set, with a confusion matrix, since the neutral class dominates
  - **Visualize the Embeddings**: Project the 200 most frequent words to 2D with PCA
    and label the finance terms
- **Bonus Ideas (Optional)**: Load the pretrained `glove-wiki-gigaword-100` vectors
  with `gensim.downloader` as `KeyedVectors` and compare them with the embeddings
  trained on 4,800 sentences; test the effect of `vector_size` and `window`

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Text
  - Result: project dir created and container running, and a table with the
    vocabulary size, the tokens per sentence, and the class counts
- Milestone 2: API notebook
  - Project tasks: Train Word Embeddings, Build Sentence Features
  - Result: `gensim.API.ipynb` covering `Word2Vec`, `FastText`, `KeyedVectors`,
    `Dictionary`, `TfidfModel`, `LdaModel` with `CoherenceModel`, and
    `MatrixSimilarity`
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Train Word Embeddings, Build Sentence
    Features, Evaluate the Models, Visualize the Embeddings
  - Result: `gensim.example.ipynb` running end to end

## Project 2: Topic Modeling of FOMC Statements and the Next Rate Decision

- **Project Objective**: Find the main topics of the Federal Open Market Committee
  (FOMC) statements, and check if the topic weights predict the next meeting's rate
  decision (hike, hold, or cut) better than a persistence baseline
- **Dataset Suggestions**:
  - [FOMC statements](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm),
    about 120 scheduled meetings from 2010 to 2025, linked from the Fed calendar page
    and from the yearly historical pages
  - [Federal Funds Target Range, Upper Limit](https://fred.stlouisfed.org/series/DFEDTARU)
    to label the rate decisions
- **Tasks**:
  - **Collect the Statements**: Download the statement of each scheduled meeting with
    `requests` and `BeautifulSoup`, and remove the paragraph that lists the voters
  - **Preprocess the Text**: Lowercase, tokenize, and remove the stop words, then
    build a `Dictionary` with `filter_extremes` and a bag-of-words corpus
  - **Fit the Topic Models**: Fit `LdaModel` for 3 to 10 topics with a fixed
    `random_state`, and choose the number of topics with the `CoherenceModel` c_v
    score
  - **Predict the Next Decision**: Label each meeting with the target change at the
    next meeting, then fit logistic regressions on the topic weights and on
    `TfidfModel` features, training only on meetings before each test year (2016 on)
  - **Evaluate the Classifiers**: Compare accuracy, macro-F1, and the recall of hikes
    and cuts against the always-hold and the repeat-last-decision baselines
  - **Interpret the Topics**: Name each topic from its top words, and plot the topic
    weights over time with the hiking and cutting cycles shaded
- **Bonus Ideas (Optional)**: Compare `LdaModel` with the `Nmf` model of Gensim; use
  `MatrixSimilarity` to measure how much each statement differs from the previous
  one and relate it to rate changes; train a `Word2Vec` classifier on the hawkish
  and dovish sentences of the
  [Trillion Dollar Words](https://huggingface.co/datasets/gtfintechlab/fomc_communication)
  dataset

## Project 3: Similar-Company Search from 10-K Risk Factors

- **Project Objective**: Retrieve the companies most similar to a given company from
  the Risk Factors section of their 10-K filings, and test if the neighbors share the
  industry and move together in the market more than random pairs, comparing three
  text representations
- **Dataset Suggestions**:
  - [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus), annual 10-K
    reports split by item, using the `section_1A` field of the filings of one year
    (stream a sample of about 2,000 filings)
  - [SEC EDGAR submissions API](https://www.sec.gov/edgar/sec-api-documentation) for
    the SIC code and the ticker of each company, with a `User-Agent` header
  - Daily prices from [yfinance](https://pypi.org/project/yfinance/)
- **Tasks**:
  - **Extract the Risk Factors**: Keep the filings with at least 500 words of
    `section_1A`, then clean and tokenize them with `simple_preprocess`
  - **Add the Industry Labels**: Get the SIC code of each company from the submissions
    API, and group the companies by 2-digit SIC major group
  - **Train the Embeddings**: Train `Word2Vec` and `FastText` on the risk factors, and
    compare the `most_similar` words of risk terms such as `inflation` and
    `cyberattack`, including a misspelled word
  - **Build the Similarity Indexes**: Represent each company with `TfidfModel`
    vectors as the baseline, and with the average `Word2Vec` and `FastText` vectors,
    then index each one with `MatrixSimilarity`
  - **Evaluate the Retrieval**: Report precision@10 of the same SIC group for each
    representation, with a bootstrap 95% confidence interval over the companies
  - **Test the Economic Meaning**: Compare the mean correlation of daily returns
    between each company and its top-10 neighbors with the one of random pairs, using
    the returns of the calendar year after the filings and skipping delisted tickers
- **Bonus Ideas (Optional)**: Cluster the companies with K-means and compare the
  clusters with the SIC groups using the adjusted Rand index; compare with SBert
  sentence embeddings; measure the year-over-year change of the risk-factor vector of
  each company
