# Project Implementation: NER on CORD-19 Dataset

This document details the implementation of a Named Entity Recognition (NER) pipeline applied to the CORD-19 scientific dataset. The project utilizes **NLTK** as the primary research tool and compares its performance against a pre-trained **spaCy** model (serving as a "Silver Standard" baseline).

The implementation is orchestrated in `NLTK.example.ipynb` and relies on helper functions defined in `NLTK_utils.py`.

---

## 1. Data Acquisition
**Goal:** Download and manage the massive CORD-19 dataset (approx. 19GB) efficiently.

The dataset is hosted on AWS S3. Since the file is too large to load into memory at once, we implemented a robust download mechanism supporting resumable downloads and streaming processing.

### Implementation Details
*   **Source:** `https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2022-06-02.tar.gz`
*   **Streaming:** We use Python's `tarfile` library in stream mode (`'r|gz'`) to read the archive sequentially without extracting the entire 19GB to disk.
*   **Generator Pattern:** The function `stream_cord19_data` yields one paper at a time, allowing us to process an infinite number of papers with constant memory usage.

```python
# From NLTK.example.ipynb
CORD19_URL = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2022-06-02.tar.gz"
download_dataset(CORD19_URL, CORD19_FILE_PATH)

# We process papers in a loop using a generator
paper_generator = stream_cord19_data(CORD19_FILE_PATH, limit=MAX_NUMBER)
for paper in paper_generator:
    # Process paper...
```

## 2. Text Cleaning

**Goal:** Remove noise from the scientific text to improve NER accuracy.

Scientific papers contain citations (e.g., [12], (Smith et al.)) and formatting artifacts that can confuse NER models. For example, a model might mistake a citation inside brackets for a noun phrase.

### Implementation Details

The `clean_text` function (in `NLTK_utils.py`) performs the following preprocessing:

1. **Remove Citations:** Uses Regex to strip brackets and their contents (e.g., `\[.*?\]`).
2. **Whitespace Normalization:** Replaces multiple spaces or newlines with a single space.
3. **Strip Artifacts:** Removes leading/trailing whitespace.

```python
# From NLTK.example.ipynb
cleaned_text = clean_text(paper['text'])
# We truncate text to MAX_LENGTH chars for faster evaluation in this demo
eval_text = cleaned_text[:MAX_LENGTH]
```

## 3. Tokenization and Tagging (NLTK Pipeline)

**Goal:** Prepare the text for entity chunking by breaking it down and understanding grammatical structure.

This is the foundation of the NLTK approach. Unlike modern end-to-end transformers, NLTK relies on a pipeline of specific linguistic steps.

### Implementation Details (`extract_entities_nltk`)

1. **Tokenization:** `nltk.word_tokenize` splits the cleaned string into a list of words and punctuation.
2. **POS Tagging:** `nltk.pos_tag` assigns a Part-of-Speech tag to every token.
    - Crucial Step: NER relies heavily on tags like **NNP** (Proper Noun) to identify potential entities.
  
```python
# Conceptual logic inside NLTK_utils.py
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
# Output example: [('Dr.', 'NNP'), ('Fauci', 'NNP'), ('said', 'VBD')...]
```

## 4. Entity Recognition: NLTK vs. spaCy

**Goal:** Extract named entities (Organizations, Persons, Locations, GPEs) and compare two different approaches.

Approach A: NLTK (Rule-Based/Statistical Chunking)

We use `nltk.ne_chunk`, which takes the POS tags and builds a Tree structure. We then traverse this tree to find subtrees that have entity labels.

- Pros: Highly transparent, easy to customize rules.
- Cons: Slower, relies heavily on capitalization, less accurate on complex scientific terms.
- 
Approach B: spaCy (Pre-trained Statistical Model)

We use spaCy's `en_core_web_sm model`. This is a modern, convolution-based statistical model that performs tokenization, tagging, and NER in a single optimized pass.

- Role: Since the CORD-19 dataset is unlabeled, we treat spaCy's output as the "Silver Standard" (Ground Truth) to evaluate NLTK.

```python
# From NLTK.example.ipynb
# 1. Run NLTK extraction
ents_nltk = extract_entities_nltk(eval_text)

# 2. Run spaCy extraction (Baseline)
ents_spacy = extract_entities_spacy(eval_text, nlp_spacy)
```

## 5. Evaluation

**Goal:** Quantify the performance of the NLTK system using standard classification metrics.

We calculate metrics by comparing the set of entities found by NLTK against the set found by spaCy.

Metrics Used

- **Precision:** Of the entities NLTK found, how many were also found by spaCy?
    * `Intersection / Total NLTK Entities`
- **Recall:** Of the entities spaCy found, how many did NLTK manage to find?
    * `Intersection / Total spaCy Entities`
- **F1-Score:** The harmonic mean of Precision and Recall.

### Implementation Details

The calculate_metrics function handles the set logic:

```python
# Logic inside calculate_metrics
reference = set(reference_entities) # spaCy
candidate = set(candidate_entities) # NLTK

true_positives = len(candidate.intersection(reference))
precision = true_positives / len(candidate)
recall = true_positives / len(reference)
f1 = 2 * (precision * recall) / (precision + recall)
```

Output

The results are aggregated into a Pandas DataFrame and saved to CSV files for analysis:

1. `extracted_entities.csv`: A raw list of what each model found.
2. `performance_metrics.csv`: The F1 scores for each paper processed.

```python
# From NLTK.example.ipynb
print(df_perf[['precision', 'recall', 'f1_score']].mean())
```