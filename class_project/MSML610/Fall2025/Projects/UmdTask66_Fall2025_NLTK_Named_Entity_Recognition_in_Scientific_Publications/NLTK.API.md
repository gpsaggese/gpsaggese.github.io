# NLTK for Named Entity Recognition (NER)

## 1. Introduction to NLTK

**NLTK (Natural Language Toolkit)** is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources (such as WordNet), along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

In this project, we utilize NLTK to perform **Named Entity Recognition (NER)** on scientific publications from the CORD-19 dataset.

### Key Features Used in This Project:
*   **Tokenization:** Splitting text into sentences and words.
*   **POS Tagging:** Identifying grammatical parts of speech (Nouns, Verbs, etc.).
*   **Chunking:** Grouping words into meaningful phrases (Named Entities).

---

## 2. The NER Pipeline with NLTK

Named Entity Recognition in NLTK follows a standard three-step pipeline. Below is a detailed explanation of each step using the logic implemented in our `NLTK_utils.py`.

### Step 1: Tokenization (`nltk.word_tokenize`)

Before a computer can understand the grammatical structure of text, the raw string must be broken down into individual units called **tokens** (words and punctuation).

*   **Function:** `nltk.word_tokenize(text)`
*   **Input:** A raw string (e.g., "Dr. Fauci works at NIAID.")
*   **Output:** A list of strings (e.g., `['Dr.', 'Fauci', 'works', 'at', 'NIAID', '.']`)

```python
import nltk

text = "The study was conducted at the University of Maryland."
tokens = nltk.word_tokenize(text)
# Result: ['The', 'study', 'was', 'conducted', 'at', 'the', 'University', 'of', 'Maryland', '.']
```
### Step 2: Part-of-Speech (POS) Tagging (nltk.pos_tag)

Once tokenized, we need to understand the grammatical role of each word. NLTK uses a pre-trained model (Averaged Perceptron Tagger) to assign tags like **NN** (Noun), **VB** (Verb), or **NNP** (Proper Noun).

- Function: `nltk.pos_tag`(tokens)
- Input: List of tokens.
- Output: List of tuples (word, tag).

**Why is this important for NER?**

Named Entities are almost always **Proper Nouns** (NNP). The POS tagger helps filter out common words (like "the", "is") so the NER model focuses on the relevant candidates.

```python
pos_tags = nltk.pos_tag(tokens)
# Result: [('The', 'DT'), ('study', 'NN'), ..., ('University', 'NNP'), ('of', 'IN'), ('Maryland', 'NNP'), ('.', '.')]
```

### Step 3: Named Entity Chunking (nltk.ne_chunk)

This is the core NER step. NLTK uses a classifier (MaxEnt) to analyze the POS tags and identify patterns that look like entities. It groups (chunks) adjacent tokens into a Tree structure.

- Function: nltk.ne_chunk(pos_tags)
- Input: List of POS-tagged tuples.
- Output: A nltk.tree.Tree object.

```python
chunks = nltk.ne_chunk(pos_tags)
# Result is a Tree where entities are grouped:
# (S
#   The/DT
#   study/NN
#   was/VBD
#   conducted/VBN
#   at/IN
#   the/DT
#   (ORGANIZATION University/NNP of/IN Maryland/NNP)
#   ./.)
```

### 3. Implementation in NLTK_utils.py
In our project, we wrapped this pipeline into a single function `extract_entities_nltk`. Here is how we process the tree to extract clean entity names.

Code Walkthrough

```python
def extract_entities_nltk(text):
    # 1. Tokenize
    tokens = nltk.word_tokenize(text)
    
    # 2. POS Tag
    pos_tags = nltk.pos_tag(tokens)
    
    # 3. NER Chunking
    chunks = nltk.ne_chunk(pos_tags)
    
    entities = set() # Use a set to avoid duplicates
    
    # 4. Tree Traversal
    for chunk in chunks:
        # If the chunk is a Subtree (has a label like 'PERSON' or 'ORGANIZATION')
        if hasattr(chunk, 'label'):
            # Rejoin the tokens inside the chunk to form the full name
            # e.g., "University" + "of" + "Maryland"
            entity_name = ' '.join(c[0] for c in chunk)
            
            # Normalize to lowercase for consistent comparison
            entities.add(entity_name.lower())
            
    return list(entities)
```

Handling NLTK Dependencies

To run this pipeline, specific NLTK data packages must be downloaded. We handle this in the setup_nltk() function:

- `punkt`: For tokenization.
- `averaged_perceptron_tagger`: For POS tagging.
- `maxent_ne_chunker`: The NER model itself.
- `words`: A corpus of standard words used by the chunker to distinguish common words from names.

### 4. Comparison: NLTK vs. Modern Approaches

While NLTK is excellent for education and baseline systems, we compare it against `spaCy` in this project.

|Feature	|NLTK	|spaCy|
| ---- | ---- | ---- |
|Architecture	|Pipeline of separate steps (Tokenize -> Tag -> Chunk)	|Integrated pipeline with statistical models
|Speed	|Slower (Python-heavy string processing)	|Very Fast (Cython optimized)
|Accuracy	|Good for standard English; relies heavily on capitalization	|State-of-the-art for general purpose NER
|Customization	|Highly customizable at each step	|Easier to train new entity types

In our `main.py`, we treat spaCy's output as the "Silver Standard" to evaluate NLTK's performance using Precision, Recall, and F1-Score.