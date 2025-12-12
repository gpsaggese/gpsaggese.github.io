# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NLTK API Reference Guide
#
# This notebook serves as a concise reference for the NLTK (Natural Language Toolkit) functions used in our Named Entity Recognition (NER) project. We will explore the core pipeline: Tokenization -> POS Tagging -> Named Entity Chunking.
#
# ## 1. Setup
#
# First, ensure NLTK and the required data packages are installed.

# %%
import nltk

# Download necessary resources
required_packages = [
    'punkt', 
    'averaged_perceptron_tagger', 
    'maxent_ne_chunker', 
    'words',
    'punkt_tab',
    'maxent_ne_chunker_tab'
]

for package in required_packages:
    nltk.download(package, quiet=True)

print("NLTK setup complete.")

# %% [markdown]
# ## 2. Tokenization: nltk.word_tokenize
#
# Function: Splits a text string into individual words and punctuation marks (tokens). This is the first step in most NLP pipelines.
#
# Usage:

# %%
text = "Dr. Smith works at the University of Maryland."
tokens = nltk.word_tokenize(text)

print(f"Original: {text}")
print(f"Tokens:   {tokens}")

# %% [markdown]
# ## 3. Part-of-Speech (POS) Tagging: nltk.pos_tag
#
# Function: Assigns a grammatical category (tag) to each token, such as Noun (NN), Verb (VB), or Proper Noun (NNP). These tags are crucial for identifying entities.
#
# Common Tags:
#
# - NNP: Proper Noun, Singular (e.g., "Maryland")
# - DT: Determiner (e.g., "the")
# - IN: Preposition (e.g., "at")
#
# Usage:

# %%
# # download resources
# nltk.download('averaged_perceptron_tagger_eng')

# Requires a list of tokens as input
pos_tags = nltk.pos_tag(tokens)

print("POS Tags:")
for word, tag in pos_tags:
    print(f"{word}: {tag}")

# %% [markdown]
# ## 4. Named Entity Chunking: nltk.ne_chunk
#
# Function: Takes a list of POS-tagged tokens and groups them into "chunks" that represent named entities (like Persons, Organizations, Locations). It returns a Tree structure.
#
# Usage:

# %%
# Requires POS tags as input
chunks = nltk.ne_chunk(pos_tags)

print("Chunk Tree Structure:")
print(chunks)

# %% [markdown]
# ## 5. Extracting Entities from the Tree
#
# The output of `ne_chunk` is a tree. To get the actual entity names, we need to traverse this tree and look for subtrees that have a label (like 'PERSON' or 'ORGANIZATION').
#
# Usage:

# %%
entities = []

for chunk in chunks:
    # Check if the chunk is a named entity (has a label)
    if hasattr(chunk, 'label'):
        label = chunk.label()
        # Rejoin the tokens to form the full entity name
        entity_name = ' '.join(c[0] for c in chunk)
        entities.append((entity_name, label))

print("Extracted Entities:")
for name, label in entities:
    print(f"Entity: {name} | Type: {label}")

# %% [markdown]
# ## Summary of the Pipeline
# 1. **Input**: Raw Text String
# 2. `word_tokenize` -> List of Strings (Tokens)
# 3. `pos_tag` -> List of Tuples (Word, Tag)
# 4. `ne_chunk` -> Tree Object
# 5. `Traversal` -> List of Entities
