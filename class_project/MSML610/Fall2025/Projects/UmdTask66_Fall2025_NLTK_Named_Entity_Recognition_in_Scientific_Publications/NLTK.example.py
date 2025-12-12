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
# # NER on CORD-19 Dataset

# %% [markdown]
# ## 1. Import necessary modules and set the environment
#
# `setup_nltk` function checks and downloads all necessary NLTK modules.

# %%
# # download necessary packages
# # %pip install nltk
# # %pip install spacy

# %%
import json
import re
import nltk
import spacy
import tarfile
import os
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from NLTK_utils import download_dataset, setup_nltk, stream_cord19_data, clean_text, extract_entities_nltk, extract_entities_spacy, calculate_metrics

# PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = "data/"
CORD19_FILENAME = 'cord-19_2022-06-02.tar.gz'
CORD19_FILE_PATH = DATA_DIR + CORD19_FILENAME
CORD19_URL = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2022-06-02.tar.gz"
OUTPUT_DIR = "output/"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

setup_nltk()
# download spacy model
print("Downloading spaCy model...")
spacy.cli.download("en_core_web_sm")

# %% [markdown]
# ## 2. Download the dataset
#
# Since the whole dataset is very big, we implement a reliable download function supporting resumable downloads and streaming processing. The definition of this function is:
# ```python
# download_dataset(url, dest_path, chunk_size=1024*1024)
# ```
#
# Parameters:
# - `url`: the link to download the dataset.
# - `dest_path`: the path where dataset is to be stored. Note that the dataset is a compressed file, and therefore the `dest_path` is a compressed folder ending with `.tar.gz`. We use a stream processor `stream_cord19_data` to read this compressed file without uncompressing.

# %%
# 2. download dataset
download_dataset(CORD19_URL, CORD19_FILE_PATH)

# %% [markdown]
# ## 3. Hyperparameters configuration
#
# There are 2 hyperparameters:
# - **MAX_NUMBER:** indicates the total number of articles that are to be analyzed. The dataset is very big and contains thousands of articles. Therefore, it is nearly impossible to analyze the whole dataset and we only choose the first **MAX_NUMBER** articles to analyze.
# - **MAX_LENGTH:** indicates where to truncate the `cleaned_text`. Because the main texts of published articles are usually long, it is a useful trick to only analyze the first **MAX_LENGTH** cleaned characters in order to get a quicker demonstration. Besides, if **MAX_LENGTH** is set to `None`, our analysis code would analyze the whole text.
#
# You can configure the parameters in the below cell. 

# %%
MAX_NUMBER = 10
MAX_LENGTH = 2000


# %% [markdown]
# ## 4. perform analysis
#
# The whole analysis process is integrated in the function `analyze` in the below cell. It invokes 2 separate entities extraction function and 1 performance analysis function. 
#
# ### Entities extraction
#
# The entities extraction function of NLTK is defined as:
# ```python
# extract_entities_nltk(text)
# ```
#
# The only parameter is the input text.
#
# The entities extraction function of spaCy is defined as:
# ```python
# extract_entities_spacy(text, nlp_model)
# ```
# There are two parameters: `text` for the input text; and `nlp_model` for the spaCy model.
#
# Both functions above return a list of entities.
#
# ### Performance analysis
#
# The performance analysis uses relative performance to assess the results of NLTK. Since the dataset is unlabeled, there is no ground truth to evaluate the accuracy, recall and F-1 score of NLTK methods. Therefore, we regard the mature spaCy method as the standard, and calculate the relative performance of NLTK to spaCy. The definition of the performance analysis function is: 
# ```python
# calculate_metrics(reference_entities, candidate_entities)
# ```
#
# - `reference_entities`: the result list of entities by the standard method that we choose, which is spaCy.
# - `candidate_entities`: the result list of entities by the method that we are to assess, which is NLTK.
#
# This function returns a dictionary that contains all performance metrics, of which the shape is:
# ```python
# {
#     "precision": round(precision, 4),
#     "recall": round(recall, 4),
#     "f1_score": round(f1, 4),
#     "overlap_count": tp,
#     "nltk_only_count": fp,
#     "spacy_only_count": fn
# }
# ```
#
# Note that the entities extracted per paper and paper-wise performance analysis are stored in `output/` as `.csv` files, in which more details can be inspected.

# %%
def analyze():
    # initialize models
    print("Initializing models...")
    
    # load spaCy
    try:
        nlp_spacy = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
        return

    # list to store results
    all_entities_data = []
    performance_metrics = []

    # process the first MAX_NUMBER papers
    MAX_NUMBER = 10
    print(f"\n--- Processing {MAX_NUMBER} papers from CORD-19 ---")
    
    paper_generator = stream_cord19_data(CORD19_FILE_PATH, limit=MAX_NUMBER)
    
    for i, paper in enumerate(paper_generator):
        paper_id = paper['id']
        print(f"Processing [{i+1}/{MAX_NUMBER}]: {paper_id}")
        
        cleaned_text = clean_text(paper['text'])
        
        # 1. run the models
        # notice: in order for fair comparison, we apply same truncation for different methods.
        # for quick demonstration, we use the first MAX_LENGTH characters 
        if MAX_LENGTH:
            eval_text = cleaned_text[:MAX_LENGTH]
        else:
            eval_text = cleaned_text
        
        ents_nltk = extract_entities_nltk(eval_text)
        ents_spacy = extract_entities_spacy(eval_text, nlp_spacy)
        
        # 2. store the output for following analysis
        for ent in ents_nltk:
            all_entities_data.append({'paper_id': paper_id, 'model': 'NLTK', 'entity': ent})
        for ent in ents_spacy:
            all_entities_data.append({'paper_id': paper_id, 'model': 'spaCy', 'entity': ent})
            
        # 3. compute the performance (use spaCy as the Silver Standard)
        metrics = calculate_metrics(reference_entities=ents_spacy, candidate_entities=ents_nltk)
        metrics['paper_id'] = paper_id
        performance_metrics.append(metrics)

    # ---------------------------------------------------------
    # store and display the results
    # ---------------------------------------------------------
    
    # store lists of entities
    df_entities = pd.DataFrame(all_entities_data)
    entities_csv_path = OUTPUT_DIR + "extracted_entities.csv"
    df_entities.to_csv(entities_csv_path, index=False)
    print(f"\n[Saved] All extracted entities saved to: {entities_csv_path}")
    
    # store performances
    df_perf = pd.DataFrame(performance_metrics)
    perf_csv_path = OUTPUT_DIR + "performance_metrics.csv"
    df_perf.to_csv(perf_csv_path, index=False)
    print(f"[Saved] Performance metrics saved to: {perf_csv_path}")
    
    # print average performances
    if not df_perf.empty:
        print("\n" + "="*40)
        print("Average Performance (NLTK vs spaCy as Baseline)")
        print("="*40)
        print(df_perf[['precision', 'recall', 'f1_score']].mean())
        print("="*40)
        print("Note: Since CORD-19 is unlabeled, we treat spaCy results as the")
        print("'Silver Standard' (Ground Truth) to evaluate NLTK's relative performance.")

# run the analysis
analyze()
