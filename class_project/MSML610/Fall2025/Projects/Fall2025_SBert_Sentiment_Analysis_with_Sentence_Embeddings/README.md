# SBERT Sentiment Analysis — Fall 2025 (MSML610)

This project sets up the preprocessing and embedding pipeline for sentiment analysis using the Financial PhraseBank dataset and Sentence-BERT (SBERT).

## Current Components

### 1. `preprocess.py`
- Loads raw CSV.
- Cleans text.
- Maps sentiment labels to integers.
- Saves cleaned CSV and label array.

Run:
```bash
python src/preprocess.py --config config.yaml
```

### 2. `sbert_embed.py`
	•	Loads cleaned CSV.
	•	Encodes sentences using SBERT (all-MiniLM-L6-v2).
	•	Saves embeddings as .npy.

Run: 
```bash
python src/sbert_embed.py --config config.yaml
```

### 3. utils.py
	•	Contains small shared helper functions (e.g., config loading).
	•	Created to follow MSML610 project conventions.

4. Folder Structure (Current)
project/
├── README.md
├── config.yaml
├── utils.py
├── src/
│   ├── preprocess.py
│   └── sbert_embed.py
├── data/
│   ├── raw/        (generated locally)
│   └── processed/  (generated locally)

## Notes
	•	No data files are committed to the repository.
	•	All outputs in data/processed/ are generated locally.
	•	More files (API, example notebooks, Dockerfile, etc.) will be added in future PRs.



