# SBERT API Documentation

This document describes the internal API used by the SBERT Sentiment Analysis project.  
It covers:  
1.	Native programming interface — preprocessing, embedding, and utility functions.  
2.	Wrapper layer — standardized functions that simplify notebook / script usage.  
3.	Example usage — minimal code snippets showing how each component is used.  
4.	Design decisions — why the API is structured this way.   

## 1. Overview

The project uses the Financial PhraseBank dataset and applies the following pipeline: 

1.	Preprocessing → Clean text, normalize labels, save clean.csv + labels.npy  
2.	Embedding → Generate SBERT embeddings using all-MiniLM-L6-v2  
3.	Modeling → Downstream models operate on the generated embeddings  
4.	Fine-tuning → Train SBERT end-to-end for better task accuracy  

All reusable logic lives inside src/ so notebooks remain clean and minimal.

## 2. Native API

### 2.1 load_config(path: str) → dict

Loads YAML configuration controlling paths, SBERT model name, and preprocessing fields.

Example
```python
from SBERT_utils import load_config
cfg = load_config("config.yaml")
```

### 2.2 load_clean_data(cfg) → pandas.DataFrame

Loads cleaned CSV created by preprocess.py.
	•	ensures consistent schema
	•	returns dataframe with text + integer labels

Example
```pyhton
df = load_clean_data(cfg)
print(df.shape)
```

### 2.3 load_embeddings(cfg) → np.ndarray

Loads precomputed SBERT embeddings from data/processed/sbert_embeddings.npy.

Example
```python
embeddings = load_embeddings(cfg)
```

## 3. Wrapper Layer

The wrapper layer simply standardizes how the notebooks load resources:
	•	load_config
	•	load_clean_data
	•	load_embeddings

This keeps notebooks clean and prevents duplicate code.
All notebooks import the API through:
```python
from src.SBERT_utils import load_config, load_clean_data, load_embeddings
```

## 4. API Usage Examples

### Example 1: Load data + embeddings
```python
cfg = load_config("config.yaml")
df = load_clean_data(cfg)
X = load_embeddings(cfg)

print(df.head())
print(X.shape)
```
### Example 2: Use in a modeling script
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, df["sentiment_int"].values, test_size=0.2, stratify=df["sentiment_int"]
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
```
## 5. Architectural Decisions

- Separation of preprocessing, embedding, and modeling
This avoids re-running expensive embedding steps when experimenting.

- All heavy logic moved out of notebooks
This makes notebooks easy to grade and read.

- Consistent config-driven design
Paths, model names, and columns are never hard-coded.

- Embeddings saved as NumPy arrays
Fast to load, memory-efficient, widely compatible.

## 7. Limitations & Notes
- The API is intentionally lightweight (not a package).
- Fine-tuning is demonstrated in notebooks, not wrapped as an API call.
- The dataset must be placed correctly before running preprocessing.

## 8. Conclusion

This API provides a clean, reproducible interface for SBERT-based classification experiments.
All notebooks depend only on this utility layer, keeping the project maintainable and extensible.
