# Example: Using the Data I/O and Post-Processing API

This document presents a complete, end-to-end example of how the project’s API
layer can be used to load a cleaned Sentiment140 dataset, compute additional
text-based features in Python, and perform basic exploratory analysis.

The corresponding executable notebook is provided in:
- `OpenAPI.example.ipynb`

---

## Load the Cleaned Dataset

The cleaned dataset is loaded using the `load_clean_csv` function from
`utils_data_io.py`.

```python
import pandas as pd
from utils_data_io import load_clean_csv

df = load_clean_csv("Sentiment140_raw_fixed.csv")
```

This returns a Pandas DataFrame containing cleaned tweet text and sentiment labels.


## Add Text-Based Features

Additional text statistics are computed using the add_counts function from
utils_post_processing.py. These features are derived from the text_clean
column.

```python
from utils_post_processing import add_counts

df = add_counts(df, text_col="text_clean")
```

This step adds Python-computed text features such as:

- `word_count_py`
- `char_count_py`

## Compute Aggregate Statistics

To demonstrate downstream usage of the API, the dataset is grouped by sentiment
label and the average word and character counts are computed.

```python
df.groupby("target")[["word_count_py", "char_count_py"]].mean().round(2)
```
This produces a concise summary of average tweet length by sentiment category.

## Conclusion

This example demonstrates how the project’s API layer supports a clean and
reusable workflow for loading data, computing derived features, and performing
basic analysis. The same API functions are reused throughout the project’s
modeling notebooks.