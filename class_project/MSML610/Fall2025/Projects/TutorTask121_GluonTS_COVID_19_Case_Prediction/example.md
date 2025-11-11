# COVID Case Prediction — Example

This document highlights a short end-to-end example of how to fetch the data and inspect it with the utility helpers. Note that data used is stored and retireved from a shared Google Drive.

Example (summary):

1. Download and load data

```python
from utils_data_io import download_and_load_all

gdrive_urls = {
    "mobility": "https://drive.google.com/open?id=1TMqG8Z8vbxmQAv1rNKczYYPCzwT4ZS_q",
    "cases":    "https://drive.google.com/open?id=1ZfZtoV3PpZblZYES0A5LHCwp54cR8RJL",
    "deaths":   "https://drive.google.com/open?id=1kYC9nrCnKbNpnoZKz8o6TDMM371gyxbl",
    "vaccine":  "https://drive.google.com/open?id=1ulTFLBbZxz636_PFqQvixLpqQV9s-P_v",
}

dfs = download_and_load_all(gdrive_urls, data_dir="data")
print(dfs["cases"].shape)
```

2. Use `utils_post_processing.summarize` to get a quick sanity summary.
