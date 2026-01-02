## Utility Modules

| File                           | Description                                                                                                                                                                    | Key Functions                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| `utils_data_io.py`             | Loads the India housing CSV and normalizes blank values to `NaN`.                                                                                                              | `load_housing_data(data_path)`                                  |
| `utils_feature_engineering.py` | Groups columns by type and converts Yes/No flags to 1/0.                                                                                                                       | `get_column_groups(X)`, `encode_binary_columns(X, binary_cols)` |
| `utils_transformers.py`        | Custom scikit-learn transformer that expands the pipe-delimited `Amenities` field into binary features.                                                                        | `AmenitiesEncoder`                                              |
| `utils_preprocessing.py`       | Builds the full preprocessing pipeline (numeric scaling, categorical encoding, amenity expansion), performs train/test split, and returns processed arrays plus feature names. | `create_preprocessor(column_groups)`, `prepare_data(...)`       |
| `auto_sklearn_utils.py`        | Convenience facade re-exporting all the functions/classes above so notebooks can simply `from auto_sklearn_utils import prepare_data`.                                         | `prepare_data`, `load_housing_data`, etc.                       |

Use `prepare_data("data/raw/india_housing_prices.csv")` to get ready to train matrices and targets for your models.
