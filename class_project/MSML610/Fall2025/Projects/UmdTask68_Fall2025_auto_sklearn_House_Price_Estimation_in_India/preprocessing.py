import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

DATA_PATH = "/Users/ritvik/src/umd_classes1/class_project/MSML610/Fall2025/Projects/UmdTask68_Fall2025_auto_sklearn_House_Price_Estimation_in_India/data/raw/india_housing_prices.csv"

# custom transformer
# converts the pipe delimited amenities text field into one hot encoded columns
class AmenitiesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        # scikitlearn helper that one hot encodes lists of tokens
        self.binarizer = MultiLabelBinarizer()

    def fit(self, X, y=None):
        # learns the set of amenities that occur in the training data
        amenities_lists = self._split(X)
        self.binarizer.fit(amenities_lists)
        return self

    def transform(self, X):
        # produces a binary matrix indicating if each amenity exists or not
        amenities_lists = self._split(X)
        return self.binarizer.transform(amenities_lists)

    def get_feature_names_out(self, input_features=None):
        # provides readable column names 
        return [f"amenity__{a}" for a in self.binarizer.classes_]

    @staticmethod
    def _split(series):
        # splits csv into token lists + handles empty cells
        return series.fillna("").apply(
            lambda x: [token.strip() for token in x.split(",") if token.strip()]
        )

# load and print raw data
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Initial shape: {df.shape}")

# harmoinize empty string to NaN so they can be detected all at once
df = df.replace(r"^\s*$", pd.NA, regex=True)
print("Replaced blank strings with NA.")
print(df.head())

# seperate features and target
# dropping ID bc its surrogate key
target = "Price_in_Lakhs"
drop_cols = ["ID", target]
X = df.drop(columns=drop_cols)
y = df[target]

print("\nTarget summary:")
print(y.describe())

# listing all column groups for preprocessing purposes
numeric_cols = [
    "BHK", "Size_in_SqFt", "Price_per_SqFt", "Year_Built",
    "Floor_No", "Total_Floors", "Age_of_Property",
    "Nearby_Schools", "Nearby_Hospitals"
]
binary_cols = ["Parking_Space", "Security"]
text_cols = ["Amenities"]
categorical_cols = list(
    set(X.columns) - set(numeric_cols) - set(binary_cols) - set(text_cols)
)

print("\nColumn groups:")
print(f"Numeric: {numeric_cols}")
print(f"Binary: {binary_cols}")
print(f"Categorical: {categorical_cols}")
print(f"Text: {text_cols}")

print("\nMissing values (% of rows):")
print(df.isna().mean().sort_values(ascending=False).head(15))

# mapping yes/no to numeric 1/0 so they can be converted into a numeric col
for col in binary_cols:
    X[col] = X[col].map({"Yes": 1, "No": 0})

print("\nBinary column sample after mapping:")
print(X[binary_cols].head())

# main preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # num cols: fill  missing with median + scale diff unscaled numeric values
        ("num", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler())
        ]), numeric_cols),
        # categorical cols: fill missing with most frequent and one hot encode
        ("cat", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols),
        # amentites fields: expand into multiple binary cols
        ("amenities", Pipeline(steps=[
            ("bin", AmenitiesEncoder())
        ]), "Amenities"),
    ],
    remainder="drop",                   # discard any cols not handled
    verbose_feature_names_out=False,    # keeping names of features clean
)

# splitting into training and testing datasets
# then transform the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nSplit data:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

print("\nFitting preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)
print("Transforming hold-out set...")
X_test_processed = preprocessor.transform(X_test)

# printing transformed data
print("\nProcessed shapes:")
print(f"X_train_processed: {X_train_processed.shape}")
print(f"X_test_processed: {X_test_processed.shape}")

feature_names = preprocessor.get_feature_names_out()
print(f"Total engineered features: {len(feature_names)}")
print("First 20 feature names:")
print(feature_names[:20])

print("\nSample rows from processed train matrix:")
sample_idx = np.random.choice(X_train_processed.shape[0], size=5, replace=False)
print(X_train_processed[sample_idx])

print("\nPreprocessing complete.")