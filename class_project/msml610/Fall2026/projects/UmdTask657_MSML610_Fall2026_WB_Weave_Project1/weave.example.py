# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tracing and Evaluating Hawkish-Dovish Models for FOMC Text
#
# This tutorial demonstrates how to use W&B Weave to trace,
# evaluate, and compare machine learning models that classify
# Federal Reserve communications as hawkish, dovish, or neutral.
#
# We also investigate the relationship between FOMC statement
# tone and changes in the 2-year Treasury yield.

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# System libraries.
import logging

# Third-party libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# # To install additional packages, use:
# import helpers.hmodule as hmodule
# hmodule.install_module_if_not_present(
#     ["pycaret"],
#     use_activate=True,
#     use_sudo=False,
#     venv_path="/opt/venv",
# )

# %%
# Use this for most notebooks.
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebook

_LOG = logging.getLogger(__name__)

# Initialize notebook configuration and logging.
hdbg.init_logger(verbosity=logging.INFO)
hnotebook.config_notebook()

# Convert `display` into `print()`.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore


# %%
from pathlib import Path
import pandas as pd
import requests

# Jupyter notebook's current working directory.
PROJECT_DIR = Path.cwd()

# Directories for caching downloaded data.
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Project directory:", PROJECT_DIR)
print("Raw data directory:", RAW_DIR)
print("Processed data directory:", PROCESSED_DIR)

# %% [markdown]
# ## Data Ingestion and Caching Labeled FOMC Hawkish/Dovish Sentences from Hugging Face

# %%
BASE_URL = (
    "https://huggingface.co/datasets/"
    "gtfintechlab/fomc_communication/resolve/main"
)

def download_csv(filename):
    destination = RAW_DIR / filename

    if destination.exists():
        print(f"Using cached file: {filename}")
    else:
        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {filename}...")

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Write the downloaded file to the cache.
        destination.write_bytes(response.content)
        print(f"Saved: {destination}")

    return pd.read_csv(destination)


train_df = download_csv("train.csv")
test_df = download_csv("test.csv")

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

# %%
print("TRAINING DATA")
display(train_df.head())

print("\nTESTING DATA")
display(test_df.head())

print("\nCOLUMN NAMES")
print(train_df.columns.tolist())

print("\nMISSING VALUES")
print(train_df.isnull().sum())

# %%
# Check the number of sentences in each dataset.
print("Training sentences:", len(train_df))
print("Testing sentences:", len(test_df))

# Check the available labels.
print("\nTraining label distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nTesting label distribution:")
print(test_df["label"].value_counts().sort_index())

# Inspect sample sentences for each label.
for label in sorted(train_df["label"].unique()):
    print(f"\n--- Label {label} ---")
    samples = train_df[
        train_df["label"] == label
    ]["sentence"].head(3)

    for sentence in samples:
        print("-", sentence)

# %%
# Preserve the original datasets.
train_clean = train_df.copy()
test_clean = test_df.copy()

# Keep the relevant columns.
columns = ["sentence", "year", "label"]

train_clean = train_clean[columns]
test_clean = test_clean[columns]

# Remove leading/trailing whitespace.
train_clean["sentence"] = (
    train_clean["sentence"].astype(str).str.strip()
)

test_clean["sentence"] = (
    test_clean["sentence"].astype(str).str.strip()
)

# Check for empty sentences.
assert train_clean["sentence"].ne("").all()
assert test_clean["sentence"].ne("").all()

# Verify the labels are valid numeric classes.
assert set(train_clean["label"].unique()).issubset({0, 1, 2})
assert set(test_clean["label"].unique()).issubset({0, 1, 2})

# Cache the processed datasets.
train_clean.to_csv(
    PROCESSED_DIR / "train_clean.csv",
    index=False
)

test_clean.to_csv(
    PROCESSED_DIR / "test_clean.csv",
    index=False
)

print("Processed datasets saved successfully!")
print("Training shape:", train_clean.shape)
print("Testing shape:", test_clean.shape)

# %%
print("Processed data files:")
for file in PROCESSED_DIR.glob("*.csv"):
    print(file.name, "-", file.stat().st_size, "bytes")

print("\nCleaned training data:")
display(train_clean.head())

print("\nCleaned testing data:")
display(test_clean.head())

# %%
# Verify the processed files exist.
train_path = PROCESSED_DIR / "train_clean.csv"
test_path = PROCESSED_DIR / "test_clean.csv"

assert train_path.exists(), "Training file missing"
assert test_path.exists(), "Testing file missing"

# Reload the cached datasets.
train_cached = pd.read_csv(train_path)
test_cached = pd.read_csv(test_path)

# Confirm the saved data matches the originals.
assert len(train_cached) == 1984
assert len(test_cached) == 496

assert train_cached["sentence"].notna().all()
assert test_cached["sentence"].notna().all()

print("Data successfully cleaned and cached!")
print("Training shape:", train_cached.shape)
print("Testing shape:", test_cached.shape)

# %% [markdown]
# ## FOMC Statement Data Ingestion (2015–2024)
#
# This section downloads and caches official Federal Open Market
# Committee monetary policy statements from the Federal Reserve.
#
# The statements will be used to calculate hawkish-dovish tone
# scores and analyze their relationship with changes in the
# 2-year Treasury yield.

# %%
from urllib.parse import urljoin
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_FED_URL = "https://www.federalreserve.gov"

calendar_urls = [
    f"{BASE_FED_URL}/monetarypolicy/fomchistorical{year}.htm"
    for year in range(2015, 2021)
]

calendar_urls.append(
    f"{BASE_FED_URL}/monetarypolicy/fomccalendars.htm"
)

# Match official monetary policy statement URLs.
statement_pattern = re.compile(
    r"/newsevents/pressreleases/monetary(\d{8})a\.htm",
    re.IGNORECASE
)

statement_links = {}

for calendar_url in calendar_urls:
    response = requests.get(calendar_url, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a", href=True):
        full_url = urljoin(calendar_url, link["href"])
        match = statement_pattern.search(full_url)

        if match:
            date = pd.to_datetime(
                match.group(1),
                format="%Y%m%d"
            )

            if 2015 <= date.year <= 2024:
                statement_links[date.strftime("%Y-%m-%d")] = full_url

# Create a table of discovered statements.
fomc_links_df = pd.DataFrame(
    [
        {"date": date, "source_url": url}
        for date, url in statement_links.items()
    ]
)

fomc_links_df = fomc_links_df.sort_values(
    "date"
).reset_index(drop=True)

print("Statements discovered:", len(fomc_links_df))

print("\nStatements by year:")
if not fomc_links_df.empty:
    print(
        pd.to_datetime(fomc_links_df["date"])
        .dt.year.value_counts()
        .sort_index()
    )

display(fomc_links_df.head(10))
display(fomc_links_df.tail(10))

# %%
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Directory for cached FOMC statements.
FOMC_DIR = RAW_DIR / "fomc_statements"
FOMC_DIR.mkdir(parents=True, exist_ok=True)

def download_fomc_statement(date, url):
    """Download, clean, and cache an FOMC statement."""

    cache_path = FOMC_DIR / f"{date}.txt"

    # Reuse cached files when available.
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        if text.strip():
            return text

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.select_one("#article")

    if article is None:
        raise ValueError(f"No article found for {date}")

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in article.find_all("p")
    ]

    raw_text = "\n".join(
        p for p in paragraphs if p
    )

    cleaned_text = clean_fomc_statement(raw_text)

    if len(cleaned_text) < 300:
        raise ValueError(
            f"Statement appears too short: {date}"
        )

    cache_path.write_text(
        cleaned_text,
        encoding="utf-8"
    )

    return cleaned_text


records = []
failed_downloads = []

for row in fomc_links_df.itertuples(index=False):
    try:
        text = download_fomc_statement(
            row.date,
            row.source_url
        )

        records.append({
            "date": row.date,
            "statement_text": text,
            "source_url": row.source_url
        })

        print(f"Loaded: {row.date}")

        # Avoid sending requests too quickly.
        time.sleep(0.3)

    except Exception as error:
        failed_downloads.append({
            "date": row.date,
            "error": str(error)
        })
        print(f"FAILED: {row.date} - {error}")


fomc_df = pd.DataFrame(
    records,
    columns=["date", "statement_text", "source_url"]
)

print("\nDownload complete!")
print("Statements loaded:", len(fomc_df))
print("Failed downloads:", len(failed_downloads))

# %%
# Check statement counts by year.
fomc_df["date"] = pd.to_datetime(fomc_df["date"])

print("Statements by year:")
print(
    fomc_df["date"]
    .dt.year.value_counts()
    .sort_index()
)

# Check missing values and duplicate dates.
print("\nMissing values:")
print(fomc_df.isnull().sum())

print("\nDuplicate dates:")
print(fomc_df["date"].duplicated().sum())

# Inspect statement lengths to identify suspicious records.
fomc_df["text_length"] = (
    fomc_df["statement_text"].str.len()
)

print("\nShortest statements:")
display(
    fomc_df.nsmallest(
        10, "text_length"
    )[["date", "text_length", "statement_text"]]
)

print("\nFailed downloads:")
display(pd.DataFrame(failed_downloads))

# %%
# Identify statements that may not represent FOMC policy decisions.

policy_phrases = [
    "federal open market committee",
    "the committee decided",
    "the committee seeks",
    "target range for the federal funds rate",
    "target for the federal funds rate",
]

def contains_policy_language(text):
    text = text.lower()
    return any(phrase in text for phrase in policy_phrases)

fomc_df["has_policy_language"] = (
    fomc_df["statement_text"]
    .apply(contains_policy_language)
)

review_df = fomc_df[
    ~fomc_df["has_policy_language"]
].copy()

print("Announcements requiring review:", len(review_df))

display(
    review_df[
        ["date", "text_length", "source_url"]
    ]
)


# %%
def fix_text_encoding(text):
    """Repair common UTF-8 text decoding artifacts."""
    if not isinstance(text, str):
        return text

    if "â" in text or "Ã" in text:
        try:
            text = text.encode("latin-1").decode("utf-8")
        except UnicodeError:
            pass

    return text


fomc_df["statement_text"] = (
    fomc_df["statement_text"]
    .apply(fix_text_encoding)
)

# Check the March 3 statement.
march3 = fomc_df.loc[
    fomc_df["date"] == pd.Timestamp("2020-03-03"),
    "statement_text"
].iloc[0]

print(march3)

# %%
# Keep the original dataframe for auditing.
fomc_all_df = fomc_df.copy()

# Exclude the unrelated FIMA Repo Facility announcement.
excluded_dates = [
    pd.Timestamp("2020-03-31")
]

fomc_filtered_df = fomc_df[
    ~fomc_df["date"].isin(excluded_dates)
].copy()

fomc_filtered_df = fomc_filtered_df.sort_values(
    "date"
).reset_index(drop=True)

print("Original announcements:", len(fomc_df))
print("Remaining announcements:", len(fomc_filtered_df))
print("Excluded announcements:", len(fomc_df) - len(fomc_filtered_df))

# %%
# Prepare the combined FOMC dataset.
fomc_final_df = fomc_filtered_df[
    ["date", "statement_text", "source_url"]
].copy()

fomc_final_df = fomc_final_df.sort_values(
    "date"
).reset_index(drop=True)

# Verify basic dataset integrity.
assert len(fomc_final_df) == 84
assert fomc_final_df["date"].is_unique
assert fomc_final_df["statement_text"].notna().all()
assert fomc_final_df["statement_text"].str.strip().ne("").all()

# Save to the processed data directory.
fomc_output_path = (
    PROCESSED_DIR / "fomc_statements_2015_2024.csv"
)

fomc_final_df.to_csv(
    fomc_output_path,
    index=False,
    encoding="utf-8"
)

print("FOMC dataset saved successfully!")
print("File:", fomc_output_path)
print("Statements:", len(fomc_final_df))

display(fomc_final_df.head())

# %%
# Reload the saved CSV to verify it.
fomc_cached_df = pd.read_csv(
    fomc_output_path,
    parse_dates=["date"]
)

assert len(fomc_cached_df) == 84
assert fomc_cached_df["date"].is_unique
assert fomc_cached_df["statement_text"].notna().all()

print("FOMC dataset verified successfully!")
print("Shape:", fomc_cached_df.shape)

print("\nStatements by year:")
print(
    fomc_cached_df["date"]
    .dt.year.value_counts()
    .sort_index()
)

# %% [markdown]
# ## Daily 2-Year Treasury Yield Ingestion
#
# Download and cache daily 2-year Treasury par yields from the
# U.S. Department of the Treasury for 2014–2024.

# %%
from io import StringIO
from pathlib import Path
import time
import pandas as pd
import requests

TREASURY_DIR = RAW_DIR / "treasury_yields"
TREASURY_DIR.mkdir(parents=True, exist_ok=True)


def download_treasury_year(year, max_retries=5):
    cache_path = TREASURY_DIR / f"treasury_{year}.csv"

    # Reuse previously downloaded files.
    if cache_path.exists():
        print(f"Using cached Treasury data: {year}")
        return pd.read_csv(cache_path)

    url = (
        "https://home.treasury.gov/resource-center/"
        "data-chart-center/interest-rates/"
        f"daily-treasury-rates.csv/{year}/all"
    )

    params = {
        "type": "daily_treasury_yield_curve",
        "field_tdr_date_value": str(year),
        "page": "",
        "_format": "csv"
    }

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"Downloading {year} "
                f"(attempt {attempt}/{max_retries})..."
            )

            response = requests.get(
                url,
                params=params,
                timeout=90,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()

            data = pd.read_csv(StringIO(response.text))
            data.columns = data.columns.str.strip()

            if "Date" not in data.columns or "2 Yr" not in data.columns:
                raise ValueError(
                    f"Unexpected columns for {year}: "
                    f"{data.columns.tolist()}"
                )

            if data.empty:
                raise ValueError(f"No data returned for {year}")

            # Cache only validated data.
            cache_path.write_text(
                response.text,
                encoding="utf-8"
            )

            print(f"Successfully downloaded {year}")
            return data

        except requests.exceptions.RequestException as error:
            print(f"Connection error: {error}")

            if attempt == max_retries:
                raise

            wait_time = min(5 * (2 ** (attempt - 1)), 60)
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


# Download all years, using cached files when available.
treasury_frames = []

for year in range(2014, 2025):
    yearly_df = download_treasury_year(year)
    treasury_frames.append(yearly_df)

    # Brief pause between requests.
    time.sleep(1)

treasury_raw_df = pd.concat(
    treasury_frames,
    ignore_index=True
)

print("\nAll Treasury data downloaded!")
print("Total rows:", len(treasury_raw_df))
print("Columns:", treasury_raw_df.columns.tolist())

display(treasury_raw_df.head())

# %%
# Standardize column names.
treasury_raw_df.columns = (
    treasury_raw_df.columns.str.strip()
)

# Keep only the date and 2-year Treasury yield.
treasury_df = treasury_raw_df[
    ["Date", "2 Yr"]
].copy()

treasury_df = treasury_df.rename(
    columns={
        "Date": "date",
        "2 Yr": "yield_2yr"
    }
)

# Convert dates and yields.
treasury_df["date"] = pd.to_datetime(
    treasury_df["date"],
    errors="raise"
)

treasury_df["yield_2yr"] = pd.to_numeric(
    treasury_df["yield_2yr"],
    errors="coerce"
)

# Remove rows with missing values.
treasury_df = treasury_df.dropna(
    subset=["date", "yield_2yr"]
)

# Sort by date.
treasury_df = treasury_df.sort_values(
    "date"
).reset_index(drop=True)

print("Treasury dataset shape:", treasury_df.shape)
print("Earliest date:", treasury_df["date"].min())
print("Latest date:", treasury_df["date"].max())

display(treasury_df.head())
display(treasury_df.tail())

# %%
print("Missing values:")
print(treasury_df.isnull().sum())

print("\nDuplicate dates:")
print(treasury_df["date"].duplicated().sum())

print("\nObservations by year:")
print(
    treasury_df["date"]
    .dt.year.value_counts()
    .sort_index()
)

print("\n2-year Treasury yield summary:")
print(treasury_df["yield_2yr"].describe())

# Validate data integrity.
assert treasury_df["date"].is_unique
assert treasury_df["yield_2yr"].notna().all()
assert treasury_df["date"].dt.year.between(
    2014, 2024
).all()

print("\nTreasury yield validation successful!")

# %%
treasury_output_path = (
    PROCESSED_DIR / "treasury_2yr_daily_2014_2024.csv"
)

treasury_df.to_csv(
    treasury_output_path,
    index=False,
    encoding="utf-8"
)

print("Treasury yield dataset saved successfully!")
print("File:", treasury_output_path)
print("Rows:", len(treasury_df))

# %% [markdown]
# ## FOMC Statement and Treasury Yield Alignment
#
# Align FOMC announcements with daily 2-year Treasury yields
# and calculate changes in basis points relative to the
# previous trading day's observation.

# %%
# Work with a copy of the Treasury data.
treasury_aligned = treasury_df.copy()

treasury_aligned = treasury_aligned.sort_values(
    "date"
).reset_index(drop=True)

# Previous available Treasury observation.
treasury_aligned["previous_trading_date"] = (
    treasury_aligned["date"].shift(1)
)

treasury_aligned["yield_previous_close"] = (
    treasury_aligned["yield_2yr"].shift(1)
)

# Yield on the current observation date.
treasury_aligned["yield_event_close"] = (
    treasury_aligned["yield_2yr"]
)

# Convert percentage-point changes to basis points.
treasury_aligned["yield_change_bps"] = (
    (
        treasury_aligned["yield_event_close"]
        - treasury_aligned["yield_previous_close"]
    ) * 100
).round(2)

display(treasury_aligned.head(10))

# %%
# Ensure both datasets use datetime dates.
fomc_final_df["date"] = pd.to_datetime(
    fomc_final_df["date"]
).dt.normalize()

treasury_aligned["date"] = pd.to_datetime(
    treasury_aligned["date"]
).dt.normalize()

# Keep the columns needed for alignment.
yield_columns = [
    "date",
    "previous_trading_date",
    "yield_previous_close",
    "yield_event_close",
    "yield_change_bps"
]

# Match FOMC announcement dates to Treasury dates.
fomc_yield_df = fomc_final_df.merge(
    treasury_aligned[yield_columns],
    on="date",
    how="left",
    validate="one_to_one"
)

print("Total FOMC announcements:", len(fomc_yield_df))

print(
    "Announcements with same-day yield data:",
    fomc_yield_df["yield_event_close"].notna().sum()
)

print(
    "Announcements missing same-day yield data:",
    fomc_yield_df["yield_event_close"].isna().sum()
)

display(fomc_yield_df.head())

# %%
missing_yields = fomc_yield_df[
    fomc_yield_df["yield_event_close"].isna()
].copy()

print("FOMC announcements without same-day yields:")
display(
    missing_yields[
        ["date", "source_url"]
    ]
)

print("\nYield change summary (basis points):")
print(fomc_yield_df["yield_change_bps"].describe())

print("\nMissing values:")
print(
    fomc_yield_df[
        [
            "yield_previous_close",
            "yield_event_close",
            "yield_change_bps"
        ]
    ].isna().sum()
)

# %%
# Preserve the original FOMC announcement date.
fomc_yield_df["yield_observation_date"] = (
    fomc_yield_df["date"]
)

# Identify announcements without same-day yields.
missing_mask = (
    fomc_yield_df["yield_event_close"].isna()
)

# Match missing announcements to the next
# available Treasury trading date.
missing_events = fomc_yield_df.loc[
    missing_mask, ["date"]
].sort_values("date")

next_trading_yields = pd.merge_asof(
    missing_events,
    treasury_aligned[
        [
            "date",
            "previous_trading_date",
            "yield_previous_close",
            "yield_event_close",
            "yield_change_bps"
        ]
    ].sort_values("date"),
    on="date",
    direction="forward",
    tolerance=pd.Timedelta(days=4)
)

# Get the actual observation date.
next_dates = pd.merge_asof(
    missing_events,
    treasury_aligned[
        ["date"]
    ].rename(
        columns={"date": "yield_observation_date"}
    ).sort_values("yield_observation_date"),
    left_on="date",
    right_on="yield_observation_date",
    direction="forward",
    tolerance=pd.Timedelta(days=4)
)

next_trading_yields["yield_observation_date"] = (
    next_dates["yield_observation_date"]
)

# Populate missing yield information.
for column in [
    "previous_trading_date",
    "yield_previous_close",
    "yield_event_close",
    "yield_change_bps",
    "yield_observation_date"
]:
    fomc_yield_df.loc[
        missing_mask, column
    ] = next_trading_yields[column].to_numpy()

# Flag whether the yield observation is on
# the actual announcement date.
fomc_yield_df["same_day_yield"] = (
    fomc_yield_df["date"]
    == fomc_yield_df["yield_observation_date"]
)

print("Same-day matches:",
      fomc_yield_df["same_day_yield"].sum())

print("Next-trading-day matches:",
      (~fomc_yield_df["same_day_yield"]).sum())

display(
    fomc_yield_df.loc[
        fomc_yield_df["date"]
        == pd.Timestamp("2020-03-15"),
        [
            "date",
            "previous_trading_date",
            "yield_observation_date",
            "yield_previous_close",
            "yield_event_close",
            "yield_change_bps",
            "same_day_yield"
        ]
    ]
)

# %%
# Verify all 84 announcements have yields.
assert len(fomc_yield_df) == 84

assert fomc_yield_df[
    "yield_change_bps"
].notna().all()

assert fomc_yield_df["date"].is_unique

# Verify the weekend event was aligned correctly.
weekend_event = fomc_yield_df.loc[
    fomc_yield_df["date"]
    == pd.Timestamp("2020-03-15")
].iloc[0]

assert weekend_event["previous_trading_date"] == (
    pd.Timestamp("2020-03-13")
)

assert weekend_event["yield_observation_date"] == (
    pd.Timestamp("2020-03-16")
)

print("Final alignment validation successful!")
print("Total announcements:", len(fomc_yield_df))

print("\nYield changes (basis points):")
print(fomc_yield_df["yield_change_bps"].describe())

# %%
aligned_output_path = (
    PROCESSED_DIR
    / "fomc_treasury_aligned_2015_2024.csv"
)

fomc_yield_df.to_csv(
    aligned_output_path,
    index=False,
    encoding="utf-8"
)

print("Aligned dataset saved successfully!")
print("File:", aligned_output_path)
print("Shape:", fomc_yield_df.shape)

# %% [markdown]
# ## Problem Definition and Baseline Sentence Classification
#
# The objective is to classify Federal Open Market Committee
# (FOMC) sentences as hawkish, dovish, or neutral.
#
# A TF-IDF vectorizer and logistic regression classifier
# serve as the baseline model.
#
# The model is trained on the labeled training dataset and
# evaluated on the held-out test dataset using macro-F1.
#
# Statement-level monetary policy tone is defined as the
# proportion of hawkish sentences minus the proportion of
# dovish sentences.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

# Load the previously cleaned and cached datasets.
train_data = pd.read_csv(PROCESSED_DIR / "train_clean.csv")
test_data = pd.read_csv(PROCESSED_DIR / "test_clean.csv")

# Separate sentences and their labels.
X_train = train_data["sentence"]
y_train = train_data["label"]

X_test = test_data["sentence"]
y_test = test_data["label"]

# Create a reproducible TF-IDF + logistic regression baseline.
baseline_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            max_features=10000,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    ),
])

# Fit only on training data.
baseline_model.fit(X_train, y_train)

# Predict labels for the held-out test set.
y_pred = baseline_model.predict(X_test)

# Evaluate the baseline.
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

accuracy = accuracy_score(y_test, y_pred)

print(f"Baseline macro-F1: {macro_f1:.4f}")
print(f"Baseline accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0,
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


# %%
def calculate_statement_tone(
    predicted_labels,
    hawkish_label,
    dovish_label,
):
    """Calculate hawkish share minus dovish share."""
    labels = list(predicted_labels)

    if not labels:
        raise ValueError("A statement must contain sentences.")

    hawkish_share = sum(
        label == hawkish_label for label in labels
    ) / len(labels)

    dovish_share = sum(
        label == dovish_label for label in labels
    ) / len(labels)

    return hawkish_share - dovish_share


# %%
# Official label mapping for the FOMC dataset.
LABEL_MAP = {
    0: "Dovish",
    1: "Hawkish",
    2: "Neutral",
}

HAWKISH_LABEL = 1
DOVISH_LABEL = 0
NEUTRAL_LABEL = 2

# Display baseline results with descriptive class names.
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Dovish", "Hawkish", "Neutral"],
        zero_division=0,
    )
)

# Verify the statement-level tone calculation.
example_labels = [1] * 8 + [0] * 5 + [2] * 7

example_tone = calculate_statement_tone(
    example_labels,
    hawkish_label=HAWKISH_LABEL,
    dovish_label=DOVISH_LABEL,
)

assert abs(example_tone - 0.15) < 1e-10
print(f"Example statement tone: {example_tone:+.2f}")

# %%
