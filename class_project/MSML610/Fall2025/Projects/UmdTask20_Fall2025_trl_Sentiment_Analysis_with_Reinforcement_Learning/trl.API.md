# TRL Airline Sentiment API Documentation

## 1. Overview

The `trl` library provides a high-level, unified interface for performing sentiment analysis on airline customer feedback. It abstracts the complexity of the Hugging Face `transformers` library, allowing users to switch seamlessly between **Discriminative** (BERT) and **Generative** (GPT-2) architectures without changing their application code.

The library is specifically designed to support the **Active Learning** workflow, enabling the comparison of baseline models against those aligned via Direct Preference Optimization (DPO).

---

## 2. Native API Layer

The library acts as a lightweight wrapper around the standard PyTorch and Transformers ecosystem.

- **`transformers.AutoModelForSequenceClassification`**: utilized for the BERT baseline (Encoder-only).
- **`transformers.AutoModelForCausalLM`**: utilized for GPT-2 based models (Decoder-only).
- **`transformers.AutoTokenizer`**: handles tokenization for both architectures.

---

## 3. Wrapper API Layer (`trl_utils`)

To simplify integration, the library exposes a single wrapper class and several data utility functions.

### Class: `SentimentModel`

The main entry point for inference. It handles model loading, device placement (CPU/GPU/MPS), and prediction formatting.

#### `__init__(repo_id: str, model_type: str, device: torch.device = None)`

Initializes the model and tokenizer from the Hugging Face Hub.

- **Parameters:**
  - `repo_id` (str): The Hugging Face repository ID (e.g., `"blank4hd/airline-sentiment-bert-baseline"`).
  - `model_type` (str): The architecture type. Must be either `"bert"` or `"gpt2"`.
  - `device` (torch.device, optional): The computing device. If `None`, it is automatically detected.

#### `predict(text: str) -> Tuple[str, float]`

Performs inference on a single string of text.

- **Parameters:**
  - `text` (str): The input tweet to analyze.
- **Returns:**
  - `label` (str): The predicted sentiment (`"negative"`, `"neutral"`, or `"positive"`).
  - `confidence` (float):
    - For **BERT**: The softmax probability of the predicted class (0.0 - 1.0).
    - For **GPT-2**: Returns `0.0` (Generative models do not output a single scalar classification probability).

---

### Utility Functions

#### `get_device() -> torch.device`

Automatically selects the best available hardware accelerator.

- **Returns:** `torch.device` (MPS for macOS, CUDA for NVIDIA, or CPU).

#### `load_airline_data(csv_path: str = "Tweets.csv") -> Tuple[pd.DataFrame, Dict]`

Loads and preprocesses the Twitter US Airline Sentiment dataset.

- **Parameters:**
  - `csv_path` (str): Path to the raw CSV file.
- **Returns:**
  - `df` (DataFrame): A pandas DataFrame containing `clean_text` and `airline_sentiment`.
  - `label2id` (Dict): A mapping dictionary `{'negative': 0, 'neutral': 1, 'positive': 2}`.

#### `get_data_splits(df: pd.DataFrame, seed: int = 42) -> Tuple[DataFrame, DataFrame, DataFrame]`

Splits the dataset into training, validation, and a held-out test set using the specific logic required for this project (Hybrid Split).

- **Parameters:**
  - `df`: The preprocessed DataFrame.
- **Returns:**
  - `train_df`, `val_df`, `test_df`.

---

## 4. Supported Model Registry

The API is currently configured to support the following model endpoints:

| Model Name         | Architecture | Type           | Hugging Face ID                                       |
| :----------------- | :----------- | :------------- | :---------------------------------------------------- |
| **Baseline BERT**  | BERT-Base    | Discriminative | `blank4hd/airline-sentiment-bert-baseline`            |
| **Baseline GPT-2** | GPT-2        | Generative     | `blank4hd/airline-sentiment-baseline-gpt2-sft`        |
| **Improved SFT**   | GPT-2        | Generative     | `blank4hd/airline-sentiment-gpt2-improved-sft`        |
| **DPO Active**     | GPT-2        | Generative     | `blank4hd/airline-sentiment-gpt2-dpo-active-learning` |
