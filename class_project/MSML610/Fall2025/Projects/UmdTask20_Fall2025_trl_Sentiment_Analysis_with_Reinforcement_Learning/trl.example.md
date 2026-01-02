# Airline Sentiment Analysis: Model Benchmark Application

## 1. Project Overview

This example application demonstrates how to use the `trl` library to benchmark different Large Language Models (LLMs) on a specific downstream task: **classifying customer sentiment in airline tweets**.

The goal is to track the performance evolution across four distinct modeling stages:

1.  **Baseline BERT**: A traditional encoder-only classifier (the "standard" approach).
2.  **Baseline GPT-2**: A generative model with basic supervised fine-tuning.
3.  **Improved SFT**: A GPT-2 model with optimized hyperparameters and better training data selection.
4.  **DPO (Active)**: The SFT model further aligned using **Direct Preference Optimization (DPO)** and Active Learning human feedback.

## 2. Models Under Evaluation

The application uses the `SentimentModel` wrapper to load the following checkpoints from the Hugging Face Hub:

| Model Label        | Architecture    | Role in Experiment                                                             |
| :----------------- | :-------------- | :----------------------------------------------------------------------------- |
| **Baseline BERT**  | BERT (Encoder)  | Establishes the discriminative baseline. High accuracy expected.               |
| **Baseline GPT-2** | GPT-2 (Decoder) | Tests if a generative model can perform classification "out of the box".       |
| **Improved SFT**   | GPT-2 (Decoder) | Tests if better fine-tuning improves generative consistency.                   |
| **DPO (Active)**   | GPT-2 (Decoder) | Tests if aligning the model to human preferences fixes specific failure cases. |

## 3. Implementation Workflow

The accompanying notebook (`trl.example.ipynb`) implements the following pipeline:

### Step 1: Data Ingestion

We use `trl_utils.load_airline_data()` to ingest the "Twitter US Airline Sentiment" dataset. This utility handles:

- Cleaning text (removing user handles and URLs).
- Mapping string labels (`negative`, `neutral`, `positive`) to integer IDs.
- Splitting data into training, validation, and a **held-out test set** for fair evaluation.

### Step 2: Model Initialization

We instantiate our custom `SentimentModel` wrapper for **all four** models. This abstraction allows us to treat discriminative (BERT) and generative (GPT-2) models identically during the prediction loop.

### Step 3: Comparative Inference

We run a batch of predictions on the held-out test set.

- For **BERT**, we extract the argmax of the logits.
- For **GPT-2**, we generate text completions and parse the resulting sentiment token.

### Step 4: Evaluation

We utilize `scikit-learn` to generate a classification report, comparing Precision, Recall, and F1-scores across all four approaches.

## 4. Key Findings

_(These findings are demonstrated in the notebook execution)_

- **BERT** often provides the highest raw accuracy but lacks generative capabilities.
- **GPT-2** requires careful prompt engineering and fine-tuning to output consistent labels.
- The **DPO Model** is expected to show improved calibration on difficult "neutral" tweets compared to the standard SFT models, demonstrating the value of the active learning loop.
