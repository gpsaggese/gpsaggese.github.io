## Project Overview
This project aims to build a Sentiment Analysis model that adapts over time using Reinforcement Learning (RL). The primary objective is to optimize sentiment classification accuracy by incorporating user feedback as a reward signal.

The system is built upon a BERT baseline model trained on the Twitter US Airline Sentiment dataset. Using the trl (Transformers Reinforcement Learning) library, this project introduces an adaptive feedback mechanism that improves the model iteratively.

## Project Objectives

Develop a robust baseline sentiment classification model using pre-trained BERT.

Integrate a reinforcement learning feedback loop for continuous model improvement.

Evaluate how feedback-driven fine-tuning impacts model accuracy and generalization.

## Dataset

Source: Twitter US Airline Sentiment Dataset (Kaggle)

Description: Contains tweets labeled as positive, neutral, or negative, providing an ideal setup for text classification and feedback-based learning.

## Work Completed


### Data Loading

- Loads ./data/Tweets.csv into a pandas DataFrame.

- Selects relevant columns: text, airline_sentiment.

- Label Mapping

- Maps sentiments to integers: negative → 0, neutral → 1, positive → 2.

### Text Cleaning

- Removes @mentions and URLs.

- Trims whitespace and ensures labels are valid integers.

### Train/Test Split

- Stratified split (80/20) to preserve class ratios.

- Prints shape stats and label distribution for both splits.

### Persist Processed Data

- Saves to ./data/train_processed.csv and ./data/test_processed.csv for reuse.

### Baseline Model (BERT)

- Loads a tokenizer and AutoModelForSequenceClassification (BERT).

- Tokenizes datasets; builds Hugging Face Dataset objects.

- Configures TrainingArguments and Trainer (includes a compute_metrics with accuracy and weighted F1).

### Training + Evaluation

- Trains the baseline model.

- Evaluates on the test set; prints accuracy and weighted F1.

### Artifacts

- Saves the final model and tokenizer to ./final-baseline-model.

## Steps to Run the Project
1. 'git clone https://github.com/gpsaggese-org/umd_classes.git'   (cloning the repo)
2. 'git checkout UmdTask20_Fall2025_trl_Sentiment_Analysis_with_Reinforcement_Learning'  (switching to branch)
3. 'python3 -m venv venv'                       (setting up a virtual environment)

    'source venv/bin/activate'    (macOS/Linux)

    'venv\Scripts\activate'       (Windows)
4. 'pip install -r requirements.txt'   (installing the libraries in requirements.txt file)
5. Open the baseline-model.ipynb file and run all cells

## Next Steps
1. Integrate Reinforcement Learning (TRL)

Set up the trl environment to enable feedback-based model optimization.

Use PPOTrainer from TRL to fine-tune the BERT model through reward-driven training.

2. Define User Feedback Reward System

Simulate or collect user feedback on model predictions.

Develop a reward function that:

Rewards correct or confident predictions.

Penalizes incorrect or uncertain predictions.

Optionally incorporates uncertainty or entropy as part of the reward signal.

3. Fine-Tune the Sentiment Model

Apply RL-based optimization using TRL.

Adjust policy gradient parameters (learning rate, KL penalty) to balance stability and adaptability.

Continue training until the reward signal stabilizes.

4. Evaluate Model Performance

Compare model accuracy, F1 score, and reward trends before and after RL fine-tuning.

Visualize performance metrics to demonstrate how feedback improves learning over time.

5. Extend and Experiment

Implement active learning, allowing the model to request feedback on uncertain samples.

Explore multi-task learning by integrating additional sentiment-related subtasks (e.g., emotion classification).

Optionally deploy the model through an interactive dashboard or API to simulate real-time feedback.
