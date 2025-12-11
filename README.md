# Project 3: Real-Time Sentiment Analysis for Stock Market Prediction  
### Sentiment RL Trader – MSML610 (Fall 2025), University of Maryland  

---

## Project Objective  

Build a **real-time sentiment analysis pipeline** to predict stock price movements based on news sentiment and use those predictions to drive a trading strategy.  

The project connects to **NewsAPI** and **Yahoo Finance**, computes sentiment on news headlines, aligns that sentiment with stock price data, trains predictive models (LSTM + sentiment features), and finally uses a **reinforcement learning (RL) trader** to optimize decisions such as Buy / Sell / Hold.  
 
**W&B Project Report**  
https://api.wandb.ai/links/siva09-university-of-maryland/s51yro7n 

---

## Dataset Suggestions & Sources  

- **Dataset**:  
  - Financial news from **NewsAPI** (headline-level text with timestamps)  
  - Stock price data from **Yahoo Finance** (OHLCV time series)  

- **APIs / Sources**:  
  - **NewsAPI**: `https://newsapi.org/` – used to fetch financial headlines and article metadata  
  - **Yahoo Finance** via `yfinance` Python package – used to fetch historical and intraday stock prices  

---

## Tasks Implemented  

### 1. API Integration  
- Implemented automated connectors to:  
  - Fetch news headlines from **NewsAPI** for selected tickers or keywords  
  - Fetch stock prices from **Yahoo Finance** using `yfinance`  
- Ensures both data sources use consistent timestamps for later alignment.  

### 2. Sentiment Analysis  
- Uses a **pre-trained transformer-based sentiment model** (RoBERTa-style, inspired by GoEmotions).  
- For each headline, the model outputs **emotion / sentiment probabilities**.  
- These probabilities are converted into a **single sentiment score** that is used as a feature in downstream models and RL trading.  

### 3. Data Alignment  
- News headlines and stock price data are joined on timestamps using a custom alignment routine.  
- Handles different time frequencies and missing values.  
- Final aligned dataset is stored as:  

  ```text
  merged_news_stock_sentiment.csv

### 4. Model Development – LSTM Price Predictor

An LSTM-based regression model is trained to forecast short-horizon stock price movement using a combination of:

- Recent historical stock prices  
- Sentiment score time series derived from news

**Target Variable**
- Short-horizon return or price delta (next-step movement prediction)

**Evaluation**
- Training and validation loss curves  
- Prediction vs. actual price plots  
- Quantitative metrics logged to Weights & Biases (W&B)


---

## 5. Reinforcement Learning Trader

A reinforcement learning (RL) trading agent is used to learn optimal trading behavior based on market state signals.

**State Inputs**
- Recent price windows  
- Log returns / technical features  
- Aggregated sentiment signals  
- Optional: LSTM price predictions

**Actions**
- `Buy`
- `Sell`
- `Hold`

**Reward Function**
- Defined by changes in portfolio value over time

**Performance Tracking**
- Episode reward accumulation  
- Portfolio equity curve tracking  
- Risk-adjusted performance metrics:
  - **Sharpe Ratio ≈ 5.06**
  - Maximum drawdown  
  - Volatility of returns


---

## 6. Experiment Tracking with Weights & Biases (W&B)

All major experiments and model runs are logged using **Weights & Biases**, enabling transparent experiment comparison and reproducibility.

**Tracked Runs**
- LSTM training experiments  
- RL training episodes

**Logged Parameters**
- Learning rate  
- Batch size  
- Reward discount factor (γ)  
- Model architecture and environment settings

**Logged Metrics**
- Training and validation losses  
- Episode rewards  
- Total portfolio return  
- Sharpe Ratio  
- Maximum drawdown

**Visualization**
- Interactive dashboards for:
  - Metric comparison across runs  
  - Hyperparameter sweeps  
  - Training curve inspection


---

## 7. Interactive Reporting & Visualizations

Generated plots and reports are stored in the `plots/` directory:

- `equity_curve.png` — Overall trading equity curve across the evaluation period  
- `AAPL_equity.png` — Equity curve from a single-ticker example (Apple)

**Jupyter Notebooks**

Step-by-step notebooks walk through the full pipeline:

- Sentiment analysis and scoring  
- LSTM price prediction training and evaluation  
- Reinforcement learning trader simulation and analysis

---

# Project Overview: Sentiment-Aware Reinforcement Learning Trader

This document outlines the structure, setup, execution, and results of the `sentiment_rl_trader` project.

## Project Files

All files live under:

`class_project/MSML610/Fall2025/Projects/sentiment_rl_trader`

### Core Python Scripts
* **`config.py`**: Central configuration for paths, tickers, training parameters, and W&B settings.
* **`data_fetcher.py`**: Connects to NewsAPI and Yahoo Finance. Downloads and preprocesses raw news and price data. Produces the merged dataset `merged_news_stock_sentiment.csv`.
* **`sentiment_model.py`**: Loads the pre-trained transformer-based sentiment model. Applies it to news headlines and converts outputs into numerical sentiment features.
* **`lstm_model.py`**: Defines and trains the LSTM model for stock price movement prediction. Handles sequence generation, train/validation splits, and model saving.
* **`rl_trader.py`**: Implements the Reinforcement Learning trading agent. Defines state representation, action space, reward function, and learning loop.
* **`sentiment_rl_trader_utils.py`**: Utility functions shared across components: data transforms, feature engineering, evaluation metrics, etc.
* **`plotter.py`**: Generates plots such as equity curves, cumulative returns, and diagnostic charts.
* **`main.py`**: Orchestrates the end-to-end pipeline:
    * Fetch/prepare data
    * Run sentiment analysis
    * Train LSTM model
    * Train RL trader
    * Generate plots and log to W&B

### Data & Outputs
* **`merged_news_stock_sentiment.csv`**: Final aligned dataset with prices, returns, and sentiment features.
* **`plots/`**
    * `equity_curve.png`
    * `AAPL_equity.png`

---

##  Jupyter Notebooks

Located under:

`sentiment_rl_trader_ipynb/`

* **`sentiment_analysis.ipynb`**: Step-by-step sentiment computation and sanity checks.
* **`lstm_model.ipynb`**: Interactive training and evaluation of the LSTM model.
* **`rl_trading.ipynb`**: Interactive RL trading experiments and visualization of trading behavior.

### API / Example Notebooks
* **`sentiment_rl_trader.API.ipynb`**: Demonstrates programmatic usage of the modules as an API.
* **`sentiment_rl_trader.example.ipynb`**: High-level example that chains the whole pipeline together.
* **`sentiment_rl_trader.API.md`** and **`sentiment_rl_trader.example.md`**: Markdown documentation describing how to use the API and example notebooks.

---

## Docker & Shell Scripts

* **`Dockerfile`**: Reproducible Docker environment (Python 3.11-slim) with all dependencies.
* **`docker_build.sh`**: Builds the Docker image for the project.
* **`docker_bash.sh`**: Starts an interactive bash session inside the container.
* **`docker_jupyter.sh`**: Launches Jupyter Notebook inside the container for running `.ipynb` files.
* **`requirements.txt`**: Python dependency list (transformers, torch, yfinance, wandb, etc.).

---

## Setup and Dependencies

### 1. Clone the Repository

From your home directory or a chosen workspace:

```bash
git clone [https://github.com/sivaakash09/umd_classes.git](https://github.com/sivaakash09/umd_classes.git)
cd umd_classes/class_project/MSML610/Fall2025/Projects/sentiment_rl_trader
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate      
```
# For Windows (PowerShell):
# venv\Scripts\Activate.ps1


### 3. Install Python Dependencies

```bash
pip install --upgrade pip
```
```bash
pip install -r requirements.txt
```

### 4. Environment Setup (API Keys)

```bash
export NEWSAPI_KEY="<your_newsapi_key>"
```
```bash
export OPENAI_API_KEY="<your_openai_api_key_if_used>"
```

In Python, these are accessed via:
```python 
import os

news_key = os.environ["NEWSAPI_KEY"]
openai_key = os.environ.get("OPENAI_API_KEY", None)
```

### 5. Running the Full Pipeline

After activating the virtual environment and setting environment variables:

`python main.py`

This will:
* **`Fetch / load news and stock data`**
* **`Compute sentiment features`**  
* **`Train the LSTM model`**  
* **`Train the RL trader`**   
* **`Save plots under plots/`**   
* **`Log metrics and artifacts to W&B under the project`**    
* **`siva09-university-of-maryland/sentiment_rl_trader`**  


### Docker Workflow

Build Docker Image

```bash
./docker_build.sh
```

Start a Bash Session Inside the Container

```bash
./docker_bash.sh
```

Launch Jupyter Notebook in Docker

```bash
./docker_jupyter.sh
```

Then open the printed URL in a browser and navigate to:

`/class_project/MSML610/Fall2025/Projects/sentiment_rl_trader/sentiment_rl_trader_ipynb`

## System Architecture Flowchart

This project follows a fully modular pipeline, combining sentiment modeling, LSTM forecasting, and reinforcement learning.  
The flowchart below summarizes the complete workflow:

<p align="center">
  <img src="https://raw.githubusercontent.com/sivaakash09/umd_classes/clean_only_trader/rl_sentiment_trader.png" width="700" />
</p>

**Figure: End-to-end architecture from data ingestion to RL-based trading decisions.**


