# Weights & Biases (W&B) Core API Documentation

This document describes the core functionality of the Weights & Biases Python SDK for experiment tracking, independent of any specific machine learning project. The use case is a generic simulation of a training run.

## Core API Steps

The process of logging an experiment with W&B involves three fundamental steps centered around the concept of a **Run** .

### 1. Authentication (`wandb.login()`)

**What's Happening:** This step authenticates the local environment with the W&B cloud service, typically by reading the user's API key from an environment variable (`WANDB_API_KEY`). This is the gateway to logging data.

### 2. Initialization and Configuration (`wandb.init()`)

**What's Happening:** A new experiment **Run** is started. When calling `wandb.init()`, static metadata, such as hyperparameters (e.g., learning rate, batch size), is logged to the `config` object. These settings define the experiment's parameters.

### 3. Logging Metrics (`wandb.log()`)

**What's Happening:** This is the primary function used *inside* the simulated training loop. Dynamic metrics (like loss and accuracy) are streamed to the W&B dashboard in real-time. Each call to `wandb.log()` records a new step or epoch of the experiment.

### 4. Finalization (`run.finish()`)

**What's Happening:** The run is explicitly closed, ensuring all buffered data and final system metrics are synced to the W&B cloud.

## Python Usage (The entire process run as a single script)

The accompanying Python code `Weights_and_biases.api.ipynb` combines these steps to simulate a 10-epoch training proce