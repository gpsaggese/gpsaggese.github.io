# Weights & Biases (W&B) Core API Documentation

This document describes the essential functionality of the W&B Python SDK (`wandb`) for experiment tracking. The use case is a generic simulation of a training run, independent of any specific machine learning project.

## Core API Steps

The process of logging an experiment with W&B is defined by the lifecycle of a **Run**. 

### 1. Authentication (`wandb.login()`)

This step connects your local script to your W&B account, verifying your identity.

### 2. Initialization and Configuration (`wandb.init()`)

A new experiment **Run** is started. This function is used to define and log the *static* settings of your experiment. All hyperparameters (e.g., learning rate, model type) are saved to the `config` object.

### 3. Logging Metrics (`wandb.log()`)

Used repeatedly inside the loop, this function streams dynamic data in real-time. Each call records a new data point or *step* to the dashboard, visualizing metrics like loss and accuracy as they progress.

### 4. Finalization (`run.finish()`)

This is the required closing call. It formally terminates the run and ensures all buffered data, code, and final system details are successfully synced to the W&B dashboard.

## Python Usage (The entire process run as a single script)

The accompanying Python code, which you can save as `Weights_and_biases.API.py`, executes this entire workflow to simulate a 10-epoch training process, logging configuration and streaming metrics.