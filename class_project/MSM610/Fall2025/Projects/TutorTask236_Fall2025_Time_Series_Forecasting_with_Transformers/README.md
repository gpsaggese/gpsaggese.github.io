# Transformer Time-Series Forecasting with Transformers

### Project Overview

This project implements a complete Transformer-based time-series forecasting system for multistep stock price prediction‹. The project demonstrates:

 * A native Transformer API implemented using JAX and Flax
 * A lightweight wrapper layer that simplifies data loading, training, evaluation, and inference
 * A fully reproducible Dockerized execution environment
 * Clean separation between core logic, API documentation, and example applications

Amazon (AMZN) historical stock price data is used as the primary example dataset.

### Repository Structure
.
├── transformer_utils.py          # Core API + wrapper utilities
├── transformer.API.md            # Native API documentation
├── transformer.API.ipynb         # Minimal API usage notebook
├── transformer.example.md        # End-to-end application description
├── transformer.example.ipynb     # Executable example notebook
├── README.md                     # This file
├── Dockerfile
├── requirements.txt
├── data/
│   └── AMZN.csv
├── checkpoints/
└── *.npy                         # Saved prediction outputs


### Project Files

This repository contains the following key files:

### Core API and Utilities
- transformer_utils.py
  Reusable utility module containing:

  - Data preparation
  - Transformer model definition
  - Training loop
  - Evaluation and inference helpers

### API Documentation
- transformer.API.md
Markdown documentation describing:

 - The native Transformer API
 - The wrapper layer design
 - Configuration usage and design philosophy

- transformer.API.ipynb
A minimal Jupyter notebook demonstrating usage of the native API and wrapper layer.


### Example Application

- transformer.example.ipynb
End-to-end Jupyter notebook demonstrating:

  - Dataset preparation
  - Model training
  - Evaluation
  - Visualization
  - Multi-step forecasting

- transformer.example.md
Markdown description of the example application and its workflow.

### Training and Execution Scripts

- train.py
Training logic using the Transformer model.

- eval.py
Model loading, inference, and evaluation utilities.

- data.py
Dataset loading and preprocessing helpers.

- model.py
Core model components (if separated).

- requirements.txt
Python dependencies.

### Data and Outputs

* data/AMZN.csv
  Historical stock price data.

* checkpoints/
  Saved model checkpoints.

* y_test_real.npy, y_pred_real.npy, next_5_prices.npy
  Saved evaluation and prediction outputs.

### Docker Support

* Dockerfile
  Defines a reproducible execution environment for the project.

### Setup and Dependencies

This project is designed to run inside a Docker container for reproducibility.

**Prerequisites**

- Docker
- Docker Desktop (Mac/Windows) or Docker Engine (Linux)

### Building and Running the Docker Container
1. Navigate to the Project Root

cd TutorTask236_Fall2025_Time_Series_Forecasting_with_Transformers

2. Build the Docker Image

docker build -t transformer-ts .

3. Run the Docker Container

docker run -it \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  transformer-ts

### Launching Jupyter Notebook

Inside the container, start Jupyter Lab:

jupyter lab --ip=0.0.0.0 --no-browser --allow-root


Then open the displayed URL in your browser (typically http://127.0.0.1:8888).

### Running the Notebooks

Recommended execution order:

1. transformer.API.ipynb
   Demonstrates minimal usage of the native API and wrapper layer.

2. transformer.example.ipynb
   Runs the full end-to-end forecasting pipeline, including:


   * Training
   * Evaluation
   * Visualization
   * Future price prediction

All notebooks invoke logic from transformer_utils.py and do not embed complex logic inline.

### Design Philosophy

* Separation of concerns:
  Core logic lives in reusable Python modules, not notebooks.

* Minimal notebooks:
  Notebooks act as declarative drivers.

* Reproducibility:
  Docker ensures consistent execution across environments.

* Clean API layering:
  Native model code is distinct from application-level wrappers.

## Summary

This project fulfills the assignment requirements by providing:

1. A documented native API
2. A lightweight wrapper layer
3. Clean API and example notebooks
4. A complete end-to-end application
5. A Dockerized, reproducible environment

The project demonstrates best practices in API design, machine learning engineering, and software organization for time-series forecasting with Transformers.