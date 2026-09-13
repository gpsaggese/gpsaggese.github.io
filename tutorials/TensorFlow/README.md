# TensorFlow Tutorial

This 60-minute hands-on tutorial introduces TensorFlow and TensorFlow Probability
through practical examples covering tensors, Keras neural networks, and
structural time series forecasting.

## Tutorial in 30 Seconds

**TensorFlow** is an open-source machine learning framework from Google for
building and training neural networks and probabilistic models.

Key capabilities:

- **Tensors and automatic differentiation**: Immutable multi-dimensional arrays
  optimized for CPUs, GPUs, and TPUs with efficient gradient computation
- **Keras API**: High-level interface for rapidly building and training neural
  networks
- **TensorFlow Probability**: Probabilistic programming for Bayesian inference
  and uncertainty quantification
- **Interpretable models**: Structural decomposition reveals which components
  drive predictions

After this tutorial, you will understand:

- The core APIs of TensorFlow (tensors, variables, automatic differentiation)
- How to build and train neural networks with Keras
- Probabilistic modeling with TensorFlow Probability distributions and
  Bayesian inference

## Official References

- [TensorFlow: An Open Source Machine Learning Framework](https://www.tensorflow.org/)
- [TensorFlow GitHub repo](https://github.com/tensorflow/tensorflow)
- [TensorFlow Probability](https://www.tensorflow.org/probability)
- [TensorFlow Probability GitHub repo](https://github.com/tensorflow/probability)

## Getting Started

### Prerequisites

This tutorial runs in a Docker container with all dependencies pre-configured. No
additional setup is required beyond the steps below.

### Setup Instructions

1. **Navigate to the tutorial directory:**
   ```bash
   > cd tutorials/TensorFlow
   ```

2. **Build the Docker image:**
   ```bash
   > ./docker_build.sh
   ```
   (See [`docker_build.sh`](docker_build.sh))

3. **Launch Jupyter Lab:**
   ```bash
   > ./docker_jupyter.sh
   ```
   (See [`docker_jupyter.sh`](docker_jupyter.sh))

## Dependency Management

This project uses `uv` for efficient Python dependency management within the
Docker container. The system works as follows:

- [`requirements.in`](requirements.in) — Lists top-level package dependencies
- [`requirements.txt`](requirements.txt) — Auto-generated pinned versions for reproducibility

The Docker container comes with all dependencies pre-compiled and synced. If you
need to update dependencies manually:
  ```bash
  # Compile top-level packages into pinned requirements
  > uv pip compile requirements.in -o requirements.txt

  # Sync the environment with the compiled list
  > uv pip sync requirements.txt
  ```

- For more informations on the Docker build system refer to [Project template
  readme](/class_project/project_template/docker_scripts.README.md)

## Tutorial Notebooks

Work through the following notebooks in order:

- [`tensorflow.API.ipynb`](tensorflow.API.ipynb): Core TensorFlow fundamentals
   - Tensors and tensor operations
   - Automatic differentiation
   - Keras regression models
   - TensorFlow Probability distributions

- [`tensorflow.example.ipynb`](tensorflow.example.ipynb): Advanced structural
  time series forecasting
   - Data Generation: Synthetic daily time series combining multiple
     components
   - Model Building: Approximate posterior over model parameters using
     Variational Inference (VI)
   - Forecasting and Evaluation: End-to-end pipeline for predictions and
     model assessment
   - Building trend and seasonality components
   - Incorporating holiday effects
   - Autoregressive modeling

- [`tensorflow_utils.py`](tensorflow_utils.py): Utility functions supporting the
  tutorial notebooks

## Changelog

- 2026-03-01: Initial release
