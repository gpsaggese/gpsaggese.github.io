# TensorFlow Tutorial

- Learn TensorFlow and TensorFlow Probability in 60 minutes through a
  hands-on introduction to tensors, Keras neural networks, and structural
  time series forecasting

## Quick Start

- Change directory to the tutorial folder:
  ```bash
  > cd tutorials/tensorflow
  ```

- Build the Docker image:
  ```bash
  > ./docker_build.sh
  ```

- Launch Jupyter Lab inside the container:
  ```bash
  > ./docker_jupyter.sh
  ```

- Open the notebooks in order:
  1. **`tensorflow.API.ipynb`**: Core TensorFlow concepts — tensors, automatic
     differentiation, Keras regression, and TFP distributions
  2. **`tensorflow.example.ipynb`**: End-to-end structural time series
     forecasting with trend, seasonality, holidays, and autoregression

- For more information on the Docker build system refer to the
  [Project template README](https://github.com/gpsaggese/umd_classes/blob/master/class_project/project_template/README.md)

## Changelog

- 2026-03-01: Initial release
