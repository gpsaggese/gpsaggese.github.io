# UmdTask51: Real Estate Price Prediction with MCP

## Project Objective

The goal of this project is to predict house prices using the "House Sales in King County, USA" dataset. We will use an XGBoost regression model and leverage the **Model Context Protocol (MCP)** to manage the model training lifecycle, track experiments, and log parameters during hyperparameter tuning.

## Dataset

* **Name:** House Sales in King County, USA
* **Source:** [https://www.kaggle.com/datasets/harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
* **Local Path:** Assumed to be in this directory as `kc_house_data.csv`.

## Core Technologies

* **Model:** XGBoost (`xgboost`)
* **Experiment Tracking:** Model Context Protocol (`mcp`)
* **Data Handling:** `pandas`, `numpy`, `scikit-learn`
* **Environment:** Docker, Jupyter

## File Structure

As per the course guidelines, the project separates logic from presentation:

* **`MCP.API.md / .ipynb`**: A technical tutorial on how to use the `mcp` library itself, with simple examples.
* **`MCP.example.md / .ipynb`**: The main notebook for our real estate project, showing the end-to-end story.
* **`utils_data_io.py`**: Contains all Python functions for loading and saving data.
* **`utils_post_processing.py`**: Contains all Python functions for data cleaning and feature engineering.
* **`MCP_utils.py`**: Contains all Python functions for model training and hyperparameter tuning that use `mcp`.
* **`Dockerfile`**: Defines the Docker environment to run the project.

## How to Run

1.  **Build the Docker Image:**
    ```bash
    ./docker_build.sh
    ```
2.  **Start the JupyterLab Container:**
    ```bash
    ./docker_jupyter.sh
    ```
3.  **Access Jupyter:**
    Open the `http://127.0.0.1:8888/lab` link printed in your terminal.
4.  **Run the Notebooks:**
    * Start with `MCP.API.ipynb` to understand the tool.
    * Run `MCP.example.ipynb` to see the full project.