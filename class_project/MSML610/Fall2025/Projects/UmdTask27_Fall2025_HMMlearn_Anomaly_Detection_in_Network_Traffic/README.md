# Anomaly Detection in Network Traffic with HMMs (`hmmlearn`)

**Author:** Marie Vetluzhskikh (UID: 120143991)

**Difficulty:** Level 3 (Hard)

## Project Overview
This project investigates the feasibility of using **Hidden Markov Models (HMMs)** to detect security anomalies in network traffic. Using the `hmmlearn` library, we attempt to model "Normal" traffic behavior and identify deviations (attacks) based on Log-Likelihood scores.

### The Core Finding (PLS Read This First!)
A key discovery of this project is that standard network intrusion datasets (like **UNSW-NB15**) are often NOT suitable for HMMs because they lack strict time-series continuity.
* **Part 1:** Here I demonstrated this limitation by attempting to model UNSW-NB15
* **Part 2:** I change the dataset to the **CESNET-TimeSeries24**, which provides the sequential data necessary for Markov modeling, allowing for a proper anomaly detection experiment.

## File Structure

| File | Description |
| :--- | :--- |
| **`HMMlearn.API.ipynb`** | **The Theory & Tutorial.** Explains *what* HMMs are, the math behind them, and uses "Toy Examples" to demonstrate the tool. **Start here.** |
| **`HMMlearn.API.md`** | Documentation summary for the API notebook. |
| **`HMMlearn.example.ipynb`** | **The Actual Experiment.** The main project code. Covers the failure on Dataset 1 and the implementation on Dataset 2. |
| **`HMMlearn_utils.py`** | Helper functions for data segmentation, simulation, and custom plotting (imported by the Example notebook). |
| **`requirements.txt`** | Python dependencies required to run the project. |

## 3. Installation & Usage

### Prerequisites
* Python 3.8+
* Docker

### Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Dataset Setup:**
    * Ensure `UNSW_NB15_training-set.csv` and `42.csv` (CESNET) are in the root directory.

### Running the Project
Launch Jupyter Lab or Notebook:
```bash
jupyter notebook
```

1. Open **`HMMlearn.API.ipynb`** first to understand the theory.
2. Open **`HMMlearn.example.ipynb`** to see the network traffic analysis.

## Key Technologies* **`hmmlearn`**: For Gaussian and Categorical Hidden Markov Models.
* `pandas` & `numpy`: For time-series manipulation and sliding window segmentation.
* `matplotlib` & `seaborn`: For visualizing Log-Likelihood distributions and anomaly timelines.

## References & Data

1. **UNSW-NB15 Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
2. **CESNET-TimeSeries24:** [Zenodo Link](https://zenodo.org/records/13382427)
3. **hmmlearn Documentation:** [ReadTheDocs](https://hmmlearn.readthedocs.io/)


