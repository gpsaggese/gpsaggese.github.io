# Anomaly Detection in Network Traffic Using Hidden Markov Models

**Author:** Marie Vetluzhskikh (UID: 120143991)
**Tool:** `hmmlearn` (Gaussian Hidden Markov Models)
**Course Project:** Network Traffic Anomaly Detection
**Related Files:**

* `HMMlearn.example.ipynb`  core experiments and results
* `HMMlearn.API.ipynb` theoretical background and API overview
* `HMMlearn_utils.py` data loading, segmentation, and visualization utilities


## Purpose of This Notebook

This notebook contains the core experimental results of the project.

The primary objective is **not** to maximize anomaly detection accuracy, but to critically evaluate whether Gaussian Hidden Markov Models (HMMs) are an appropriate modeling choice for network traffic anomaly detection.

In this project, I intentionally embraced a negative-result framework:

> If a model fails, the goal is to understand **why** it happens.

Theoretical background on Markov chains, HMM assumptions, and the `hmmlearn` API is provided separately in `HMMlearn.API.ipynb`. This notebook focuses exclusively on:

* modeling decisions,
* experiments,
* evaluation,
* and interpretation of results.


## HMM Assumptions

Gaussian HMMs rely on several key assumptions:
1. **First-order Markov property**
   Hidden states depend only on the previous state.

2. **Stationarity**
   Transition probabilities do not change over time.

3. **Gaussian emissions**
   Observations are generated from Gaussian distributions conditioned on the hidden state.

4. **Meaningful latent regimes**
   The system repeatedly visits a small number of stable hidden states.

Modern network traffic often violates *all four* assumptions. This notebook empirically investigates the consequences of these violations.

## Part I: UNSW-NB15

In this part, I demonstrate then Dataset–Model Mismatch. The UNSW-NB15 dataset consists of independent flow records rather than a true time series.
This dataset is misaligned with HMMs. This section is included **only** to demonstrate how forcing non-sequential data into an HMM framework can produce *misleadingly good results*.

#### Key experiment
* Independent rows are grouped into fixed-length sequences
* A Gaussian HMM is trained on artificially constructed “normal” sequences
* Likelihood separation between normal and attack traffic is observed

#### Critical finding
Although the model converges quickly and shows apparent likelihood separation, this separation is driven by static feature distribution differences, not by temporal modeling.
Transition dynamics have no semantic meaning.

> **Conclusion:**
> Apparent success on UNSW-NB15 is illusory and results from violating HMM assumptions.

## Part II: CESNET-TimeSeries24

To test whether the failure is caused by data mismatch rather than the model itself, the second half of the notebook uses the **CESNET-TimeSeries24** dataset.

This dataset is more appropriate since the observations are temporally ordered, it's designed explicitly for network traffic time-series analysis, and it has meaningful sequence modeling

### Modeling choices
* Data is treated as a single long sequence
* Hidden states represent latent traffic regimes, not attack labels
* Gaussian HMMs are trained with varying numbers of components
* Model selection is based on likelihood behavior (AIC/BIC intentionally avoided)

I acknowledge a key limitation:
> Even with sequential data, regime boundaries and semantics are still unknown.

This is a **structural limitation**, not an implementation error.

## Anomaly Detection Methodology

I detect anomalies are detected using **likelihood-based thresholding**.

I chose percentile-based thresholds for their interpretability. Sensitivity to threshold choice is explicitly explored. Thresholding is treated as an analytical tool, not an optimization step

## Synthetic Anomaly Injection

Unfortunately, CESNET dataset doesn't provide fine-grained anomaly labels. To get quantitative evaluation:
* I inject synthetic anomalies by amplifying feature values over fixed windows
* These anomalies simulate abrupt distributional shifts
* They are **not** intended to represent realistic attacks

> The purpose is to test **model sensitivity**, not realism.

## Evaluation Results

### Key quantitative outcome
* **Recall (Anomaly):** 1.00
* **Precision (Anomaly):** ≈ 0.10
* **Macro F1:** ≈ 0.54

### Interpretation

* The model detects all injected anomalies
* But produces a large number of false positives
* Overall accuracy is misleading due to severe class imbalance

This behavior is characteristic of density-based anomaly detectors applied to complex, non-stationary data.

> The model becomes overly sensitive and flags benign variations as anomalies.

## Core Experimental Conclusion

In this project, I tried to demonstrate that:

* Gaussian HMMs can capture **coarse temporal regularities**
* But they fail to reliably distinguish malicious behavior from benign variation
* Likelihood-based anomaly detection is poorly aligned with adversarial network activity
* Violations of Gaussianity, Markovianity, and stationarity dominate results

> **Final conclusion:**
> The observed poor performance is not due to implementation error, but to a **fundamental mismatch between HMM assumptions and real-world network traffic dynamics**.

I found this negative result by itself informative. It highlights the importance of aligning model assumptions with data characteristics.

## Reproducibility

All experiments can be reproduced using:

* `HMMlearn.example.ipynb`
* `HMMlearn_utils.py`
* `requirements.txt` / Docker configuration (if applicable)

Random seeds are fixed where applicable.

### Final note

This project intentionally prioritizes **model understanding over leaderboard performance**.
The results emphasize that **choosing the wrong model can be more instructive than tuning the wrong one well**.

