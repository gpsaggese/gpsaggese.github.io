# Theoretical Guide & API Documentation

## Overview
This document summarizes the **`HMMlearn.API.ipynb`** notebook, which I designed to provide some theory on Markov Chains and Hidden Markov Models (HMMs) for a reader unfamiliar with the topic.
It also documents the usage and core functions of the tool I used in my porject, the Python `hmmlearn` library.

My key focus was not just show *how* HMMs work, but *when* they are theoretically unsuited for a given task, specifically for Anomaly Detection in complex, real-world data like network traffic.

## Key Topics Covered in `HMMlearn.API.ipynb`

-  Markov Chains: introduces the memory-1 and the stationarity assumptions. These are the two constraints network traffic data may violate
- HMMs: defines the three core problems solved by HMMs
- The `hmmlearn` API: basic examples using `hmmlearn`, including the `GaussianHMM` (for continuous data) used in my project

## Critical Insight: Why HMMs Struggle with Network Traffic

Notebook provides the core justification for the experimental results we will see in the main project (`HMMlearn.example.ipynb`), such as memory-1 violation, overlapping states, and non-stationarity.

## Usage

To run the theoretical examples and verify the library installation:
1.  Ensure `hmmlearn`, `numpy`, and `matplotlib` are installed (see `requirements.txt`)
2.  Execute all cells sequentially in the **`HMMlearn.API.ipynb`** notebook