# Stumpy Tutorial: Matrix Profiles for Time-Series Anomaly Detection

A 60-minute tutorial on the [stumpy](https://stumpy.readthedocs.io/) library,
which computes matrix profiles for fast and scalable time-series motif and
discord (anomaly) discovery. The end-to-end example applies the multivariate,
causal LEFT matrix profile to SPY (S&P 500 ETF) daily data from 2018 to 2025
to detect market-stress regimes.

## Quick Start

1. Clone this repo and `cd` into the project directory:
   ```bash
   cd class_project/data605/Spring2026/projects/UmdTask428_DATA605_Spring2026_Stumpy
   ```
2. Build the Docker image (first build takes 5 to 10 minutes):
   ```bash
   ./docker_build.sh
   ```
3. Launch Jupyter Lab inside the container:
   ```bash
   ./docker_jupyter.sh
   ```
4. Open the URL printed in the terminal, then open the notebooks **in this order**:
   1. `stumpy.API.ipynb` — guided walkthrough of the stumpy API on synthetic
      data (matrix profile concept, `stump`, `mstump`, `stumpi`, `fluss`, motifs
      vs. discords). About 20 minutes.
   2. `stumpy.example.ipynb` — end-to-end application: stationary feature
      engineering on SPY, multivariate LEFT matrix profile, validation against
      labeled stress events, baselines (Isolation Forest, GARCH), ensemble, and
      a vol-targeted risk-off backtest. About 25 minutes.

For Docker details (architecture, scripts, helpers), see
[`class_project/project_template/docker_scripts.README.md`](../../../../project_template/docker_scripts.README.md).

## A note from the author

Thank you for the semester. This was a fun project to work on and a perfect
addition to my portfolio. Grateful for this experience.

