# Ax - Multi-Objective Optimization for Marketing Campaigns

## Summary

This project demonstrates the use of the Ax (Adaptive Experimentation) library to leverage the Bayesian Optimization technique to optimize the parameters of a marketing campaign.

The content of this project includes the description of the Ax library, the Bayesian Optimization algorithms, and the implementation of this method to optimize the hyperparameters of a marketing campaign in a real-time bidding (RTB) scenario.

## Getting Started

### Prerequisites

The project is designed to be run in a Docker container. As long as Docker is available, the project can be run locally without the need to install any dependencies.

### Installation and Docker Setup

1. Build the Docker image
   ```bash
   > ./docker_build.sh
   ```

2. Run the Docker container
   ```bash
   > ./docker_bash.sh
   ```

3. Open the Jupyter notebook
   ```bash
   > ./data/run_jupyter.sh
   ```

4. Open in browser
   - Go to [http://localhost:8888](http://localhost:8888) in your web browser

## Structure of the Project

### Auxiliary Files

- `docker_build.sh`
  - Builds the Docker image for the project
- `docker_bash.sh`
  - Runs the Docker container and opens the Jupyter notebook
- `docker_clean.sh`
  - Removes the Docker image from the local system
- `docker_exec.sh`
  - Executes a command in the Docker container
- `docker_push.sh`
  - Pushes the Docker image to the Docker Hub
- `install_jupyter_extensions.sh`
  - Installs the Jupyter extensions
- `run_jupyter.sh`
  - Runs the Jupyter notebook
- `version.sh`
  - Displays the versions of the installed software and packages

### Data Files

`models/` contains the models and datasets used to simulate a real-time bidding campaign.

### Description of the API

The API documentation is available in [Ax.API.md](Ax.API.md).

The Jupyter Notebook to demonstrate the use of this API is available in [Ax.API.ipynb](Ax.API.ipynb).

### Real-Time Bidding Algorithms with Bayesian Optimization

The use of the Bayesian Optimization technique is described in [Ax.example.md](Ax.example.md). The code for the simulation is available in [Ax.example.ipynb](Ax.example.ipynb).

### Multi-Armed Bandits

A second example of Bayesian Optimization is the use of the Ax library to find the optimal bandit strategy to maximize the reward in a multi-armed bandit problem. It's also described in [Ax.example.md](Ax.example.md).

### DSP Simulation

Bayesian Optimization is used to find the optimal hyperparameters for a bidding strategy. This strategy is usually based on the prediction of the Click-Through Rate (CTR) for a marketing campaign. In order to demonstrate the optimization of the hyperparameters, a simulation of a CTR prediction model is used.

The creation of this predictive model is described in [dsp_pctr_prediction_model.ipynb](dsp_pctr_prediction_model.ipynb).