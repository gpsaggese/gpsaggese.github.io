# Measuring the Impact of Lifestyle Programs on Diabetes Outcomes

## Overview :
### Project Objective :
To estimate the causal impact of lifestyle interventions such as dietary modifications and structured exercise programs on diabetes-related health outcomes (for example, HbA1c levels and disease progression), while rigorously accounting for confounding variables to ensure credible and unbiased effect estimates.

## Installation & Docker Setup
To ensure reproducibility, this project is containerized. Follow these steps to build and run the analysis.

### 1. Build the Image
Run this command in the project root (where the `Dockerfile` is located):
```bash
docker build -t causalml_project .
```

### 2. Run the Container
Start the Jupyter environment with volume mounting (to save your notebook changes):
```bash
# Mac/Linux/WSL
docker run -p 8888:8888 -v "$(pwd)":/app causalml_project
```

### 3. Access the Project
- Click the `http://127.0.0.1:8888...` link in your terminal to open JupyterLab.
- Open `CausalML.API.ipynb` to test the tool.
- Open `CausalML.example.ipynb` to see the full Diabetes analysis.

## Folder Structure :
- `data/` : Contains datasets used for analysis. 
- CausalML.API.ipynb : Jupyter notebook demonstrating the application of CausalML methods to measure the impact of lifestyle programs on diabetes outcomes.
- CausalML.API.md : Documentation for the CausalML API used in the notebook.
- CausalML.examples.ipynb : Additional examples of CausalML applications.
- CausalML.examples.md : Documentation for the examples provided.
- utils.py : Utility functions to support data processing and analysis.
- Dockerfile : Configuration file for building the Docker image.
- README.md : This file, providing an overview and setup instructions for the project.

## Resources :

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

- [CausalML Library Documentation](https://causalml.readthedocs.io/en/latest/)

- [MSML610 Advanced Machine Learning - Fall 2025](https://github.com/gpsaggese-org/umd_classes)