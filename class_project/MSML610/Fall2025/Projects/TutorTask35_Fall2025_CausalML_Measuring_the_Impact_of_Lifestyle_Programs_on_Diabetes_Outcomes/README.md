# Measuring_the_Impact_of_Lifestyle_Programs_on_Diabetes_Outcomes

## Overview :
### Project Objective :
To estimate the causal impact of lifestyle interventions such as dietary modifications and structured exercise programs on diabetes-related health outcomes (for example, HbA1c levels and disease progression), while rigorously accounting for confounding variables to ensure credible and unbiased effect estimates.

## Setup Instructions :
1. Build the Docker image using the provided Dockerfile:
   ```bash
   docker build -t causalml-tutorial .
   ```
2. Run the Docker container:
    ```bash
    docker run --rm -it \ -p 8888:8888 \ -v "$PWD":/app \ -w /app \ causalml-tutorial \ jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    ```
## Folder Structure :
- `data/` : Contains datasets used for analysis. 
- CausalML.API.ipynb : Jupyter notebook demonstrating the application of CausalML methods to measure the impact of lifestyle programs on diabetes outcomes.
- CausalML.API.md : Documentation for the CausalML API used in the notebook.
- CausalML.Examples.ipynb : Additional examples of CausalML applications.
- CausalML.Examples.md : Documentation for the examples provided.
- utils.py : Utility functions to support data processing and analysis.
- Dockerfile : Configuration file for building the Docker image.
- README.md : This file, providing an overview and setup instructions for the project.

## Resources :

- [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
