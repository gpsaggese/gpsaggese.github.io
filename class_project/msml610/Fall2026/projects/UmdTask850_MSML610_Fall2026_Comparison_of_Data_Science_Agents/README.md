# Comparison of Data Science Agents

- Class: MSML610, Fall 2026 (group 16)
- Team: Aditi Karanjkar (@aditik168), Rithvik Kommareddy (@rkommare)
- GitHub issue: [#850](https://github.com/gpsaggese/gpsaggese.github.io/issues/850)
- Project description:
  [`research/ideas/in_progress.Comparison_of_Data_Science_Agents.md`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/in_progress.Comparison_of_Data_Science_Agents.md)

## Goal

- Data Science Agents are AI tools that automate parts of the data science
  workflow: loading and cleaning data, EDA, modeling, and evaluation
- This project compares a small set of agents on the same datasets and tasks
  and scores them on:
  - Model performance (e.g., accuracy, F1, RMSE)
  - Code readability and reproducibility
  - Explainability of the results
  - Runtime and tokens used
  - Robustness to faulty data: does the agent detect and fix the errors?

## Directory Layout

```text
UmdTask850_MSML610_Fall2026_Comparison_of_Data_Science_Agents/
├── README.md                     # This file
├── <project>.API.ipynb / .py     # How to use each agent's API
├── <project>.example.ipynb / .py # End-to-end comparison of the agents
├── <project>_utils.py            # Shared helper functions
├── Dockerfile, requirements.txt  # Container definition (from the template)
├── docker_*.sh, run_jupyter.sh   # Scripts to build and run the container
├── docker_scripts.README.md      # Reference docs for the Docker scripts
└── test/                         # Tests that run the notebooks in Docker
```

- `<project>` is `UmdTask850_MSML610_Fall2026_Comparison_of_Data_Science_Agents`
- The `.py` files are the `jupytext` pairs of the notebooks

## How to Run

- Prerequisites: Docker installed and running, repo cloned locally

- Go to the project directory
  ```bash
  > cd class_project/msml610/Fall2026/projects/UmdTask850_MSML610_Fall2026_Comparison_of_Data_Science_Agents
  ```

- Build the Docker image
  ```bash
  > ./docker_build.sh
  ```

- Start Jupyter and open `http://localhost:8888`
  ```bash
  > ./docker_jupyter.sh
  ```
  - If port 8888 is already in use, stop the other container first
    (`docker ps`, then `docker stop <name>`)

- Open the notebooks in
  `git_root/class_project/msml610/Fall2026/projects/UmdTask850_MSML610_Fall2026_Comparison_of_Data_Science_Agents`
  and run `Kernel > Restart Kernel and Run All Cells`
  - Start with `<project>.API.ipynb`, then `<project>.example.ipynb`

- (Optional) Open a shell in the container
  ```bash
  > ./docker_bash.sh
  ```
