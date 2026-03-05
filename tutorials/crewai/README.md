# CrewAI Tutorial

- This folder contains a self-contained tutorial for building agentic AI
  workflows with **CrewAI** and local Ollama LLMs — no cloud API key needed.

## Quick Start

- Change directory to the tutorial folder:
  ```bash
  cd tutorials/crewai
  ```

- Build the Docker image:
  ```bash
  ./docker_build.sh
  ```

- Run the container and launch Jupyter:
  ```bash
  ./docker_jupyter.sh
  ```

- Follow this sequence to explore the tutorial:
  1. **`crewai.API.ipynb`** – Start here to learn the core CrewAI concepts:
     Agents, Tasks, Crew, Tools, and Process modes.
  2. **`crewai.example.ipynb`** – Build a complete Agentic EDA pipeline where
     a Data Analyst agent uses custom tools to explore a sales dataset.

- For Docker details refer to the [Project template
  README](https://github.com/gpsaggese/umd_classes/blob/master/class_project/project_template/README.md)
