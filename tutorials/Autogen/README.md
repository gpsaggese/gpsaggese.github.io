# AutoGen Tutorial

- This folder contains the setup for running AutoGen tutorials within a
  containerized environment

## Quick Start
- From the root of the repository, change your directory to the Autogen tutorial
  folder:
  ```bash
  > cd tutorials/Autogen
  ```

- Once the location has been changed to the repo run the command to build the
  image to run dockers:
  ```bash
  > ./docker_build.sh
  ```

- Once the docker has been built you can then go ahead and run the container and
  launch jupyter notebook using the created image using the command:
  ```bash
  > ./docker_jupyter.sh
  ```

- Once the `./docker_jupyter.sh` script is running, you can execute the tutorials

- For more informations on the Docker build system refer to [Project template
  readme](/class_project/project_template/docker_scripts.README.md)

## Tutorial Notebooks

Work through the following notebooks in order:

- [`autogen.API.ipynb`](autogen.API.ipynb): Tutorial notebook focusing on API
  configurations and basic agent setup
  - Master the fundamental commands and basic configurations of the AutoGen
    framework

- [`autogen.example1.ipynb`](autogen.example1.ipynb): Advanced end-to-end
  agentic workflow example Part 1
  - Fetches real-time stock data from Yahoo Finance
  - Bull and Bear strategist agents debate market trends
  - Selector agent dynamically decides which expert to call at each step
  - Generates stock charts and financial summaries

- [`autogen.example2.ipynb`](autogen.example2.ipynb): Advanced end-to-end
  agentic workflow example Part 2
  - Pulls 10-K filings from SEC EDGAR and cleans them
  - Embeds documents into a ChromaDB vector database
  - Senior Quant Analyst agent queries the database to extract revenue splits,
    risk factors, and other insights
  - Quant Runtime agent executes Python code locally to transform raw tables
    into structured visualizations

- [`autogen_utils.py`](autogen_utils.py): Utility functions required by the
  example notebooks

## Changelog

- 2026-03-01: Initial release
