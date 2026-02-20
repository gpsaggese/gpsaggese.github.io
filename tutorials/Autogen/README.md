# AutoGen Tutorial

This folder contains the setup for running AutoGen tutorials within a containerized environment. 

## Quick Start

### 1. Navigate to the Directory
From the root of the repository, change your directory to the Autogen tutorial folder:
```bash 
cd tutorials/Autogen
```
Once the location has been changed to the repo run the command to build the image to run dockers: 
```bash 
./docker_build.sh
```

Once the docker has been built you can then go ahead and run the container and launch jupyter notebook using the created image using the command: 
```bash 
./docker_jupyter.sh
  ```

Once the `./docker_jupyter.sh` script is running, follow this sequence to explore the tutorials:

1. **`autogen.API.ipynb`**: Start here to master the fundamental commands and basic configurations of the AutoGen framework.
2. **`Autogen.example.ipynb`**: Proceed to this notebook to explore more complex, multi-agent scenarios and advanced problem-solving techniques.

---
Below is a quick reference for the shell scripts included in this directory:

- `docker_build.sh`: Builds the Docker image from the local Dockerfile.
- `docker_jupyter.sh`: Runs the container and starts a Jupyter Notebook server.
- `docker_bash.sh`: Opens an interactive Bash terminal inside the running container.
- `docker_exec.sh`: Executes a specific command inside an active container.
- `docker_clean.sh`: Removes unused containers and dangling images to free up space.
- `docker_cmd.sh`: Runs the container with a custom command passed as an argument.
- `docker_push.sh`: Pushes the local Docker image to a remote registry.
- `docker_name.sh`: Sets or retrieves the standard naming convention for the project's images.
- `run_jupyter.sh`: The internal entry point script used to initialize Jupyter inside the container.

Files Overview
- `autogen.example.ipynb`: Contains advanced example of how to use Autogen covering end-to-end agentic workflow. 
- `autogen_utils.py`: Contains the utilility functions required by `autogen.example.ipynb`
- `autogen.API.ipynb`: Tutorial notebook focusing on API configurations and basic agent setup.
- `Dockerfile` - Defines the environment configuration and OS-level dependencies.
- `requirements.txt` - Lists the Python packages required for AutoGen and data analysis.

Coming to the most important section of it

# autogen.example.ipynb

This notebook provides a practical, end-to-end example of using **AutoGen** to demonstrate a complete agentic workflow.

## Part 1: Dynamic Market Debate & Live Data
- Fetches real-time stock data from Yahoo Finance.  
- Bull and Bear strategist agents debate market trends.  
- Selector agent dynamically decides which expert to call at each step.  
- Generates stock charts and financial summaries.

## Part 2: SEC Filings & Quantitative RAG Analysis (Extension of Part 1)
- Pulls 10-K filings from SEC EDGAR and cleans them.  
- Embeds documents into a **ChromaDB** vector database.  
- Senior Quant Analyst agent queries the database to extract revenue splits, risk factors, and other insights.  
- Quant Runtime agent executes Python code locally to transform raw tables into structured visualizations.  

> Part 2 extends Part 1 by combining live market data with deep, structured analysis of SEC filings, showing how multiple agents collaborate, leverage private databases via RAG, and produce actionable insights in a single integrated workflow.

This example shows **how multiple agents collaborate**, use live data, leverage private databases via RAG, and produce actionable insights in a single integrated workflow. 

