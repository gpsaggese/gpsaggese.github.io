# Gensim LDA Tutorial

Topic modeling and sentiment analysis on text corpora using Gensim LDA. 

## Quick Start

- `cd tutorials/Gensim_LDA`
- `./docker_build.sh` — builds the Docker image, installs all dependencies from `requirements.txt`, downloads NLTK data and pre-caches the RoBERTa sentiment model
- `./docker_jupyter.sh` — launches a Jupyter Lab server inside the container on port 8888, mounts the project directory and opens at `http://localhost:8888`

Open the following notebooks in order:

1. `tutorial.ipynb` — Gensim LDA API overview covering preprocessing, dictionary and corpus construction, model training, coherence evaluation, topic inspection and pyLDAvis visualization using a sample Reuters dataset
2. `data_exploration.ipynb` — Full LDA topic modelling and RoBERTa sentiment analysis pipeline on the BBC News dataset including topic labelling, confidence distribution, inter-topic entropy, sentiment variance and polarity analysis
3. For Docker setup and configuration details see the
[project template README](../../class_project/project_template/docker_scripts.README.md). 