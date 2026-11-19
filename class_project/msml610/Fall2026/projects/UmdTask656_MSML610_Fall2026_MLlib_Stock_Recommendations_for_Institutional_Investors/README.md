# MLlib Tutorial

- Recommend new stock positions to institutional managers from SEC Form 13F
  holdings, using Spark MLlib ALS

## Quick Start

- From the repository root, in Git Bash:
  ```bash
  > cd class_project/msml610/Fall2026/projects/UmdTask656_MSML610_Fall2026_MLlib_Stock_Recommendations_for_Institutional_Investors
  > ./docker_build.sh
  > ./docker_jupyter.sh
  ```
- Docker Desktop needs about 8 GB of memory. The Spark driver asks for 4 GB
- Open the notebooks in this order:
  1. [`mllib.API.ipynb`](mllib.API.ipynb): `StringIndexer`, implicit `ALS`,
     tuning, `RankingEvaluator`, and `KMeans` on a toy manager-by-CUSIP
     matrix
  2. [`mllib.example.ipynb`](mllib.example.ipynb): real 13F holdings, ALS
     against a most-held baseline, nearest stocks, and item-factor clusters
- The example notebook downloads the four SEC quarterly zips into `data/`
  when those files are missing
- Docker script details are in the
  [project template README](/class_project/project_template/docker_scripts.README.md)

## Notebook Skills

- `notebook.create_api_intro`: shaped `mllib.API.ipynb` as a short native-API
  walkthrough on synthetic holdings, with each MLlib call in `mllib_utils.py`
- `notebook.rules.md`, used for both notebooks:
  - Notebooks stay thin and call `mllib_utils.py`
  - Each notebook is paired with a jupytext percent `.py` file
