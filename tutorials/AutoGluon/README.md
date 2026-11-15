# AutoGluon Tutorial
- Learn AutoGluon Tabular in a containerized environment

## Quick Start
- From the root of the repository, change your directory to the AutoGluon
  tutorial folder:
  ```bash
  > cd tutorials/AutoGluon
  ```

- Build the image (it installs the pinned packages in `requirements.txt`):
  ```bash
  > ./docker_build.sh
  ```

- Run the container and launch Jupyter:
  ```bash
  > ./docker_jupyter.sh
  ```

- For more information on the Docker build system refer to [Project template
  README](/class_project/project_template/docker_scripts.README.md)

## Tutorial Notebooks
- Work through the following notebooks in order:
  - [`AutoGluon.API.ipynb`](AutoGluon.API.ipynb): the `TabularPredictor` API on
    synthetic churn data: feature inference, leaderboard, feature importance,
    bagging and stacking, presets, regression, multiclass, deployment
  - [`AutoGluon.example.ipynb`](AutoGluon.example.ipynb): employee attrition on
    the IBM HR dataset: baseline, ensemble, decision thresholds and costs, SHAP
  - [`AutoGluon_utils.py`](AutoGluon_utils.py): data, metrics, and plots used by
    the notebooks

## Changelog
- 2026-11-09: Initial release
