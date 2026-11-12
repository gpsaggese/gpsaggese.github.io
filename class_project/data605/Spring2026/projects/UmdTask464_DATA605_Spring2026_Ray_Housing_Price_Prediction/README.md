# Ray Housing Price Prediction

A scalable end-to-end machine-learning pipeline that predicts California
median house values using **Ray** for distributed data loading, parallel
training, hyperparameter tuning, and model serving.

This project is a tutorial-style introduction to Ray's core APIs in the
context of a realistic regression problem: a curious computer scientist
should be able to read this README and the two notebooks and understand
what Ray offers in roughly 60 minutes.

---

## Table of Contents

1. [What is Ray?](#what-is-ray)
2. [Project Objective](#project-objective)
3. [File Layout](#file-layout)
4. [Quick Start](#quick-start)
5. [Running the Notebooks](#running-the-notebooks)
6. [Querying the Deployed Model](#querying-the-deployed-model)
7. [Results](#results)
8. [Architectural Decisions](#architectural-decisions)
9. [References](#references)

---

## What is Ray?

[Ray](https://www.ray.io/) is an open-source framework for distributed
Python. It provides a single API for scaling code across multiple cores
and machines, plus a stack of higher-level libraries built on top of
that core:

| Library       | What it does                                       |
| ------------- | -------------------------------------------------- |
| **Ray Core**  | Task and actor parallelism (`@ray.remote`)         |
| **Ray Data**  | Distributed data loading and preprocessing         |
| **Ray Tune**  | Distributed hyperparameter search                  |
| **Ray Serve** | Scalable model serving as a REST endpoint          |

This project uses all four to take a problem from raw data to a live API.

## Project Objective

Predict the **median house value** in California census tracts from
features such as median income, average rooms, location, and population.
The dataset is the standard scikit-learn `fetch_california_housing` dump
(20,640 rows x 8 features), so the project is reproducible without any
authentication or external download.

The pipeline:

1. Load and wrap the dataset with **Ray Data**
2. Train a baseline `RandomForestRegressor` with scikit-learn
3. Run multiple training jobs in parallel with **Ray Core** (`@ray.remote`)
4. Search the hyperparameter space with **Ray Tune**
5. Deploy the best model as a REST API with **Ray Serve**

## File Layout

```
.
├── Dockerfile                  # python:3.12-slim base + Ray + ML stack
├── .dockerignore
├── .gitignore
├── README.md                   # this file
├── requirements.txt            # pinned Python dependencies
├── ray_utils.py                # load_data() helper (uses Ray Data)
├── ray_housing.API.ipynb       # Tour of Ray's native APIs
├── ray_housing.example.ipynb   # End-to-end housing pipeline
│
├── docker_build.sh             # Build the project's Docker image
├── docker_bash.sh              # Open a bash shell inside the container
├── docker_jupyter.sh           # Start JupyterLab inside the container
├── docker_clean.sh             # Remove the project's Docker image
├── docker_cmd.sh               # Run an arbitrary command in the container
├── docker_exec.sh              # Attach to a running container
├── docker_push.sh              # Push the image to a registry
├── docker_name.sh              # Image-naming configuration
├── run_jupyter.sh              # JupyterLab launcher (called inside Docker)
├── version.sh                  # Logs Python/pip/Jupyter versions during build
├── bashrc                      # Container shell configuration
└── etc_sudoers                 # Container sudoers file
```

The `docker_*.sh` scripts and the support files (`bashrc`, `etc_sudoers`,
`version.sh`, `.dockerignore`, `Dockerfile`) come from the canonical
class template at `class_project/project_template/`. Only `Dockerfile`,
`docker_name.sh`, and `requirements.txt` were customized for this
project.

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  (or any working Docker daemon) running on your machine
- A Bash-compatible shell (Git Bash on Windows, any terminal on macOS / Linux)

### Build the image

From this project directory:

```bash
./docker_build.sh
```

The first build takes 5-10 minutes. It downloads `python:3.12-slim`,
installs Ray, scikit-learn, JupyterLab and all transitive dependencies,
and produces an image named:

```
gpsaggese/umd_data605_spring2026_ray_housing_price_prediction:latest
```

The final image is roughly 1.7 GB.

### Verify the build

```bash
docker images gpsaggese/umd_data605_spring2026_ray_housing_price_prediction
```

You should see one row with size around 1.7 GB.

## Running the Notebooks

### Start JupyterLab inside the container

```bash
./docker_jupyter.sh
```

This:

- Starts a container from the project image
- Mounts the current directory at `/data` inside the container
- Launches JupyterLab on port 8888 with no token / password (development mode)

Open <http://localhost:8888> in your browser. You'll see this project's
files in the file browser.

### Notebook tour

- **`ray_housing.API.ipynb`** - A short, didactic walk through Ray's
  core APIs in isolation: `@ray.remote`, `ray.data.from_pandas`,
  `tune.run`, `@serve.deployment`. Read this first if you've never used
  Ray.
- **`ray_housing.example.ipynb`** - The applied end-to-end pipeline:
  load housing data, train a baseline model, distribute multiple
  training runs, tune hyperparameters, and deploy the best model as a
  REST API.

Run the cells top-to-bottom. The Ray Tune section takes about 2-3 minutes;
everything else is fast.

## Querying the Deployed Model

Once `ray_housing.example.ipynb` has run the Ray Serve cell, the model
listens on port 8000 inside the container. Because the container maps
that port to the host, you can query it with `curl` or `requests`:

```bash
curl -X POST http://127.0.0.1:8000/ \
  -H "Content-Type: application/json" \
  -d '{"features": [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]}'
```

Expected response:

```json
{"prediction": 4.32}
```

The eight features, in order, are:
`MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`.
The prediction is the median house value in **hundreds of thousands of
dollars** (so `4.32` is about $432,000), matching the `MedHouseVal` units in
the original dataset.

## Results

Baseline `RandomForestRegressor` (`n_estimators=100`, default depth):

| Metric  | Value |
| ------- | ----- |
| RMSE    | ~0.51 |
| R^2     | ~0.81 |

After Ray Tune sweeps over `n_estimators` in {50, 100, 200} and
`max_depth` in {5, 10, 20}, the best configuration improves RMSE to
roughly 0.49 (final numbers depend on the random seed of each Tune
trial).

The tuned model is the one served by the Ray Serve endpoint.

## Architectural Decisions

A few non-obvious choices worth flagging for graders and future
maintainers.

**Base image: `python:3.12-slim`.** The class template offers Ubuntu,
Python-slim, and `uv`-based variants. Slim was chosen because this
project doesn't need system tools like `postgresql-client` or `graphviz`,
and the smaller base shaves both build time and final image size.

**Ray version: `2.49.0`.** Pinned for reproducibility. Ray < 2.31 has no
Python 3.12 wheels; Ray 2.49 is recent enough to be supported and stable
enough to have known-good behavior with the Tune and Serve APIs used
here.

**Ray Data inside `load_data()`.** The brief asks the project to use
Ray Data for loading and preprocessing. `ray_utils.load_data()` calls
`ray.data.from_pandas(...)` and returns the materialized DataFrame, so
the rest of the pipeline can stay in pandas-land. A future iteration
could keep the data as a Ray `Dataset` and use `.map_batches()` for
preprocessing.

**Single REST endpoint.** Ray Serve makes multi-endpoint deployments
trivial, but this project exposes one POST handler at `/` for clarity.
The class returns JSON with the predicted value and nothing else.

## References

- Ray documentation: <https://docs.ray.io/en/latest/>
- California Housing dataset:
  [`sklearn.datasets.fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- Class project template guide:
  `class_project/project_template/docker_scripts.README.md` in this repo
- Project guidelines:
  `class_project/data605/Spring2026/Class_Project_Guidelines.md`

---

*Project tag: `UmdTask464_DATA605_Spring2026_Ray_Housing_Price_Prediction`*
