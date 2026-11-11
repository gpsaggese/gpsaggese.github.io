# Semantic Kernel Housing Price Analyzer

<!-- toc -->

- [Summary](#summary)
- [Project Information](#project-information)
- [Objective](#objective)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Implemented Functionality](#implemented-functionality)
  * [Data Loading](#data-loading)
  * [Model Training and Evaluation](#model-training-and-evaluation)
  * [Semantic Kernel Integration](#semantic-kernel-integration)
- [Project Files](#project-files)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Verified Results](#verified-results)
  * [Model Comparison](#model-comparison)
  * [Linear Regression Coefficients](#linear-regression-coefficients)
  * [Random Forest Feature Importances](#random-forest-feature-importances)
- [Current Scope and Limitations](#current-scope-and-limitations)
- [Troubleshooting](#troubleshooting)
- [Team](#team)
- [References](#references)

<!-- tocstop -->

## Summary

- This project analyzes housing prices from the UCI Real Estate Valuation
  dataset by combining a scikit-learn machine-learning pipeline with Microsoft
  Semantic Kernel
- The machine-learning layer loads the data, trains Linear Regression and
  Random Forest regressors, and evaluates them with MAE, RMSE, and R²
- The Semantic Kernel layer exposes the analysis as plugin functions
- All numbers come from the Python functions, not from the kernel
- The project runs inside the Docker environment provided by the course
  project template

## Project Information

- Course: MSML610, Fall 2026, University of Maryland, College Park
- Project: Semantic Kernel Housing Price Analyzer (Project 2, Medium)
- Issue: #668
- Group: 15

## Objective

- The goal of the project is to build a housing price analyzer in which a
  Semantic Kernel agent coordinates a reproducible machine-learning workflow
- The project separates two responsibilities:
  - **Computation**: data loading, model training, and evaluation are
    implemented as ordinary, independently testable Python functions
  - **Orchestration**: Semantic Kernel registers these functions as plugin
    functions and invokes them
- This separation keeps every reported metric traceable to deterministic code
  with a fixed random seed

## Architecture

- The current implementation has four modules arranged in layers:

  ```text
  housing_kernel.py              Creates the Kernel, registers the plugin,
          |                      and invokes the plugin functions
          v
  housing_plugins/               HousingAnalysisPlugin: kernel functions
    housing_plugin.py            that return JSON strings
          |
          v
  housing_models.py              Train/test split, Linear Regression,
          |                      Random Forest, MAE / RMSE / R²
          v
  housing_data.py                Download and validate the UCI dataset
  ```

- The plugin functions call the machine-learning modules directly and do not
  duplicate any modeling logic
- Results are returned to the kernel as JSON strings, so that they can be
  parsed and checked programmatically

## Dataset

- Source:
  [UCI Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)
  (UCI dataset id 477)
- The dataset contains 414 real-estate transactions from the Sindian District
  of New Taipei City, Taiwan, recorded in 2012–2013
- The dataset is downloaded at runtime with the `ucimlrepo` package. No copy
  of the data is stored in the repository
- Input features:

  | Column | Description                                          |
  | ------ | ---------------------------------------------------- |
  | X1     | Transaction date (as a fractional year)              |
  | X2     | House age (years)                                    |
  | X3     | Distance to the nearest MRT station (meters)         |
  | X4     | Number of convenience stores within walking distance |
  | X5     | Latitude (degrees)                                   |
  | X6     | Longitude (degrees)                                  |

- Target: `Y`, house price per unit area, in units of 10,000 New Taiwan
  Dollars per ping (1 ping ≈ 3.3 m²)
- Verified properties of the downloaded data:
  - 414 rows, 6 numeric features, 1 numeric target
  - No missing values in the features or the target

## Implemented Functionality

### Data Loading

- `housing_data.load_housing_data()` downloads the dataset and returns the
  features `X` as a `pandas.DataFrame` and the target `y` as a
  `pandas.Series`
- The function removes the transaction identifier column if present and
  raises a `ValueError` if the data is empty or if the feature and target row
  counts differ

### Model Training and Evaluation

- `housing_models.get_models()` returns the two baseline regressors:
  - `LinearRegression`
  - `RandomForestRegressor` with 100 trees and `random_state=42`
- `housing_models.train_models(X=None, y=None)`:
  - Loads the UCI dataset when no data is passed, or uses the provided `X`
    and `y`
  - Splits the data into 80% training and 20% test data with
    `random_state=42`
  - Trains both models on the same training split and evaluates them on the
    same test split
  - Returns the trained models and a comparison table with the columns
    `MAE`, `RMSE`, and `R2`
- `housing_models.evaluate_model()` computes the Mean Absolute Error, Root
  Mean Squared Error, and coefficient of determination (R²)
- Running `housing_models.py` as a script prints the comparison table, the
  Linear Regression coefficients, and the Random Forest feature importances

### Semantic Kernel Integration

- `housing_plugins/housing_plugin.py` defines `HousingAnalysisPlugin` with two
  functions decorated with `@kernel_function`:

  | Kernel function   | Description                                | Output (JSON)                                  |
  | ----------------- | ------------------------------------------ | ---------------------------------------------- |
  | `dataset_summary` | Describes the housing dataset              | `rows`, `features`, `target`, `missing_values` |
  | `compare_models`  | Trains and compares both regression models | `MAE`, `RMSE`, `R2` for each model             |

- `housing_kernel.create_kernel()` creates a `semantic_kernel.Kernel` and
  registers the plugin under the name `HousingAnalysis`
- Running `housing_kernel.py` as a script invokes `dataset_summary` and
  `compare_models` through `Kernel.invoke()` and prints the parsed JSON
  results

## Project Files

- Project-specific files:

  | File                                | Purpose                                              |
  | ----------------------------------- | ---------------------------------------------------- |
  | `housing_data.py`                   | Dataset download and basic validation                |
  | `housing_models.py`                 | Model definitions, training, and evaluation          |
  | `housing_plugins/housing_plugin.py` | Semantic Kernel plugin exposing the analysis         |
  | `housing_kernel.py`                 | Kernel creation, plugin registration, and invocation |
  | `test/test_housing_models.py`       | Tests for model training and reproducibility         |
  | `test/test_housing_kernel.py`       | Tests for plugin registration and invocation         |
  | `requirements.txt`                  | Python dependencies installed in the Docker image    |
  | `docker_name.sh`                    | Docker image name for this project                   |

- The remaining files (`Dockerfile`, `docker_*.sh`, `utils.sh`, `version.sh`,
  `run_jupyter.sh`, `*.API.*`, `*.example.*`, `*_utils.py`, and
  `test/test_docker_template.py`) come from the course project template
  - Their usage is described in `docker_scripts.README.md`
  - The `*.API.*` and `*.example.*` notebooks and scripts are still the
    template versions and do not yet contain project content

## Setup

- Prerequisites:
  - Docker installed and running
  - A clone of the class repository
  - Internet access, which is required to download the dataset from UCI

- Go to the project directory:

  ```bash
  > cd $GIT_ROOT/class_project/msml610/Fall2026/projects/UmdTask668_MSML610_Fall2026_SemanticKernel_Housing_Price_Analyzer
  ```

- Build the Docker image:

  ```bash
  > ./docker_build.sh
  ```

- Verify that the image exists:

  ```bash
  > docker image ls | grep umd_project_umdtask668
  ```

  - The image is named
    `gpsaggese/umd_project_umdtask668_msml610_fall2026_semantickernel_housing_price_analyzer`

## Running the Project

- Start a shell inside the container:

  ```bash
  > ./docker_bash.sh
  ```

- Inside the container, go to the project directory (the repository is
  mounted under `/git_root`):

  ```bash
  > cd /git_root/class_project/msml610/Fall2026/projects/UmdTask668_MSML610_Fall2026_SemanticKernel_Housing_Price_Analyzer
  ```

- Train and compare the models:

  ```bash
  > python housing_models.py
  ```

  - Expected: a comparison table followed by the Linear Regression
    coefficients and the Random Forest feature importances (see
    [Verified Results](#verified-results))

- Invoke the analysis through Semantic Kernel:

  ```bash
  > python housing_kernel.py
  ```

  - Expected: the JSON output of `dataset_summary` (414 rows, 6 features, 0
    missing values) followed by the JSON output of `compare_models`

- Alternatively, run a single command from the host without an interactive
  shell:

  ```bash
  > ./docker_cmd.sh "cd /git_root/class_project/msml610/Fall2026/projects/UmdTask668_MSML610_Fall2026_SemanticKernel_Housing_Price_Analyzer && python housing_kernel.py"
  ```

- Start JupyterLab (default port 8888):

  ```bash
  > ./docker_jupyter.sh
  ```

## Testing

- Run the project tests from the project directory inside the container:

  ```bash
  > python -m pytest test/test_housing_models.py test/test_housing_kernel.py -v
  ```

- Expected output:

  ```text
  test/test_housing_kernel.py::test_kernel_registration PASSED
  test/test_housing_kernel.py::test_kernel_invocation PASSED
  test/test_housing_models.py::test_train_models PASSED
  test/test_housing_models.py::test_reproducible_results PASSED
  4 passed
  ```

- Test coverage:

  | Test                        | Verifies                                                         |
  | --------------------------- | ---------------------------------------------------------------- |
  | `test_train_models`         | Both models train and return finite, non-negative error metrics |
  | `test_reproducible_results` | Repeated training with the fixed seed yields identical metrics  |
  | `test_kernel_registration`  | The plugin and both kernel functions are registered             |
  | `test_kernel_invocation`    | `dataset_summary` runs through the kernel and returns valid JSON |

- The model tests use small synthetic datasets and do not require network
  access
- `test_kernel_invocation` downloads the UCI dataset and therefore requires
  internet access
- `test/test_docker_template.py` belongs to the course template and tests the
  template notebooks. It is not part of the project test suite

## Verified Results

- The results below were obtained inside the project Docker image with
  Python 3.12.16, scikit-learn 1.9.1, pandas 3.1.0, NumPy 2.5.4,
  semantic-kernel 1.46.0, and ucimlrepo 0.0.7
- Both models use the same 80/20 train/test split with `random_state=42`
  (331 training rows, 83 test rows)
- MAE and RMSE are expressed in the target unit (10,000 New Taiwan Dollars
  per ping)

### Model Comparison

| Model             |    MAE |   RMSE |     R² |
| ----------------- | -----: | -----: | -----: |
| Linear Regression | 5.3054 | 7.3148 | 0.6811 |
| Random Forest     | 3.9652 | 5.6705 | 0.8083 |

- On this held-out split, the Random Forest has lower errors and explains a
  larger share of the variance in the test data than Linear Regression
- These figures come from a single train/test split of a small dataset and
  can change with a different split

### Linear Regression Coefficients

| Feature                                | Coefficient |
| -------------------------------------- | ----------: |
| X1 transaction date                    |      5.4407 |
| X2 house age                           |     -0.2708 |
| X3 distance to the nearest MRT station |     -0.0048 |
| X4 number of convenience stores        |      1.0914 |
| X5 latitude                            |    229.0431 |
| X6 longitude                           |    -29.4926 |

- The model is fitted on unscaled features, so each coefficient is expressed
  in the unit of its feature
- The magnitudes are therefore not directly comparable across features. For
  example, the large latitude coefficient reflects the small range of
  latitude values in the data

### Random Forest Feature Importances

| Feature                                | Importance |
| -------------------------------------- | ---------: |
| X3 distance to the nearest MRT station |     0.5663 |
| X2 house age                           |     0.1691 |
| X5 latitude                            |     0.1250 |
| X6 longitude                           |     0.0703 |
| X1 transaction date                    |     0.0427 |
| X4 number of convenience stores        |     0.0267 |

- These are scikit-learn impurity-based importances, which sum to 1
- Impurity-based importances can favor continuous features with many distinct
  values, and they describe the fitted model rather than causal effects on
  price

## Current Scope and Limitations

- `housing_kernel.py` invokes the plugin functions in a fixed order. No
  language model is connected yet, so the kernel cannot interpret
  natural-language requests or choose functions
- Model evaluation uses a single train/test split. Cross-validation is not
  yet implemented
- Exploratory data analysis, correlation analysis, and plots are not yet
  implemented
- The project notebooks do not yet contain project content
- The dataset is downloaded on every run, so the pipeline and
  `test_kernel_invocation` depend on the availability of the UCI repository

## Troubleshooting

- **`docker_build.sh` fails or the image is missing**
  - Verify that Docker is running with `docker info`
  - Rebuild without cache with `./docker_build.sh --no-cache`
- **Import errors such as `ModuleNotFoundError: No module named
  'housing_data'`**
  - Run the scripts and tests from the project directory inside the
    container, as shown in [Running the Project](#running-the-project)
- **`pytest: error: unrecognized arguments: --new-first`**
  - The repository `pytest.ini` relies on the built-in pytest cache plugin.
    Do not disable it with `-p no:cacheprovider`
- **Network or download errors from `ucimlrepo`**
  - The dataset is fetched from the UCI repository. Check the internet
    connection and retry

## Team

| Member            | Contributions to the current implementation                                                                                                          |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tejarsha Chappidi | Project template and Docker setup, dataset loader, Linear Regression and Random Forest pipeline, Semantic Kernel plugin and kernel, automated tests |
| Syed Muazuddin    | Data validation and preprocessing (under review, not yet included in this version)                                                                   |

## References

- Yeh, I. C., and Hsu, T. K. (2018). Building real estate valuation models
  with comparative approach through case-based reasoning. _Applied Soft
  Computing_, 65, 260–271
- [UCI Machine Learning Repository: Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)
- [Microsoft Semantic Kernel documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
