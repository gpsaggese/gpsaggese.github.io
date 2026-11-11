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
  * [Semantic Kernel Plugin](#semantic-kernel-plugin)
  * [Natural-Language Agent](#natural-language-agent)
  * [LLM Providers](#llm-providers)
  * [Execution Trace](#execution-trace)
  * [Metric Check](#metric-check)
- [Project Files](#project-files)
- [Setup](#setup)
  * [Docker Image](#docker-image)
  * [LLM Configuration](#llm-configuration)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Verified Results](#verified-results)
  * [Model Comparison](#model-comparison)
  * [Linear Regression Coefficients](#linear-regression-coefficients)
  * [Random Forest Feature Importances](#random-forest-feature-importances)
  * [Live Agent Runs](#live-agent-runs)
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
- A Semantic Kernel agent answers natural-language requests: the language
  model chooses which plugin functions to call, Semantic Kernel runs them,
  and the model explains the results
- Every function call, including calls that Semantic Kernel rejects, is
  recorded in an execution trace
- The language model is a local Ollama server (default) or the GroqCloud
  Free plan, both used through Semantic Kernel's OpenAI connector
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
    functions, and the agent decides which of them to call for each request
- All numbers come from the Python functions, so every reported metric can be
  traced to deterministic code with a fixed random seed. The machine-learning
  code can also be tested without a language model

## Architecture

- The implementation is organized in layers:

  ```text
  housing_agent.py        ChatCompletionAgent with FunctionChoiceBehavior.Auto():
     |                    the model chooses functions; Semantic Kernel invokes them
     |-- housing_llm.py       chat service: Ollama (default) or Groq,
     |                        through the Semantic Kernel OpenAI connector
     |-- housing_trace.py     execution trace of invoked and rejected calls
     |-- housing_verify.py    check of labeled metrics in the answer
     v
  housing_kernel.py       Kernel with the HousingAnalysis plugin registered
     v
  housing_plugins/        HousingAnalysisPlugin: kernel functions returning
    housing_plugin.py     JSON strings; caches the data and model results
     v
  housing_models.py       Train/test split, Linear Regression, Random Forest,
     |                    MAE / RMSE / R²
     v
  housing_data.py         Download and validate the UCI dataset
  ```

- The plugin functions call the machine-learning modules directly and do not
  duplicate any modeling logic
- The machine-learning modules do not depend on Semantic Kernel or on any
  language model
- The fixed-order demonstration in `housing_kernel.py` from the first project
  phase is kept unchanged alongside the agent

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

### Semantic Kernel Plugin

- `housing_plugins/housing_plugin.py` defines `HousingAnalysisPlugin` with two
  functions decorated with `@kernel_function`:

  | Kernel function   | Description                                         | Output (JSON)                                  |
  | ----------------- | --------------------------------------------------- | ---------------------------------------------- |
  | `dataset_summary` | Number of rows, feature names, target, missing values | `rows`, `features`, `target`, `missing_values` |
  | `compare_models`  | Trains both models and returns their test scores    | `MAE`, `RMSE`, `R2` for each model             |

- Results are cached per plugin instance: the dataset is downloaded at most
  once and the models are trained at most once, however often the functions
  are called. Failed downloads are not cached
- The data source can be replaced (e.g., with synthetic data in tests) by
  passing a `data_loader` to the plugin
- `housing_kernel.create_kernel()` creates a `semantic_kernel.Kernel` and
  registers the plugin under the name `HousingAnalysis`

### Natural-Language Agent

- `housing_agent.py` defines a Semantic Kernel `ChatCompletionAgent` that
  receives the plugin functions as tools
- With `FunctionChoiceBehavior.Auto()`, the language model decides which
  functions each request needs, if any. Semantic Kernel invokes them, returns
  the results to the model, and repeats until the model answers
- The agent may use only the `HousingAnalysis` plugin. At most 5 model
  requests per run may call functions (the Semantic Kernel default, set
  explicitly). After that, the model must answer without tools
- The agent instructions require the model to call only the functions the
  question needs, to report numbers exactly as returned, to report function
  errors, and to say when a question cannot be answered with the available
  functions
- `housing_agent.run_agent(prompt)` answers one request and returns the
  answer, the execution trace, and the metric check
- Running `housing_agent.py` as a script answers one or more requests given
  on the command line, sharing one plugin (and therefore one data download)
  across all requests

### LLM Providers

- `housing_llm.py` builds the chat service from environment variables. All
  providers use the Semantic Kernel OpenAI connector, so no additional
  packages are required

  | Provider           | `HOUSING_LLM_PROVIDER` | Default model         | Endpoint                                | Credentials      |
  | ------------------ | ---------------------- | --------------------- | --------------------------------------- | ---------------- |
  | Ollama (local)     | `ollama` (default)     | `qwen3:4b`            | `http://host.docker.internal:11434/v1`  | none             |
  | GroqCloud          | `groq`                 | `openai/gpt-oss-120b` | `https://api.groq.com/openai/v1`        | `GROQ_API_KEY`   |
  | OpenAI (optional)  | `openai`               | must be set           | OpenAI API                              | `OPENAI_API_KEY` |

- Groq rejects requests whose messages contain a `name` field, which the
  Semantic Kernel agent adds to its instruction message. The Groq service
  (`GroqChatCompletion`) removes that field and leaves the rest of the request
  unchanged
- Failed requests, such as rate-limit (HTTP 429) errors, are retried
  (`HOUSING_LLM_MAX_RETRIES`, default 2)
- The OpenAI provider is supported by the configuration but has not been
  tested against the OpenAI API in this project

### Execution Trace

- `housing_trace.ExecutionTrace` records the function calls of one agent run:
  - **Executed calls** (`records`): order, model request, function name,
    arguments, status (`ok` or `error`), duration, full result, and error
    message if the function raised an exception
  - **Requested calls** (`attempts`): every call the model requested, marked
    `executed` or `rejected`. A call is rejected when Semantic Kernel refuses
    it before running the function, e.g., because the function name is
    unknown or the arguments do not match the signature. Rejected calls are
    read from the conversation history, keep the reason returned to the
    model, and are not counted as executions
- The trace can be printed (`format()`) or exported as JSON (`to_json()`,
  `attempts_to_dicts()`)
- Example for the Groq diagnostic run of the prompt "How many rows and
  features does the housing dataset have?" (shortened, with long lines
  wrapped):

  ```text
  1. HousingAnalysis.dataset_summary({}) [request 1] -> ok in 745 ms
     result: {"rows": 414, "features": ["X1 transaction date", ...
  Rejected function calls (not executed):
  - HousingAnalysis-dataset_summary({"":{}}) [request 0]
     reason: Received unexpected argument(s): ['']. Please revise the
       arguments to match the function signature.
  ```

### Metric Check

- `housing_verify.check_reported_metrics(answer, trace)` compares the labeled
  values in the answer with the results of the executed functions:
  - Number of rows, number of features, and missing-value count
  - MAE, RMSE, and R² of each model
- A reported value matches when it equals the function value after rounding
  to the precision used in the answer. R² may also be given as a percentage
- Each expected value is reported as `verified`, `mismatch`, or
  `not_reported`. Values reported without a supporting function result are
  flagged as issues
- The check covers only these labeled values and the phrasings it recognizes
  (sentences, lists, and markdown tables). It does not show that the
  explanations, comparisons, or conclusions in an answer are correct

## Project Files

- Project-specific files:

  | File                                | Purpose                                                     |
  | ----------------------------------- | ----------------------------------------------------------- |
  | `housing_data.py`                   | Dataset download and basic validation                       |
  | `housing_models.py`                 | Model definitions, training, and evaluation                 |
  | `housing_plugins/housing_plugin.py` | Semantic Kernel plugin exposing the analysis, with caching  |
  | `housing_kernel.py`                 | Kernel creation and fixed-order demonstration               |
  | `housing_llm.py`                    | LLM provider configuration and chat service                 |
  | `housing_agent.py`                  | Natural-language agent and command-line interface           |
  | `housing_trace.py`                  | Execution trace                                             |
  | `housing_verify.py`                 | Check of labeled metrics in agent answers                   |
  | `.env.example`                      | Template for the local LLM configuration                    |
  | `.gitignore`                        | Keeps `.env`, caches, and generated test files out of Git   |
  | `test/test_housing_models.py`       | Tests for model training and reproducibility                |
  | `test/test_housing_kernel.py`       | Tests for plugin registration and invocation                |
  | `test/test_housing_llm.py`          | Tests for the LLM configuration and Groq request format     |
  | `test/test_housing_agent.py`        | Tests for the agent, function calling, trace, and caching   |
  | `test/test_housing_verify.py`       | Tests for the metric check                                  |
  | `requirements.txt`                  | Python dependencies installed in the Docker image           |
  | `docker_name.sh`                    | Docker image name for this project                          |

- The remaining files (`Dockerfile`, `docker_*.sh`, `utils.sh`, `version.sh`,
  `run_jupyter.sh`, `*.API.*`, `*.example.*`, `*_utils.py`, and
  `test/test_docker_template.py`) come from the course project template
  - Their usage is described in `docker_scripts.README.md`
  - The `*.API.*` and `*.example.*` notebooks and scripts are still the
    template versions and do not yet contain project content

## Setup

### Docker Image

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

### LLM Configuration

- The agent reads its configuration from environment variables or from a
  `.env` file in the project directory
  - `.env` is ignored by Git. Never commit API keys
  - The Docker scripts mount the repository into the container, so the
    `.env` file is visible inside the container
- Create `.env` from the template and edit it:

  ```bash
  > cp .env.example .env
  ```

- Option 1: Groq (hosted, Free plan)
  - Create an API key at [console.groq.com](https://console.groq.com)
  - Set in `.env`:

    ```text
    HOUSING_LLM_PROVIDER=groq
    GROQ_API_KEY=<your-key>
    ```

- Option 2: Ollama (local, no account needed)
  - Install Ollama on the host, download the model, and start the server:

    ```bash
    > ollama pull qwen3:4b
    > ollama serve
    ```

  - Set `HOUSING_LLM_PROVIDER=ollama` in `.env`, or leave it unset
  - The container reaches the host server at `host.docker.internal`
    (Docker Desktop)

- Optional settings: `HOUSING_LLM_MODEL`, `HOUSING_LLM_BASE_URL`,
  `HOUSING_LLM_TIMEOUT` (seconds, default 180), and `HOUSING_LLM_MAX_RETRIES`
  (default 2). See `.env.example`

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

- Train and compare the models without a language model:

  ```bash
  > python housing_models.py
  ```

- Invoke the plugin functions in a fixed order without a language model:

  ```bash
  > python housing_kernel.py
  ```

- Ask the agent one or more questions (requires the LLM configuration):

  ```bash
  > python housing_agent.py "How many rows and features does the housing dataset have?" "Which model performs better on the test set?"
  ```

  - For each question, the script prints the answer, the execution trace,
    and the metric check

- Alternatively, run a single command from the host without an interactive
  shell:

  ```bash
  > ./docker_cmd.sh "cd /git_root/class_project/msml610/Fall2026/projects/UmdTask668_MSML610_Fall2026_SemanticKernel_Housing_Price_Analyzer && python housing_agent.py \"Compare the two models.\""
  ```

- Start JupyterLab (default port 8888):

  ```bash
  > ./docker_jupyter.sh
  ```

## Testing

- Run the project tests from the project directory inside the container:

  ```bash
  > python -m pytest test/test_housing_models.py test/test_housing_kernel.py test/test_housing_llm.py test/test_housing_agent.py test/test_housing_verify.py
  ```

- Expected result: `61 passed`
- No test calls a language model, and no test needs an API key or an Ollama
  server:
  - The agent tests replace the language model with a scripted chat service
    that returns fixed function calls. The agent, Semantic Kernel's automatic
    function invocation, the plugin, the models, and the trace run unchanged
  - The plugin in the agent tests uses synthetic data
  - The Groq request format is checked with a mock HTTP transport, so no
    request is sent
- Only `test_kernel_invocation` downloads the UCI dataset and requires
  internet access. The other 60 tests also pass without network access
- Test coverage:

  | Test file               | Tests | Verifies                                                                                     |
  | ----------------------- | ----: | -------------------------------------------------------------------------------------------- |
  | `test_housing_models.py` |    2 | Model training, metric validity, reproducibility with the fixed seed                          |
  | `test_housing_kernel.py` |    2 | Plugin registration and invocation through the kernel                                         |
  | `test_housing_llm.py`    |   17 | Provider configuration, required credentials, hidden API keys, Groq request without `name`    |
  | `test_housing_agent.py`  |   20 | Function selection, results returned to the model, multi-step order, errors, rejected calls, invocation limit, caching |
  | `test_housing_verify.py` |   20 | Metric check on real agent answers, rounding, mismatches, swapped models, unsupported values  |

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
- The Random Forest metrics can differ between runs in the last floating-point
  digits (beyond the fourth decimal), which does not affect the rounded values

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
- The coefficients and importances are printed by `housing_models.py`, but
  they are not yet available to the agent as plugin functions

### Live Agent Runs

- The agent was run against live language models with three prompts. Each
  prompt was run once with Groq and twice with Ollama

  | Prompt                                                                    | Expected functions                    | Groq `openai/gpt-oss-120b` | Ollama `qwen3:4b` |
  | ------------------------------------------------------------------------- | ------------------------------------- | -------------------------- | ----------------- |
  | How many rows and features does the housing dataset have?                | `dataset_summary`                     | correct                    | correct (2/2)     |
  | Which model performs better on the test set, Linear Regression or Random Forest? | `compare_models`               | correct                    | correct (2/2)     |
  | Give me an overview of the dataset and then compare the two models.      | `dataset_summary`, `compare_models`   | correct                    | correct (2/2)     |

- A fourth prompt outside the scope of the plugin ("What is the capital of
  France?") was answered by Ollama without calling any function (2/2)
- Groq measurements (one run per prompt, 8 model requests in total):
  - All MAE, RMSE, and R² values and dataset facts in the answers matched the
    function results
  - Tokens per prompt: 734–1,229 input and 107–247 output
  - End-to-end time per prompt: 0.9–2.3 s
  - No rate-limit or other HTTP errors occurred. The Free plan reported
    limits of 1,000 requests per day and 8,000 tokens per minute for this
    model
  - In the first prompt, the model's first call to `dataset_summary` had
    invalid arguments (`{"": {}}`). Semantic Kernel rejected it, and the
    model retried correctly in the next request. A second diagnostic run
    showed the same behavior, which the execution trace now records
- Ollama measurements (Apple M4, 16 GB): about 45 s per prompt, with 60–90%
  of the output tokens spent on the model's reasoning
- These runs confirm that both providers work. They are too few to measure
  how reliably the models choose functions

## Current Scope and Limitations

- The agent currently has two plugin functions. Exploratory data analysis,
  correlation analysis, cross-validation, and model explanations (e.g.,
  coefficients and feature importances) are not yet available as plugin
  functions
- Model evaluation uses a single train/test split. Cross-validation is not
  yet implemented
- The live agent results are based on a small number of runs, and language
  model output can vary between runs
- The Groq Free plan limits this model to 8,000 tokens per minute. The three
  test prompts used 3,538 tokens in total (3,066 input and 472 output tokens
  in 8 requests), so longer multi-step analyses may reach the limit and wait
  for retries
- The `openai/gpt-oss-120b` model on Groq does not issue parallel function
  calls, so each function call takes one model request, and it occasionally
  sends invalid arguments that cost an extra request
- The metric check covers only labeled dataset facts and model metrics. It
  does not validate explanations
- The dataset is downloaded once per plugin instance (i.e., once per script
  run), so running the project requires access to the UCI repository
- The Python dependencies in `requirements.txt` are not pinned to specific
  versions, so a rebuilt image may install newer versions than those listed
  in [Verified Results](#verified-results)
- The project notebooks do not yet contain project content

## Troubleshooting

- **`docker_build.sh` fails or the image is missing**
  - Verify that Docker is running with `docker info`
  - Rebuild without cache with `./docker_build.sh --no-cache`
- **Import errors such as `ModuleNotFoundError: No module named
  'housing_data'`**
  - Run the scripts and tests from the project directory inside the
    container, as shown in [Running the Project](#running-the-project)
- **`ValueError: GROQ_API_KEY is required for the 'groq' provider.`**
  - Add the key to `.env` in the project directory, as shown in
    [LLM Configuration](#llm-configuration)
- **`Agent run failed: ... Connection error` with Ollama**
  - Start the server on the host with `ollama serve` and check that
    `ollama list` shows the configured model
- **HTTP 429 (rate limit) errors with Groq**
  - The Free plan limits requests per day and tokens per minute. Wait a
    minute and retry, or increase `HOUSING_LLM_MAX_RETRIES`
- **`pytest: error: unrecognized arguments: --new-first`**
  - The repository `pytest.ini` relies on the built-in pytest cache plugin.
    Do not disable it with `-p no:cacheprovider`
- **Network or download errors from `ucimlrepo`**
  - The dataset is fetched from the UCI repository. Check the internet
    connection and retry

## Team

| Member            | Contributions to the current implementation                                                                                                                                        |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tejarsha Chappidi | Project template and Docker setup, dataset loader, Linear Regression and Random Forest pipeline, Semantic Kernel plugin, kernel, and agent, LLM configuration, execution trace, metric check, automated tests |
| Syed Muazuddin    | Data validation and preprocessing (under review, not yet included in this version)                                                                                                 |

## References

- Yeh, I. C., and Hsu, T. K. (2018). Building real estate valuation models
  with comparative approach through case-based reasoning. _Applied Soft
  Computing_, 65, 260–271
- [UCI Machine Learning Repository: Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)
- [Microsoft Semantic Kernel documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Semantic Kernel function calling](https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion/function-calling/function-choice-behaviors)
- [GroqCloud OpenAI compatibility](https://console.groq.com/docs/openai)
- [GroqCloud rate limits](https://console.groq.com/docs/rate-limits)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
