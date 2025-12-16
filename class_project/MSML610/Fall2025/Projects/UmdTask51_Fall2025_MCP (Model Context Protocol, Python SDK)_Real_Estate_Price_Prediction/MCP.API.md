# Role and Definition of the API Layer
The file MCP.API.ipynb serves as the central repository for defining the core Application Programming Interface (API) of the project.

It is responsible for the formal establishment of:

Strongly Typed Contracts: Implementing rigorous, type-hinted data structures for all system interaction points.

Data Flow Architecture: Governing the standardized flow of data, encompassing initial configurations, raw inputs, system execution metadata, and final, processed outputs.

Interoperability and Reproducibility: Centralizing these definitions to ensure clear component separation, consistent communication contracts, and reliable execution across various clients and dependent applications.

# Dataclasses

## Data Context (`DataContext`)

`DataContext` is a shared state container that manages datasets across the MCP platform. It provides a consistent interface for accessing and updating both the raw dataset and its feature-engineered version, enabling clean data flow between components.

### Attributes

| Field      | Type           | Description                                      |
| :--------- | :------------- | :----------------------------------------------- |
| `_data`    | `pd.DataFrame` | Primary (raw or latest) dataset.                 |
| `_fe_data` | `pd.DataFrame` | Feature-engineered dataset derived from `_data`. |

---

### Methods

| Method                                   | Description                                                           |
| :--------------------------------------- | :-------------------------------------------------------------------- |
| `set_data(new_data: pd.DataFrame)`       | Updates `_data` and initializes `_fe_data` as a copy of the new data. |
| `get_data() -> pd.DataFrame`             | Returns the primary dataset.                                          |
| `set_fe_data(new_fe_data: pd.DataFrame)` | Updates the feature-engineered dataset.                               |
| `get_fe_data() -> pd.DataFrame`          | Returns the feature-engineered dataset.                               |



## Experiment Configuration Dataclasses (Inputs)

These dataclasses define the configuration inputs required to execute individual components within the Multi-Component Platform (MCP). They provide a standardized, type-safe way to specify model choices and runtime parameters.

---

### `ModelConfig`

`ModelConfig` specifies the machine learning model to be trained along with its associated hyperparameters.

---

### Fields

| Field             | Type             | Default | Description                                                |
| :---------------- | :--------------- | :------ | :--------------------------------------------------------- |
| `model_name`      | `str`            | —       | Identifier of the model to use (e.g., class name).         |
| `hyperparameters` | `Dict[str, Any]` | `{}`    | Model-specific hyperparameters used during initialization. |


### `TrainingConfig`

`TrainingConfig` defines how the dataset is split and how the training process is executed. It encapsulates all parameters required to ensure reproducible and consistent model training.

---


### Fields

| Field           | Type    | Default | Description                                              |
| :-------------- | :------ | :------ | :------------------------------------------------------- |
| `target_column` | `str`   | —       | Name of the target (dependent) variable to be predicted. |
| `test_size`     | `float` | `0.2`   | Fraction of data reserved for the test split.            |
| `random_state`  | `int`   | `42`    | Random seed used for deterministic data splitting.       |

> **Note:** Evaluation metrics are currently fixed and implemented directly within the training component.


## 3. Experiment Record and Registry (Outputs/Persistence)

These dataclasses are used to store and persist the results of a component run, ensuring a complete and trackable record of every experiment.

### `Metrics`

This class defines the structure for recording all computed performance metrics and run-time statistics for a single model training and evaluation run.

### Class Definition


| Field | Type | Description |
| :--- | :--- | :--- |
| `train_r2` | `float` | R-squared value on the **training** set (measure of fit). |
| `train_rmse` | `float` | Root Mean Squared Error on the **training** set. |
| `test_r2` | `float` | R-squared value on the **test/validation** set. |
| `test_rmse` | `float` | Root Mean Squared Error on the **test/validation** set. |
| `test_rmse_pct_mean` | `float` | Test RMSE as a percentage of the target column's mean. |
| `test_rmse_pct_range` | `float` | Test RMSE as a percentage of the target column's range. |
| `train_time_sec` | `float` | The duration of the model fitting process (in seconds). |
| `best_param` | `float` | The single best hyperparameter value found (e.g., from a grid search). |

-----

### `ExperimentRecord`

`ExperimentRecord` is a comprehensive, immutable snapshot of a single component run. It collates inputs, configuration, metadata, and outputs to ensure reproducibility and traceability.

---

### Fields

| Field             | Type             | Description                                               |
| :---------------- | :--------------- | :-------------------------------------------------------- |
| `run_id`          | `str`            | Unique identifier for the experiment (e.g., UUID).        |
| `timestamp`       | `str`            | Timestamp when the run was recorded.                      |
| `model_config`    | `ModelConfig`    | Full model configuration (name and hyperparameters).      |
| `training_config` | `TrainingConfig` | Full training configuration (target column, split, etc.). |
| `metrics`         | `Metrics`        | Performance results and runtime statistics.               |
| `artifact_path`   | `str`            | File path of the trained model artifact.                  |
| `features_used`   | `List[str]`      | List of features used as predictors.                      |

---

## Experiment Registry (`ExperimentRegistry`)

`ExperimentRegistry` manages the persistence of all experiment metadata, storing `ExperimentRecord` objects in a central JSON file. It serves as a lightweight MLOps registry.

---


**Registry Files:**

* `REGISTRY_DIR`: `"artifacts"` – directory for artifacts and metadata.
* `REGISTRY_FILE`: `"artifacts/registry.json"` – central JSON metadata file.

---

### Core Methods

| Method                                    | Access  | Description                                                                                       |
| :---------------------------------------- | :------ | :------------------------------------------------------------------------------------------------ |
| `_load_registry()`                        | Private | Loads all experiment records from the JSON file; handles missing or corrupted files.              |
| `_save_registry(records)`                 | Private | Saves the current list of records back to the JSON file.                                          |
| `append_record(record: ExperimentRecord)` | Public  | Persists a new `ExperimentRecord` by converting nested dataclasses to JSON; returns the `run_id`. |
| `list_runs()`                             | Public  | Returns a raw list of all recorded runs.                                                          |
| `get_run_summary(run_id: str)`            | Public  | Returns a summary of a specific run by ID, or `None` if not found.                                |


## Production Registry (`ProductionRegistry`)

`ProductionRegistry` manages the final stage of the MLOps workflow: tracking which model (`run_id`) is currently designated as the **Production** model for serving or deployment.

---

**Registry File:** `artifacts/production_registry.json`

---

### Core Methods

| Method                              | Access  | Description                                                                                       |
| :---------------------------------- | :------ | :------------------------------------------------------------------------------------------------ |
| `_load_production_id()`             | Private | Reads the current `production_run_id` from the JSON file. Returns `None` if missing or corrupted. |
| `_save_production_id(run_id)`       | Private | Writes the provided `run_id` (or `None`) to the JSON file.                                        |
| `set_production_model(run_id: str)` | Public  | Marks a new model run as **Production** and persists it.                                          |
| `get_production_model()`            | Public  | Retrieves the currently active production `run_id`.                                               |

---





## 4. MCP Tools (Component API Functions)

These functions are decorated with `@mcp.tool()` and represent the callable components of the platform, designed to interact with the `DataContext` and the various registries.

### `upload_file`

This function is responsible for ingesting raw CSV data into the platform's execution context, populating the primary data store within `DataContext`.


| Parameter | Type | Description |
| :--- | :--- | :--- |
| `path` | `str` | The absolute file path to the CSV file to be loaded. |

| Returns | Type | Description |
| :--- | :--- | :--- |
| **Success** | `str` | A confirmation message including the shape (rows, columns) of the loaded data. |
| **Error** | `str` | An error message indicating issues like file not found, incorrect extension, or read failure. |

### Logic Summary

1.  **Validation:** Checks if the provided file path exists and ensures the file has a `.csv` extension.
2.  **Ingestion:** Uses `pd.read_csv()` to load the data into a Pandas DataFrame.
3.  **Context Storage:** Calls `data_context.set_data(data)` to store the DataFrame in `_data` and initialize `_fe_data` within the platform's `DataContext`.
4.  **Reporting:** Returns a success message with the data's dimensions.



### `get_columns_info`

This utility function provides a quick way for users to inspect the features (columns) currently available in the primary dataset stored in the `DataContext`.

| Returns | Type | Description |
| :--- | :--- | :--- |
| `str` | A comma-and-space separated string listing all column names from the primary dataframe (`_data`). |

### Logic Summary

1.  **Access Data:** Retrieves the current primary DataFrame using `data_context.get_data()`.
2.  **Extract Columns:** Extracts the column index.
3.  **Format Output:** Joins the column names into a single, readable string.



### `download_data`

This function allows a client to retrieve the entire primary dataset currently held in the `DataContext` as a structured JSON string, which is highly useful for client-side display or further external processing.

| Returns | Type | Description |
| :--- | :--- | :--- |
| **Success** | `str` | A JSON string representing the DataFrame, formatted as a list of records (rows). |
| **Error** | `str` | An error message if no data has been loaded into the context yet. |

### Logic Summary

1.  **Check Context:** Verifies that a DataFrame exists in the `DataContext`.
2.  **Serialization:** Uses the Pandas `to_json()` method with `orient='records'`. This ensures the output is an array of JSON objects, where each object represents a row, a robust format for web consumption.



### `engineer_features`

This component executes a predefined set of feature engineering transformations on the primary dataset. Its purpose is to derive more informative variables from existing columns, which are essential for improving model performance.

| Returns | Type | Description |
| :--- | :--- | :--- |
| **Success** | `str` | A confirmation message including the shape of the newly engineered DataFrame. |
| **Error** | `str` | An error message indicating missing input data (`date` column), or a general processing error. |

### `add_features`

This component performs a secondary, specific set of feature engineering tasks, focusing on creating ratio-based and combined-area features using the already prepared dataset from the `_fe_data` context.

| Returns | Type | Description |
| :--- | :--- | :--- |
| **Success** | `str` | A confirmation message indicating the successful completion of the feature creation. |
| **Error** | `str` | An error message indicating issues with the input data (e.g., `_fe_data` is empty) or a runtime error during calculation. |

### `run_experiment`

This is the central orchestration component. It takes configuration, trains a machine learning model, evaluates performance, serializes the trained model, and registers the full experiment metadata.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `mod_config` | `Dict[str, Any]` | Dictionary matching the `ModelConfig` structure (model name, hyperparameters). |
| `train_config` | `Dict[str, Any]` | Dictionary matching the `TrainingConfig` structure (target, split size, random state). |
| `feature_list` | `List[str]` | The exact list of column names to be used as predictors (`X`). |

| Returns | Type | Description |
| :--- | :--- | :--- |
| **Success** | `Dict[str, Any]` | A summary dictionary of the run, including all key metrics and the new `run_id`. |
| **Error** | `Dict[str, Any]` | A dictionary containing a single key `"error"` with a description of the failure. |

### Execution Workflow

The function follows a strict, traceable workflow:

1.  **Load Data & Configuration:** Retrieves the feature-engineered data from `DataContext` and converts input dictionaries into `ModelConfig` and `TrainingConfig` dataclasses.
2.  **Data Splitting:** Splits the feature data (`X`) and target (`y`, assumed to be `"price"`) into training and testing sets using parameters from `TrainingConfig`.
3.  **Model Initialization & Training:**
      * Initializes the model artifact using the helper function `_get_model_artifact`.
      * If `Ridge` or `Lasso` is selected and hyperparameters are provided, it performs a **5-fold cross-validated Grid Search** to find the optimal parameter (`best_param`).
      * The selected or tuned model is fitted to the training data.
4.  **Metric Computation:** Calculates various regression metrics (R-squared, RMSE) for both training and test sets, as well as percentage-based RMSE values for robust evaluation. The results are assembled into a `Metrics` dataclass.
5.  **Artifact Persistence:**
      * A unique `run_id` is generated.
      * The trained model object (`model_artifact`) is serialized using `pickle` and saved to the `artifacts/` directory.
6.  **Metadata Registration:** A complete `ExperimentRecord` is created using all inputs, outputs, and metadata, and is then persisted to the `ExperimentRegistry` JSON file.
7.  **Summary Return:** Retrieves and returns the final run summary for immediate client use.


### `predict_house_price`

This component is the serving API endpoint. It retrieves a registered model (defaulting to the one designated for Production) and uses it to generate a prediction based on a new set of raw input features provided by the client.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `run_id` | `str` | `None` | The specific run ID to use for prediction. If `None`, the model from `ProductionRegistry` is used. |
| `raw_features_dict` | `Dict[str, Any]` | `None` | A dictionary containing the raw input data for the house (e.g., `{'date': '2014-05-02', 'sqft_living': 1180, ...}`). |

| Returns | Type | Description |
| :--- | :--- | :--- |
| **Success** | `Dict[str, Any]` | A dictionary with `status`, `predicted_price` (float), and the `model_run_id` used. |
| **Error** | `Dict[str, Any]` | A dictionary with an error message. |

### Execution Workflow

1.  **Model Selection:** Determines the `run_id_to_use`. If not provided, it consults the `ProductionRegistry`.
2.  **Metadata Retrieval:** Fetches the `ExperimentRecord` to get the `artifact_path` and the `features_used` list.
3.  **Model Loading:** Deserializes the model (or Pipeline) artifact from disk using `pickle`.
4.  **Feature Preparation:**
      * Converts the `raw_features_dict` into a Pandas DataFrame (`X_raw`).
      * Calls the helper function `apply_feature_engineering(X_raw)` to recreate the features necessary for prediction **exactly as they were created during training**.
      * Filters the result (`X_engineered`) down to only the columns listed in `features_used`.
5.  **Prediction:** Calls `artifact.predict(X_final)`. The loaded model (which may be a Pipeline with a `StandardScaler`) automatically handles any necessary pre-processing steps before prediction.

-----

### `set_production_model`

This management tool automatically identifies and promotes the best performing model (based on the highest `test_r2`) from the `ExperimentRegistry` to the `ProductionRegistry`.

| Returns | Type | Description |
| :--- | :--- | :--- |
| **Success** | `Dict[str, str]` | A message confirming which `run_id` was promoted to Production status. |
| **Error** | `str` | An error message if no runs are found in the registry. |

### Logic Summary

1.  **Best Run Selection:** Loads all existing records from the `ExperimentRegistry` and uses the Python `max()` function to find the record with the **highest `test_r2` metric**. This run is implicitly selected as the "best" candidate for promotion.
2.  **Promotion:** Calls `production_registry.set_production_model()` with the selected best `run_id`.
3.  **Confirmation:** Returns a success message detailing the promoted `run_id`.

-----

### `get_production_model`

This tool retrieves the unique ID of the currently active production model and provides its full summary metrics for quick reference.

| Returns | Type | Description |
| :--- | :--- | :--- |
| `Dict[str, Any]` | A dictionary containing the `production_run_id` and the corresponding `summary` of its metrics and configurations. |

### Logic Summary

1.  **Retrieve ID:** Calls `production_registry.get_production_model()` to fetch the ID of the current production model.
2.  **Retrieve Summary:** Uses `experiment_registry.get_run_summary()` to look up the performance metrics and metadata associated with that ID.
3.  **Output:** Returns both the ID and the summary data (or `None` if the production ID has not been set).

-----
