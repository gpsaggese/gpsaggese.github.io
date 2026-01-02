# LakeFS Fraud Detection API Wrapper

## 1. Overview
This module (`LakeFS_Fraud_utils.py`) serves as the core integration layer between the Data Science workflow and the LakeFS Version Control System. It implements a lightweight, object-oriented wrapper (`LakeFSDataHandler`) around the native `lakefs_client` API, abstracting away the complexity of authentication, object serialization, and branch management.

The module also includes specialized utility functions for machine learning operations, including preprocessing, model training, and visualization generation.

## 2. Class Documentation: `LakeFSDataHandler`

### Initialization
```python
handler = LakeFSDataHandler(host, access_key, secret_key, repo_name)
````

  * **host**: URL of the LakeFS server (e.g., `http://host.docker.internal:8000`).
  * **access\_key / secret\_key**: User credentials for authentication.
  * **repo\_name**: The target repository for data versioning.

### Core Methods

#### `upload_df(df, branch, path, message)`

Serializes a Pandas DataFrame into a CSV byte stream and commits it to LakeFS.

  * **Functionality**: Converts data to binary in-memory (no local temp files required), uploads to the specified branch/path, and performs an immediate commit.
  * **Error Handling**: Automatically detects if the file content has not changed and suppresses "empty commit" errors to prevent pipeline crashes.

#### `upload_file(file_path, branch, dest_path, message)`

Uploads arbitrary local files to LakeFS.

  * **Use Case**: Primarily used for archiving visualization artifacts (PNG images of Confusion Matrices, ROC Curves) alongside experiment metrics.

#### `load_df(branch, path)`

Retrieves a CSV file from LakeFS and returns it directly as a Pandas DataFrame.

  * **Optimization**: Utilizes `_preload_content=False` to handle the raw data stream directly, bypassing client-side parsing overhead for improved stability.

#### `create_branch(new_branch, source_branch="main")`

Creates a new isolated branch for experimentation.

  * **Behavior**: Checks if a branch exists; if so, it logs the event and proceeds without error, allowing for idempotent experiment runs.

-----

## 3. Data Science Utilities

### Preprocessing: `preprocess_data_pro(df)`

Implements a robust data cleaning pipeline tailored for financial fraud data:

  * **Stratified Split**: Preserves the ratio of fraud cases (0.17%) in both train and test sets.
  * **Imbalance Handling**: Applies **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set to prevent the model from biasing toward the majority class.
  * **Scaling**: Normalizes features using `StandardScaler`.

### Training: `train_and_eval(train_df, test_df, algo)`

A unified interface for training and evaluating various model architectures. It returns true labels, predictions, and probabilities.

**Supported Algorithms (`algo` argument):**

  * `lr`: Logistic Regression (Baseline)
  * `rf`: Random Forest Classifier
  * `xgb`: XGBoost (Gradient Boosting)
  * `lgbm`: LightGBM
  * `nn`: Deep Neural Network (Keras/TensorFlow)
  * `ensemble`: Soft Voting Ensemble (All models)
  * `power_ensemble`: Voting Ensemble (Tree-based models only)
  * `tuned_xgb`: XGBoost with Grid Search Hyperparameter Tuning

### Visualization

Helper functions to generate metric plots for reporting:

  * `save_confusion_matrix`: Generates heatmap of True Positives vs False Negatives.
  * `save_roc_curve`: Plots Receiver Operating Characteristic curve with AUC score.
  * `save_pr_curve`: Plots Precision-Recall curve (crucial for imbalanced datasets).