# Employee Attrition Prediction (IBM HR Analytics)

This project is a beginner-friendly tutorial that walks through building and evaluating machine-learning models to predict employee attrition using the **IBM HR Analytics Employee Attrition** dataset.

The goal is to show how to:
- Load and preprocess the dataset (with automatic download from Kaggle)
- Build train/test splits with proper preprocessing
- Train multiple classifiers (e.g., XGBoost, Random Forest, Logistic Regression)
- Evaluate models with Accuracy, F1, ROC-AUC and classification reports
- Inspect feature importance and model explanations

---

## Dataset

- **Source**: [IBM HR Analytics Employee Attrition dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
- The dataset is **automatically downloaded** via [`kagglehub`](https://github.com/Kaggle/kagglehub) inside `utils_data_io.py`.
- Main file used: `WA_Fn-UseC_-HR-Employee-Attrition.csv`

No manual CSV download is required as long as `kagglehub` is installed and configured.

---

## Project Structure

```text
.
├── employee_attrition_eda.ipynb   # Main tutorial / analysis notebook
├── employee_attrition_main.py     # Main Python entry point
├── utils_data_io.py               # Data loading, cleaning & preprocessing utilities
├── utils_post_processing.py       # Evaluation, feature names, and feature importance plotting
├── requirements.txt               # Python dependencies
└── README.md
```

### `utils_data_io.py`

- `load_hr_dataset()`  
  - Downloads the IBM HR dataset from Kaggle using `kagglehub`.
  - Drops non-informative columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`).
  - Maps `Attrition` to a binary target `AttritionFlag` (`No`→0, `Yes`→1).
  - Returns: `X`, `y`, `categorical_cols`, `numeric_cols`.

- `train_test_split_stratified(...)`  
  - Wrapper around `train_test_split` with stratification on `y`.

- `build_preprocessor(...)`  
  - Builds a `ColumnTransformer`:
    - `StandardScaler` for numeric features
    - `OneHotEncoder(handle_unknown="ignore")` for categorical features

### `utils_post_processing.py`

- `evaluate_classifier(...)`  
  - Computes **Accuracy**, **F1-score**, **ROC-AUC** and prints a classification report.

- `get_feature_names_from_preprocessor(...)`  
  - Extracts transformed feature names from the fitted `ColumnTransformer`
    (numeric + one-hot encoded categorical features).

- `plot_feature_importance(...)`  
  - Plots and saves top-N feature importances for a fitted `XGBClassifier` to  
    `xgb_feature_importance.png`.

---

## Installation & Setup

1. **Clone this repo** (or download the project folder):

   ```bash
   git clone <your-repo-url>.git
   cd <your-repo-folder>
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   # .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the project**

   - To start the pipeline, simply run the main script:
   
   ```bash
   python3 employee_attrition_main.py
   ```


---

## How to Run the Notebook

1. Start Jupyter:

   ```bash
   jupyter notebook
   ```

2. Open **`employee_attrition_eda.ipynb`**.

3. Run the cells in order. The notebook will:

   - Import `load_hr_dataset`, `build_preprocessor`, and `train_test_split_stratified`  
     from `utils_data_io.py`.
   - Build preprocessing pipelines and split into train/test sets.
   - Train models such as XGBoost, Random Forest, and Logistic Regression.
   - Use functions from `utils_post_processing.py` to:
     - Evaluate models (`evaluate_classifier`)
     - Inspect feature names (`get_feature_names_from_preprocessor`)
     - Plot feature importances for XGBoost (`plot_feature_importance`)

---

## Results (Example)

You can customize this section with your own final numbers from the notebook, for example:

- **XGBoost**: Accuracy ≈ 0.83, ROC-AUC ≈ 0.77  
- **Random Forest**: Accuracy ≈ 0.85, ROC-AUC ≈ 0.79  
- **Logistic Regression**: Accuracy ≈ 0.74, ROC-AUC ≈ 0.80  

(Replace with your actual results from the latest run.)

---

---

## Future Improvements / Work in Progress

The following features and enhancements are planned for upcoming updates:

- **Model Comparison**  
  Extend the evaluation pipeline to include a detailed comparison of **XGBoost**, **Logistic Regression**, and **Random Forest** models — highlighting trade-offs in accuracy, interpretability, and computational cost.

- **Class Imbalance Handling**  
  Implement **class weighting** and alternative sampling techniques (e.g., SMOTE, Random Undersampling) to better handle the imbalance between “Attrition” and “No Attrition” cases.

These improvements will help make the predictions more balanced and interpretable while providing a fair benchmark across models.






