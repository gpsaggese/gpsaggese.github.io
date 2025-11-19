# API Notebook – Renewable Energy Forecasting

This notebook demonstrates how to use the renewable energy forecasting project programmatically.  
It provides two clean API interfaces:

1. A **command-line style API** using Python’s `subprocess`  
2. A **pure Python function-level API** using utilities from the project

These interfaces make the project reusable, scriptable, and easy to integrate into other notebooks or applications.

---

## 1. Command-Line Style API

The project already includes two scripts:

- `scripts/make_features.py` – builds features (raw → processed)  
- `scripts/train.py` – trains the baseline model  

In the API notebook, I wrap these scripts using Python's `subprocess` module.  
This allows me to run the entire pipeline **without opening a terminal**.

### What this provides

- A simple “shell-like API”  
- Automatic triggering of the feature creation pipeline  
- Automatic triggering of the training pipeline  
- Output printed inside the notebook  

This approach is useful for:

- automation  
- batch experiments  
- scheduled workflows  
- demos and reproducible runs  

---

## 2. Python-Level API

In addition to shell-style wrappers, the notebook also demonstrates a **pure Python API**.

I import the following functions from `RenewableEnergy_utils.py`:

- `load_raw_solar`  
- `add_basic_time_features`  
- `save_processed`  
- `PROCESSED_DIR`  

Using these, I define a high-level function:

**`run_full_pipeline_and_return_metrics()`**

This function performs:

1. Load raw solar data  
2. Create time-based features  
3. Save processed dataset (e.g., `train_from_api.csv`)  
4. Train a Random Forest baseline model  
5. Compute MAE and RMSE  
6. Return the metrics as a Python dictionary  

### Why this is useful

- integrates easily inside notebooks  
- allows programmatic experiments  
- great for comparing multiple models  
- avoids shell commands  
- ideal for advanced workflows  

---

## 3. Example Usage

Inside the API notebook, after defining the function, I run:


This confirms the entire pipeline works end-to-end from within Python.

---

## 4. Purpose of This Notebook

This notebook shows how the project can be controlled at two levels:

### **CLI-level API**
- simple wrappers  
- good for automation  
- works like a small Python CLI tool  

### **Python-level API**
- more flexible  
- integrates well with Jupyter  
- easier for iterative modeling  
- supports advanced experimentation  

Both approaches make the project:

- reusable  
- extendable  
- easy to automate  
- easy to integrate into MLflow or other tools  

This completes the programmatic interface for the renewable energy forecasting project.

