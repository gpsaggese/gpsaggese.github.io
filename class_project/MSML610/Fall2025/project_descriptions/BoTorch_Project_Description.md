## Description  
BoTorch is a library built on PyTorch that facilitates Bayesian optimization, enabling users to optimize expensive-to-evaluate functions efficiently. It provides a flexible interface for defining and optimizing objective functions using Gaussian processes, allowing for the incorporation of prior knowledge and uncertainty quantification.  

**Technologies Used**  
BoTorch  
- Implements advanced Bayesian optimization techniques.  
- Supports multi-fidelity and multi-objective optimization.  
- Integrates seamlessly with PyTorch for deep learning applications.  

---

### Project 1: Hyperparameter Optimization for Classification Models  
**Difficulty**: 1 (Easy)  

**Project Objective**:  
Use BoTorch to optimize hyperparameters of a classification model (e.g., Random Forest) to achieve the best performance on a tabular dataset.  

**Dataset Suggestions**:  
- **Dataset**: Heart Disease UCI Dataset  
- **Link**: [Heart Disease UCI on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  

**Tasks**:  
- **Data Preprocessing**: Clean the dataset, handle missing values, and encode categorical variables.  
- **Define Objective Function**: Create a function that evaluates model accuracy given a set of hyperparameters.  
- **Set Up BoTorch**: Apply Bayesian optimization to tune hyperparameters such as `n_estimators`, `max_depth`, and `min_samples_split`.  
- **Model Evaluation**: Use cross-validation to assess the best configuration and compare against random search or grid search.  
- **Visualization**: Plot convergence of the optimization process and accuracy improvements.  

---

### Project 2: Multi-Objective Optimization for Renewable Energy Forecasting  
**Difficulty**: 2 (Medium)  

**Project Objective**:  
Leverage BoTorch’s multi-objective optimization to forecast renewable energy production while minimizing prediction error and model complexity.  

**Dataset Suggestions**:  
- **Dataset**: Global Wind and Solar Energy Dataset  
- **Link**: [Renewable Energy Production (Kaggle)](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)  

**Tasks**:  
- **Data Preparation**: Load energy production data, handle missing timestamps, and create lag features.  
- **Define Objectives**: Create two objectives: (1) minimize RMSE of forecasts, (2) minimize number of features/model complexity.  
- **Set Up BoTorch**: Implement multi-objective Bayesian optimization to balance accuracy and simplicity.  
- **Pareto Front Analysis**: Visualize trade-offs between error and complexity.  
- **Evaluation**: Compare optimized model results against a baseline ARIMA or simple regression model.  

---

### Project 3: Optimizing Compound Selection from ChEMBL Subset  
**Difficulty**: 3 (Hard)  

**Project Objective**:  
Apply BoTorch to optimize the selection of chemical compounds from a ChEMBL-derived subset to balance predicted bioactivity (lower IC50 → better potency) and experimental cost/effort.  

**Dataset Suggestions**:  
- **Dataset**: *Human Acetylcholinesterase Dataset from ChEMBL* — includes compounds tested against human acetylcholinesterase, with IC50 (“standard_value”) measurements.
- **Optional alternate/more complex**: *MultiTarget Bioactivity ChEMBL* — includes multiple protein targets for compounds, allowing multi-target optimization.

**Tasks**:  
- **Data Loading & Preprocessing**  
  - Download the ChEMBL subset dataset (e.g. “Human Acetylcholinesterase ...”).  
  - Clean data: remove missing or non-numeric IC50 values, convert units if needed (e.g. ensure all µM or nM), log transform IC50 if useful.  
  - Extract or compute molecular descriptors or fingerprints (e.g. Morgan fingerprints) for compounds.  

- **Define Objective Function(s)**  
  - Primary objective: predicted potency (e.g. minimize log(IC50) or some proxy).  
  - Secondary objective: minimize “testing cost / experimental effort” (could be approximated by e.g. molecular weight, number of rings, synthetic accessibility, or just a constant cost per compound).  

- **Set Up BoTorch Optimization**  
  - Use BoTorch’s Bayesian optimization to choose among a candidate pool of compounds the ones expected to give best trade-off between potency and cost.  
  - If using the MultiTarget dataset: you could also set up multi-objective optimization over potency for multiple targets.  

- **Evaluation**  
  - Split the dataset: train a surrogate model (e.g. Gaussian Process) to predict potency from descriptors.  
  - Validate predictions on held-out compounds.  
  - Compare optimized selection (what BoTorch suggests) vs baseline selections (random sampling; top potency only; etc.).  

- **Visualization & Interpretation**  
  - Visualize the Pareto front (potency vs cost).  
  - Show chemical structures (SMILES) of selected compounds and discuss trade-offs.  


**Bonus Ideas (Optional)**:  
- Project 1: Extend comparison to random forest vs. gradient boosting models.  
- Project 2: Add multi-fidelity optimization (e.g., low-res vs. high-res time series data).  
- Project 3: Integrate additional biological or genetic datasets to enhance compound prioritization.  
