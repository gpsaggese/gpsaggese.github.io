## Description  
pgmpy is a Python library designed for probabilistic graphical models, particularly Bayesian networks and Markov networks. It provides a flexible framework for constructing, manipulating, and performing inference on graphical models, making it a valuable tool for data scientists interested in probabilistic reasoning.  

**Features of pgmpy:**  
- **Graphical Model Creation**: Create and visualize Bayesian networks and Markov models.  
- **Inference Algorithms**: Apply exact and approximate inference (variable elimination, belief propagation).  
- **Parameter Learning**: Learn parameters from data using Maximum Likelihood Estimation (MLE) or Bayesian Estimation.  
- **Model Evaluation**: Assess accuracy using likelihood-based and classification metrics.  

---

## Project 1: Predicting Student Performance  
**Difficulty**: 1 (Easy)  

**Project Objective**  
Build a Bayesian network to predict student exam performance based on attendance, study habits, and demographics.  

**Dataset Suggestions**  
[Student Performance Dataset (Kaggle, UCI)](https://www.kaggle.com/datasets/whenamancodes/student-performance)  

**Tasks**  
- **Data Preprocessing**: Clean data and discretize variables for Bayesian modeling.  
- **Bayesian Network**: Build a network in pgmpy linking study habits, attendance, and grades.  
- **Parameter Learning**: Train using Maximum Likelihood Estimation.  
- **Inference**: Predict probability of passing given new evidence.  
- **ML Model Comparisons**:  
  - **Decision Tree Classifier** (scikit-learn).  
  - **Logistic Regression** baseline.  
- **Evaluation**: Use accuracy, precision, recall, F1-score for all models.  

**Bonus Ideas (Optional)**  
- Add socioeconomic background features for more complex inference.  
- Compare model interpretability between Bayesian networks and black-box models.  

---

## Project 2: Health Risk Assessment  
**Difficulty**: 2 (Medium)  

**Project Objective**  
Develop a probabilistic model to assess cardiovascular disease risk using lifestyle and medical factors.  

**Dataset Suggestions**  
[Heart Disease UCI Dataset (Kaggle)](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)  

**Tasks**  
- **EDA**: Explore relationships between age, cholesterol, exercise habits, and disease presence.  
- **Bayesian Network**: Build dependency graph between lifestyle/clinical features and risk.  
- **Parameter Learning**: Apply Bayesian Estimation with priors.  
- **Inference**: Compute probabilities of heart disease given lifestyle inputs.  
- **ML Model Comparisons**:  
  - **Random Forest Classifier** for non-linear relationships.  
  - **Support Vector Machine (SVM)** for comparison.  
- **Evaluation**: ROC-AUC, precision, recall, confusion matrix.  

**Bonus Ideas (Optional)**  
- Perform **sensitivity analysis** to see which variables most affect predictions.  
- Compare predictions across age groups or gender.  

---

## Project 3: Fraud Detection in Financial Transactions  
**Difficulty**: 3 (Hard)  

**Project Objective**  
Create a Bayesian network for anomaly detection in financial transactions, comparing performance against ML baselines.  

**Dataset Suggestions**  
[Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

**Tasks**  
- **Data Preprocessing**: Handle class imbalance via undersampling/oversampling.  
- **Bayesian Network**: Model dependencies among transaction amount, time, location, and fraud risk.  
- **Parameter Learning**: Train using both MLE and Bayesian Estimation.  
- **Inference**: Use belief propagation to classify transactions as fraudulent or normal.  
- **ML Model Comparisons**:  
  - **XGBoost Classifier** (highly effective for tabular fraud detection).  
  - **Random Forest Classifier** baseline.  
- **Evaluation**: Precision, recall, F1-score, ROC-AUC (focus on minimizing false negatives).  

**Bonus Ideas (Optional)**  
- Combine Bayesian network with anomaly detection methods (Isolation Forest).  
- Deploy a streaming fraud detection pipeline for real-time inference.  

---
