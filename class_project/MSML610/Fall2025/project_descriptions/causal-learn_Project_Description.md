## Description
causal-learn is a Python library designed for causal inference and discovery, enabling researchers to identify and estimate causal relationships in data. It provides algorithms for causal discovery, supports DAGs and SEMs, and allows estimation of causal effects with regression and matching techniques.  

**Key Features:**  
- Algorithms for constraint-based and score-based causal discovery.  
- Estimation of causal effects using regression, matching, and SEM.  
- Visualization of causal graphs.  
- Support for Bayesian networks and graphical models.  

---

## Project 1: Causal Factors of Student Alcohol Consumption  
**Difficulty**: 1 (Easy)  

**Project Objective**  
Identify the causal impact of lifestyle factors (study habits, family background, alcohol consumption) on student grades, to uncover which behaviors most influence academic outcomes.  

**Dataset Suggestions**  
[Student Alcohol Consumption Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/student+performance) — includes demographic, social, and study-related attributes, with grades as the outcome.  

**Tasks**  
- **Data Preprocessing**: Encode categorical variables (e.g., parental education, family size).  
- **Causal Discovery**: Use causal-learn to explore relationships among alcohol consumption, study time, and final grade.  
- **Causal Effect Estimation**: Estimate effect of alcohol use and study hours on exam scores.  
- **ML Model Comparisons**:  
  - **Logistic Regression** to classify pass/fail.  
  - **Random Forest Classifier** for improved predictive accuracy.  
- **Visualization**: Draw DAGs showing causal paths.  

**Bonus Ideas (Optional)**  
- Investigate the role of family support as a mediator.  
- Compare causal results with purely predictive accuracy.  

---

## Project 2: Causal Impact of Lifestyle on Diabetes Risk  
**Difficulty**: 2 (Medium)  

**Project Objective**  
Assess how lifestyle factors (BMI, physical activity, blood pressure, smoking) causally affect diabetes risk and health outcomes.  

**Dataset Suggestions**  
[PIMA Indians Diabetes Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — widely used dataset with demographic + health metrics.  

**Tasks**  
- **EDA**: Explore distributions of BMI, glucose, and blood pressure.  
- **Causal Discovery**: Use PC or GES algorithms to infer causal structure among features.  
- **Causal Effect Estimation**: Apply regression adjustment to estimate effect of BMI and glucose on diabetes probability.  
- **ML Model Comparisons**:  
  - **Logistic Regression** for baseline prediction.  
  - **XGBoost Classifier** for advanced supervised modeling.  
- **Evaluation**: Compare causal estimates with ML predictive results.  

**Bonus Ideas (Optional)**  
- Perform subgroup analysis (e.g., by age group).  
- Test robustness with sensitivity analysis for unobserved confounders.  

---

## Project 3: Economic Factors and Employment Outcomes  
**Difficulty**: 3 (Hard)  

**Project Objective**  
Analyze causal effects of macroeconomic indicators (e.g., inflation, unemployment) on employment outcomes (e.g., wage growth, job satisfaction).  

**Dataset Suggestions**  
[World Development Indicators (World Bank, Kaggle)](https://www.kaggle.com/datasets/theworldbank/world-development-indicators) or [US Labor Statistics (FRED, Kaggle)](https://www.kaggle.com/datasets/bls/employment)  

**Tasks**  
- **Data Preprocessing**: Time-align economic indicators and employment outcomes.  
- **Causal Discovery**: Apply causal-learn to identify causal pathways between inflation, unemployment, and wages.  
- **Causal Effect Estimation**: Use Structural Equation Modeling (SEM) to quantify effects.  
- **Temporal Analysis**: Apply causal inference over rolling time windows.  
- **ML Model Comparisons**:  
  - **Random Forest Regressor** to predict wage growth.  
  - **LSTM (Keras)** to capture temporal dependencies.  
- **Visualization**: Draw causal DAGs and show temporal changes.  

**Bonus Ideas (Optional)**  
- Compare causal relationships across countries or demographic groups.  
- Perform scenario analysis (e.g., simulate effect of inflation shocks on wages).  

