# xgboost.example.md

## 1. Example Application Overview

This document presents a **complete example** of how the Employee Attrition Prediction project uses the
XGBoost-based API layer described in `xgboost.API.md`.

The goal of the application is to:

- Predict whether an employee is likely to **leave the company** (attrition = Yes/No).  
- Use demographic, job-related, and satisfaction features from the **IBM HR Analytics** dataset.  
- Build a **reusable modeling pipeline** that combines preprocessing, XGBoost, and evaluation.  
- Provide **interpretable insights** for HR (which groups are at risk and why).

Instead of focusing on code, this document explains:

- The end-to-end workflow in the main notebook.  
- How the API layer (pipeline + evaluation + threshold tuning) is used.  
- The main results, analysis, and final takeaway.

---

## 2. Dataset and Problem Setting

### 2.1 Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance** dataset from Kaggle.

Each row is an employee, with columns such as:

- **Demographics**: Age, Gender, MaritalStatus, Education, EducationField  
- **Job & compensation**: JobRole, Department, JobLevel, MonthlyIncome, OverTime, YearsAtCompany  
- **Satisfaction & work–life**: JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, RelationshipSatisfaction  
- **Career history**: NumCompaniesWorked, TotalWorkingYears, YearsInCurrentRole, YearsWithCurrManager  

The target label is:

- `Attrition` ∈ {**Yes**, **No**}

A numeric flag `AttritionFlag` is created:

- 1 = employee left (Yes)  
- 0 = employee stayed (No)

### 2.2 Class imbalance

Most employees **do not leave**, so the dataset is **imbalanced**:
there are many more `Attrition = No` cases than `Yes`.  
This matters because a naïve classifier can achieve high overall accuracy by predicting “No” most of the time,
but that would be useless for detecting at-risk employees.

The project explicitly handles this imbalance inside the XGBoost configuration and in the baseline models.

---

## 3. API Layer Used in the Example

The application relies on a small **API layer** that sits on top of native libraries:

1. A **scikit-learn Pipeline** that combines:
   - A `ColumnTransformer` to:
     - one-hot encode categorical features
     - scale numeric features  
   - An `XGBClassifier` model configured with:
     - boosting parameters (trees, depth, learning rate, subsampling)
     - `scale_pos_weight` to handle class imbalance.

2. A generic **`evaluate_model(...)` helper** that:
   - Trains any pipeline.
   - Computes Accuracy, F1-score, ROC-AUC.
   - Prints a classification report and confusion matrix.
   - Returns scores so models can be compared in a table.

3. A small **threshold tuning routine** that:
   - Uses predicted probabilities from the trained pipeline.
   - Sweeps through different thresholds (0.1–0.9).
   - Shows how precision, recall, and F1-score change.
   - Highlights thresholds that favor **higher F1** vs **higher recall**.

This API layer lets the notebook read like a story about the *problem* (attrition) rather than about low-level data wrangling.

---

## 4. End-to-End Notebook Workflow

### 4.1 Data loading and basic structure

The notebook:

1. Downloads the dataset via **KaggleHub** or loads a local CSV.  
2. Displays basic information (`info`, `describe`) to understand column types.  
3. Creates derived helpers for analysis:
   - `AttritionFlag` (0/1)  
   - `AgeBin` (18–25, 26–35, 36–45, 46–55, 56+)  
   - `WorkLifeBalanceLabel` (Bad, Good, Better, Best

These are mainly used for **EDA**, not fed directly into the model.

### 4.2 Exploratory Data Analysis (EDA)

The EDA section uses **pandas**, **matplotlib**, and **seaborn** to explore attrition patterns.

Key visual analyses include:

1. **Overall attrition distribution**  
   - A count plot of Attrition Yes vs No.  
   - Shows that leavers are a minority → confirms class imbalance.

2. **Attrition by age group**  
   - Attrition rate computed per `AgeBin`.  
   - Younger employees (e.g., 18–25 or 26–35) show **higher attrition rates** than older groups.

3. **Attrition by job role**  
   - Attrition rates by `JobRole`.  
   - Certain roles (e.g., Sales Representative, Laboratory Technician) tend to have **higher attrition**.

4. **Attrition vs work–life balance**  
   - Groups employees by `WorkLifeBalanceLabel`.  
   - Poorer work–life balance categories show **higher attrition**.

5. **Overtime and monthly income**  
   - Count plot of `OverTime` vs `Attrition`:
     - Employees working overtime more often have **higher attrition rates**.  
   - Boxplot of `MonthlyIncome` vs `Attrition`:
     - Employees who leave tend to be concentrated in **lower income ranges**.

6. **Correlation heatmap**  
   - Correlation matrix of numeric features including `AttritionFlag`.  
   - Tenure-related features (Age, TotalWorkingYears, YearsAtCompany, JobLevel, MonthlyIncome) are all positively correlated.  
   - `AttritionFlag` shows **weak negative correlations** with seniority, pay, and satisfaction, and slightly positive with distance and number of previous companies.  
   - No single numeric feature perfectly predicts attrition — justifying the use of a **non-linear model** like XGBoost.

### 4.3 Preprocessing design

For modeling, the notebook:

- Defines **categorical columns** (e.g., JobRole, Department, MaritalStatus, OverTime).  
- Defines **numeric columns** (e.g., Age, MonthlyIncome, YearsAtCompany, JobSatisfaction).  
- Drops ID-like or constant columns (EmployeeNumber, EmployeeCount, Over18, StandardHours).  
- Excludes EDA-only helper columns (AgeBin, WorkLifeBalanceLabel) from modeling.

These are then passed into a `ColumnTransformer` that:

- **One-hot encodes** the categorical features.  
- **Standardizes** the numeric features.

All of this preprocessing is encapsulated inside the pipeline, so the notebook never manually handles encoded matrices.

### 4.4 Model training with XGBoost (API layer)

The main model is an **XGBoost classifier**, configured with:

- A moderate number of trees and depth.  
- A small learning rate.  
- Row and column subsampling.  
- `scale_pos_weight` set to *(# negatives / # positives)* to compensate for class imbalance.

The XGBoost classifier is wrapped inside a **Pipeline** along with the preprocessor.  
The training step then becomes conceptually simple:

- Fit the pipeline on the training data.  
- Let the pipeline internally:
  - Transform raw DataFrames into model-ready arrays.  
  - Train XGBoost on the transformed features.

This is exactly the pattern described in `xgboost.API.md` and demonstrated in `xgboost.API.ipynb`.

### 4.5 Evaluation with default threshold

Using the **evaluation helper**, the notebook reports for XGBoost:

- **Accuracy** on the test set.  
- **F1-score** for the positive class (attrition = Yes).  
- **ROC-AUC**, measuring the quality of ranking across probabilities.  
- Full **classification report** (precision, recall, F1 per class).  
- **Confusion matrix**, showing true/false positives and negatives.

This gives a baseline view of how well the model performs when using the **default 0.5 threshold**.

### 4.6 Threshold tuning: balancing F1 vs recall

The notebook then performs **threshold tuning** on top of the trained XGBoost pipeline.

Key steps:

- Use the pipeline to obtain **predicted probabilities** for attrition (class 1).  
- For thresholds from 0.1 to 0.9, compute:
  - Precision, recall, F1-score for the positive class.  
- Highlight two thresholds:
  - **Threshold = 0.4** → best F1 in the tested range.  
  - **Threshold = 0.3** → higher recall, useful if missing leavers is more costly.

From the project’s run:

- At **threshold 0.4** (F1-focused):  
  - Overall accuracy ≈ 0.80.  
  - Recall for attrition ≈ 0.51.  
  - Better balance between catching leavers and avoiding too many false alarms.

- At **threshold 0.3** (recall-focused):  
  - Overall accuracy drops to ≈ 0.75.  
  - Recall for attrition ≈ 0.60 (we catch more leavers).  
  - More false positives, which might be acceptable if early HR intervention is cheap.

This illustrates how the **API layer + threshold tuning** allows aligning the model with **business priorities**.

### 4.7 Baseline models: Logistic Regression and Random Forest

To benchmark XGBoost, the notebook also trains:

- **Logistic Regression** with balanced class weights.  
- **Random Forest** with balanced class weights.

Both are wrapped in the **same preprocessing pipeline** and evaluated with the **same helper function**.  
This ensures a **fair comparison**, since all models see the same inputs.

Findings (qualitative):

- Logistic Regression tends to underfit slightly and captures mainly linear trends.  
- Random Forest is competitive but typically lags behind XGBoost in ROC-AUC and F1 for the positive class.  
- XGBoost emerges as the **best-performing model** across the main metrics.

### 4.8 Feature importance and SHAP analysis

The notebook explores **why** the model makes its predictions:

1. **Feature importance from XGBoost**  
   - Extracts the trained model and reconstructed feature names from the pipeline.  
   - Displays a ranked list of most important features.  
   - Common top drivers include:
     - OverTime (Yes/No)  
     - MonthlyIncome  
     - YearsAtCompany / total experience  
     - JobSatisfaction and other satisfaction scores  
     - Age, JobLevel

2. **SHAP summary plot**  
   - Uses SHAP to compute feature contributions for a sample of employees.  
   - Visualizes which features **push probabilities up or down** for attrition.  
   - Example interpretations:
     - Working overtime, low satisfaction, low tenure, and lower salary **increase** predicted attrition.  
     - Higher tenure, higher income, and better work–life balance **decrease** predicted attrition.

This turns the Gradient Boosted Trees model into something **explainable** for HR and managers.

### 4.9 Clustering employee profiles

As an extra analysis, the notebook applies **KMeans clustering** on a few numeric features:

- Age  
- MonthlyIncome  
- TotalWorkingYears  
- JobSatisfaction

The model finds **4 clusters** of employees, and the attrition rate is computed within each cluster.

Examples of cluster profiles:

- **Cluster 1 – Senior, highly paid, low attrition (~7%)**  
  - Oldest, longest tenure, highest income, moderate satisfaction.  
  - Very stable group.

- **Cluster 2 – Mid-30s, lower satisfaction, higher attrition (~20%)**  
  - Younger than Cluster 1, lower income, especially **low job satisfaction**.  
  - One of the riskiest clusters.

- **Cluster 3 – Youngest, early career, higher attrition (~20%)**  
  - Very young, low tenure, lower income but reasonably high satisfaction.  
  - Likely early-career employees more open to changing jobs.

- **Cluster 0 – Mid-career, moderate risk (~12%)**  
  - Middle of the road for age, income, tenure, and satisfaction.

This complements the supervised model by showing **natural segments** of employees and their attrition tendencies.

### 4.10 Error analysis

The notebook also inspects **where the model struggles** by examining misclassified employees.  
For example, among incorrectly predicted cases, some roles appear more often (e.g., Research Scientist, Laboratory Technician, Sales Representative).

This suggests:

- For certain roles, attrition may be harder to predict with the available features.  
- Additional role-specific features or context might further improve the model.

---

## 5. Insights

Putting everything together, the example application shows that:

1. A properly configured **XGBoost + preprocessing pipeline** can predict employee attrition with solid performance.  
   - Good ROC-AUC (the model ranks likely leavers ahead of non-leavers).  
   - Reasonable F1-score for the positive class after **threshold tuning**.  

2. **Threshold tuning** allows choosing between:
   - A setting that focuses on overall F1 (e.g., threshold 0.4).  
   - A setting that favors higher recall on leavers (e.g., threshold 0.3).  

3. **Feature importance and SHAP** indicate that employees are more likely to leave when:
   - They are younger and have shorter tenure.  
   - They work overtime.  
   - Their monthly income is relatively low.  
   - Their job, environment, or work–life satisfaction is poor.

4. **Clustering** reveals interpretable groups of employees with different risk profiles,
   which can guide targeted retention strategies.

5. **Baseline comparisons** show that XGBoost outperforms simpler models (Logistic Regression)
   and is competitive with or better than Random Forest, especially in ranking high-risk employees.

---

## 6. How This Example Uses the API Layer

This application illustrates the strength of the API layer defined in the project:

- The main notebook never worries about:
  - Manually encoding categories.  
  - Manually scaling numeric features.  
  - Passing low-level arrays into XGBoost.

- Instead, it:
  - Constructs a **single pipeline** that handles preprocessing + XGBoost.  
  - Uses a generic **evaluation helper** to compare multiple models.  
  - Applies **threshold tuning** on top of `predict_proba`.  
  - Reuses the same logic to plug in Logistic Regression and Random Forest.

This architecture makes it easy to:

- Extend the project (add new models, new features, or new evaluation metrics).  
- Apply the same approach to **other tabular prediction problems**, not just attrition.  
- Keep the notebook focused on **business questions and interpretation**, while the API layer handles technical details.

## 7. Overall Summary of Results

### 7.1 Performance recap

Across all experiments, the models learned useful patterns about who is likely to leave, but the **minority “Attrition = Yes” class** remained challenging to predict perfectly.

From a representative run:

- **Class balance**
  - About **16%** of employees had `Attrition = Yes`, and **84%** had `Attrition = No`.
  - This imbalance means that a model can achieve high accuracy by mostly predicting “No”, so we focused on **F1** and **ROC-AUC** rather than accuracy alone.

- **XGBoost with class weighting (scale_pos_weight)**
  - Accuracy ≈ **0.83**
  - F1 (attrition = Yes) ≈ **0.42** at the default 0.5 threshold  
  - ROC-AUC ≈ **0.78**
  - Good ranking ability (who is more likely to leave), but default threshold slightly under-detects leavers.

- **Logistic Regression (class_weight="balanced")**
  - Accuracy ≈ **0.75**
  - F1 (Yes) ≈ **0.45**
  - ROC-AUC ≈ **0.80**
  - Simpler and more interpretable, with better recall on the positive class, but lower overall accuracy.

- **Random Forest (class_weight="balanced")**
  - Accuracy ≈ **0.84**
  - F1 (Yes) ≈ **0.18**
  - ROC-AUC ≈ **0.79**
  - Very strong on predicting “stayers” (few false positives) but misses many leavers, so its F1 for attrition is poor.

Overall, **XGBoost + proper class weighting** gave a strong balance of accuracy, ROC-AUC, and F1, and was chosen as the **primary model**.

---

### 7.2 Impact of class imbalance

The strong skew (many more “No” than “Yes”) affected the results in several ways:

- **Raw accuracy is misleading**  
  A model can exceed 80% accuracy while still missing a large fraction of leavers. That’s why we explicitly report:
  - **F1-score for the positive class** (how well we balance precision and recall on attrition),
  - **Recall** (how many leavers we actually catch),
  - **ROC-AUC** (how well probabilities rank high-risk employees).

- **Class weighting was necessary**  
  - In XGBoost, we used `scale_pos_weight = (# negatives / # positives)` to tell the model that misclassifying a leaver is more costly than misclassifying a stayer.
  - In Logistic Regression and Random Forest, we used `class_weight="balanced"` for the same reason.
  - Without these adjustments, the models tend to predict “No” even more aggressively, further hurting recall on attrition.

Even with these techniques, predicting rare leavers remains harder than predicting stayers, which is expected in real HR data.

---

### 7.3 Threshold tuning: F1 vs recall

By default, all classifiers use a **0.5 probability threshold** to convert scores into Yes/No predictions.  
However, for HR, missing an at-risk employee can be more costly than occasionally flagging a stable one.  
The project therefore uses the API layer to **scan thresholds from 0.1 to 0.9** and examine precision, recall, and F1.

Two thresholds stand out in the XGBoost runs:

- **Threshold = 0.4 – best F1 (balanced)**
  - Accuracy ≈ **0.80**
  - Recall (Yes) ≈ **0.51**
  - F1 (Yes) ≈ **0.45**
  - Interpretation:  
    - Good **overall balance** between correctly catching leavers and avoiding too many false alarms.
    - Suitable when HR wants a pragmatic compromise between workload and missed risk.

- **Threshold = 0.3 – higher recall (risk-averse)**
  - Accuracy ≈ **0.75**
  - Recall (Yes) ≈ **0.60**
  - F1 (Yes) ≈ **0.43**
  - Interpretation:  
    - The model becomes more **sensitive** to attrition risk: more leavers are flagged.
    - This comes at the cost of more false positives (more “stable” employees flagged as at risk).
    - Suitable when the organization is more concerned about **not missing** potential leavers than about extra follow-up.

In practice:

- **Use ~0.4** if you want the **best F1 and good overall accuracy** (balanced view).  
- **Use ~0.3** if your **priority is recall** of at-risk employees, and HR can handle a higher number of follow-ups.

---

### 7.4 Takeaway



- The combination of **XGBoost + scikit-learn pipeline + class weighting + threshold tuning** yields a robust and flexible framework for predicting employee attrition on an imbalanced dataset.
- The models confirm intuitive patterns:
  - Early-career employees with lower income, overtime, and poor satisfaction are at **highest risk**.
  - Long-tenured, better-paid, and more satisfied employees have **much lower** attrition rates.
- The API layer makes it easy to:
  - Swap models (XGBoost vs Logistic Regression vs Random Forest),
  - Adjust thresholds to match business risk tolerance,
  - Add interpretability via feature importance and SHAP.

This example shows how a modern ML stack can move beyond “one number accuracy” and deliver **actionable, interpretable insights** for HR decision-making on real, imbalanced data.



