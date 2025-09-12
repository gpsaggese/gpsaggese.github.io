## Description  
CausalML is a Python library that provides machine learning–based methods for estimating treatment effects and uplift modeling. It enables users to quantify the causal impact of interventions in observational and experimental data, going beyond simple correlations to estimate true cause–effect relationships. CausalML supports tree-based learners, meta-learners (e.g., T-, S-, X-learners), and advanced methods such as causal forests and Bayesian approaches.  

**Features of CausalML:**  
- Implements uplift modeling for marketing and policy evaluation.  
- Provides meta-learners and causal forests for heterogeneous treatment effect estimation.  
- Supports continuous and discrete outcomes for flexible analysis.  
- Integrates easily with Pandas and Scikit-learn workflows.  

---

## Project 1: Estimating the Effect of Promotions on Online Purchases  
**Difficulty**: 1 (Easy)  

**Project Objective**:  
Estimate the causal effect of promotional exposure on customer purchasing behavior, identifying how effective promotions are at increasing conversions.  

**Dataset Suggestions**:  
- **Dataset**: Online Shoppers Purchasing Intention Dataset (UCI ML Repository)  
- **Link**: [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)  

**Tasks**:  
- **Data Preprocessing**: Load and clean the dataset, focusing on promotional/referral variables and purchase outcomes.  
- **Define Treatment & Outcome**: Treatment = customers exposed to a promotional channel; Control = customers not exposed; Outcome = whether a purchase occurred.  
- **Estimate Treatment Effect**: Use CausalML (e.g., uplift modeling or propensity score matching) to estimate causal effects.  
- **Visualization**: Plot uplift curves to illustrate how promotion effectiveness varies across customer groups.  
- **Interpretation**: Summarize which types of customers respond most positively to promotions.  

**Bonus Idea (Optional)**: Compare results across different referral channels (e.g., paid ads vs. organic traffic).  

---

## Project 2: Evaluating Loan Approval Interventions on Repayment Rates  
**Difficulty**: 2 (Medium)  

**Project Objective**:  
Analyze the causal effect of different loan approval strategies on repayment behavior, optimizing for policies that balance access to credit with repayment likelihood.  

**Dataset Suggestions**:  
- **Dataset**: "Give Me Some Credit" dataset on Kaggle  
- **Link**: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data)  

**Tasks**:  
- **Data Exploration**: Explore credit history, demographics, and loan repayment variables.  
- **Define Treatment**: Treatment = customers given a loan; Control = customers not approved (or simulated subgroup).  
- **Estimate Treatment Effects**: Use causal forests or uplift trees in CausalML to estimate the impact of loan approval on repayment rates.  
- **Heterogeneous Effects**: Identify subgroups (e.g., by income level or credit score) where loan approval has different effects.  
- **Interpretation**: Provide insights for designing fairer, more effective loan policies.  

**Bonus Idea (Optional)**: Compare causal ML estimates with logistic regression to highlight differences in subgroup-level effects.  

---

## Project 3: Measuring the Impact of Lifestyle Programs on Diabetes Outcomes  
**Difficulty**: 3 (Hard)  

**Project Objective**:  
Estimate the causal effect of lifestyle interventions (diet, exercise programs) on diabetes outcomes (e.g., HbA1c levels, health progression), accounting for confounders.  

**Dataset Suggestions**:  
- **Dataset**: Diabetes 130-US hospitals for years 1999–2008 (UCI ML Repository)  
- **Link**: [Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)  

**Tasks**:  
- **Data Preprocessing**: Clean the dataset, focusing on patients with diabetes, treatments received, and outcome measures (readmission, lab results).  
- **Define Treatment**: Treatment = patients enrolled in lifestyle programs or receiving specific therapies; Control = those without.  
- **Estimate Treatment Effects**: Apply causal forests or X-learners in CausalML to estimate the effect of lifestyle interventions on outcomes.  
- **Robustness Checks**: Conduct sensitivity analysis to test robustness of treatment effect estimates under confounding.  
- **Interpretation**: Summarize which interventions are most effective and for which patient subgroups.  

**Bonus Idea (Optional)**: Extend the analysis to predict long-term effects by modeling sequential interventions (diet + exercise + medication).  
