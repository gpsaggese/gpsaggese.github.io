**Description**

CausalML is a Python library designed for causal inference and machine learning, enabling users to estimate treatment effects and understand the impact of interventions. It provides tools for both observational and experimental data, allowing researchers to identify causal relationships and make data-driven decisions.

Technologies Used
CausalML

- Supports various causal inference algorithms, including uplift modeling and propensity score matching.
- Offers tools for analyzing both continuous and discrete outcomes.
- Facilitates the estimation of treatment effects using methods such as causal forests and Bayesian approaches.

---

### Project 1: Customer Retention Analysis
**Difficulty**: 1 (Easy)  
**Project Objective**: The goal is to estimate the causal effect of a marketing campaign on customer retention rates using observational data. Students will identify how effective the campaign was in retaining customers compared to those who did not receive the campaign.

**Dataset Suggestions**: Use the "Online Retail" dataset available on Kaggle, which contains transactional data for a UK-based online retailer.

**Tasks**:
- Data Preprocessing:
  - Clean and preprocess the dataset to focus on customer transactions before and after the campaign.
  
- Define Treatment and Control Groups:
  - Identify customers who received the marketing campaign as the treatment group and those who did not as the control group.

- Estimate Treatment Effects:
  - Use CausalML to apply propensity score matching to estimate the causal effect of the marketing campaign on customer retention rates.

- Analyze Results:
  - Visualize the results using bar charts to compare retention rates between groups.

- Report Findings:
  - Summarize findings in a report detailing the estimated impact of the campaign.

---

### Project 2: Evaluating Educational Interventions
**Difficulty**: 2 (Medium)  
**Project Objective**: This project aims to evaluate the causal impact of an educational intervention (e.g., tutoring sessions) on student performance in mathematics, using a dataset of student test scores.

**Dataset Suggestions**: Utilize the "Students Performance in Exams" dataset available on Kaggle, which includes various features related to student demographics and performance.

**Tasks**:
- Data Exploration:
  - Perform exploratory data analysis (EDA) to understand the dataset and identify relevant features.

- Define Treatment:
  - Classify students who received tutoring as the treatment group and those who did not as the control group.

- Model Treatment Effects:
  - Implement causal inference methods using CausalML, such as causal forests, to estimate the effect of tutoring on test scores.

- Compare Performance:
  - Analyze differences in performance using visualizations and statistical tests to validate findings.

- Interpret Results:
  - Discuss the implications of the results for educational policy and future interventions.

---

### Project 3: Impact of Health Interventions on Patient Outcomes
**Difficulty**: 3 (Hard)  
**Project Objective**: The aim of this project is to assess the causal effect of a new health intervention (e.g., a medication or therapy) on patient recovery times in a clinical setting, utilizing a complex dataset with various confounding factors.

**Dataset Suggestions**: Use the "Heart Disease UCI" dataset available on Kaggle, which includes patient data, treatment types, and recovery outcomes.

**Tasks**:
- Data Preparation:
  - Clean and preprocess the dataset, focusing on relevant features such as treatment types, patient demographics, and recovery times.

- Identify Confounders:
  - Determine potential confounding variables that may affect the treatment outcomes.

- Causal Inference Modeling:
  - Apply CausalML techniques, such as Bayesian causal inference, to estimate the causal effect of the health intervention on recovery times while controlling for confounders.

- Evaluate Model Robustness:
  - Conduct sensitivity analyses to assess the robustness of the causal estimates under various assumptions.

- Communicate Findings:
  - Prepare a comprehensive report detailing the methodology, findings, and implications for clinical practice.

**Bonus Ideas (Optional)**:
- For Project 1, explore different marketing strategies and compare their effectiveness using the same causal framework.
- In Project 2, consider additional variables such as parental involvement or socioeconomic status and their interaction with the treatment effect.
- For Project 3, extend the analysis to include long-term effects of the intervention on patient health outcomes.

