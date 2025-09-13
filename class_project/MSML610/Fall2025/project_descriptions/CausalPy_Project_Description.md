## Description  
CausalPy is a Python library designed for causal inference, enabling data scientists to analyze the effects of interventions and understand causal relationships in data. It provides a user-friendly interface for estimating causal effects through various methods, including propensity score matching and regression-based approaches.  

**Features of CausalPy:**  
- Implements state-of-the-art causal inference methods for estimating treatment effects.  
- Allows for easy integration with popular data manipulation libraries like Pandas.  
- Provides visualization tools for causal graphs and treatment effect estimates.  

---

## Project 1: Understanding the Impact of Advertising on Sales  
**Difficulty**: 1 (Easy)  

**Project Objective**: Estimate how different advertising channels (TV, online, print) affect sales figures for a retail company, identifying the most effective channel.  

**Dataset Suggestions**:  
- **Dataset**: "Advertising Dataset" on Kaggle  
- **Link**: [Advertising Dataset](https://www.kaggle.com/datasets/ashydv/advertising-dataset)  

**Tasks**:  
- **Data Preprocessing**: Clean and prepare the dataset, ensuring proper formats.  
- **Exploratory Analysis**: Visualize relationships between sales and advertising spend.  
- **Causal Setup**: Define treatment groups based on advertising channels.  
- **Causal Estimation**: Apply propensity score matching in CausalPy to estimate treatment effects.  
- **Interpretation**: Summarize which channel has the largest causal impact on sales.  

**Bonus Idea (Optional)**: Compare results against a simple linear regression to highlight the benefit of causal methods.  

---

## Project 2: Evaluating the Effect of Training on Earnings  
**Difficulty**: 2 (Medium)  

**Project Objective**: Investigate how participation in a training program affects individuals’ earnings, optimizing for unbiased causal effect estimation.  

**Dataset Suggestions**:  
- **Dataset**: "Lalonde Dataset" (classic causal inference dataset for job training vs. control)  
- **Link**: [Lalonde Dataset](https://github.com/robjellis/lalonde)  

**Tasks**:  
- **Data Preprocessing**: Load the dataset, clean covariates, and ensure treatment vs. control groups are defined.  
- **Exploratory Analysis**: Compare distributions of covariates between treated and control groups.  
- **Causal Setup**: Define treatment = program participation, control = non-participants.  
- **Causal Estimation**: Use propensity score weighting or matching in CausalPy to estimate the average treatment effect on earnings.  
- **Reporting**: Visualize balance before/after matching and summarize the causal estimate.  

**NOTE**: The Lalonde dataset is widely used for benchmarking causal methods and works well for testing multiple approaches.  

**Bonus Idea (Optional)**: Apply multiple methods (matching, weighting, regression adjustment) and compare results for robustness.  

---

## Project 3: Analyzing the Effects of Healthcare Policies on Patient Outcomes  
**Difficulty**: 3 (Hard)  

**Project Objective**: Estimate the causal effect of healthcare interventions or policy changes on patient outcomes using observational healthcare data.  

**Dataset Suggestions**:  
- **Dataset**: "Healthcare Dataset" on Kaggle  
- **Link**: [Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)  

**Tasks**:  
- **Data Preprocessing**: Focus on key outcome variables (e.g., health scores, recovery rates) and align them with policy or treatment indicators.  
- **Exploratory Analysis**: Visualize outcome trends across treatment vs. control groups.  
- **Causal Setup**: Define treatment = patients exposed to a healthcare intervention, control = those without.  
- **Causal Estimation**: Apply difference-in-differences (DiD) or regression discontinuity in CausalPy to estimate effects.  
- **Interpretation**: Provide a detailed policy-relevant interpretation of results.  

**NOTE**: To keep the project feasible, select a **subset of patients, time windows, or policy changes** rather than using the full dataset.  

**Bonus Idea (Optional)**: Extend the analysis with lagged outcomes to measure long-term effects of interventions.  
