**Description**

CausalPy is a Python library designed for causal inference, enabling data scientists to analyze the effects of interventions and understand causal relationships in data. It provides a user-friendly interface for estimating causal effects through various methods, including propensity score matching and instrumental variable analysis.

Features of CausalPy:
- Implements state-of-the-art causal inference methods for estimating treatment effects.
- Allows for easy integration with popular data manipulation libraries like Pandas.
- Provides visualization tools for causal graphs and treatment effect estimates.

---

### Project 1: Understanding the Impact of Advertising on Sales
**Difficulty**: 1 (Easy)

**Project Objective**: 
Analyze how different advertising channels (TV, online, print) impact sales figures for a retail company, optimizing for understanding the most effective channel.

**Dataset Suggestions**: 
- Retail data from the "Advertising" dataset on Kaggle (https://www.kaggle.com/ashishpatel26/advertising-dataset).

**Tasks**:
- Data Preprocessing:
    - Clean and prepare the dataset, ensuring appropriate formats and handling missing values.
  
- Exploratory Data Analysis:
    - Visualize sales trends and advertising spend across different channels using Seaborn or Matplotlib.
  
- Causal Inference Setup:
    - Use CausalPy to define treatment groups based on advertising channels and control for confounding variables.
  
- Estimate Causal Effects:
    - Apply propensity score matching to estimate the impact of each advertising channel on sales.
  
- Interpret Results:
    - Summarize findings, highlighting which advertising channel yielded the highest sales increase.

---

### Project 2: Evaluating the Effect of Employee Training on Productivity
**Difficulty**: 2 (Medium)

**Project Objective**: 
Investigate how a structured employee training program affects productivity levels, optimizing for the most effective training methods.

**Dataset Suggestions**: 
- Employee productivity data from the "Employee Training" dataset on Kaggle (https://www.kaggle.com/datasets/benroshan/employee-productivity).

**Tasks**:
- Data Preprocessing:
    - Clean the dataset and encode categorical variables related to training methods and productivity metrics.
  
- Exploratory Data Analysis:
    - Analyze productivity trends before and after training, segmenting by different training methods.
  
- Causal Framework:
    - Define treatment and control groups based on training participation using CausalPy.
  
- Causal Effect Estimation:
    - Implement instrumental variable analysis to control for selection bias and estimate the causal impact of training on productivity.
  
- Reporting:
    - Generate visualizations to present the causal estimates and discuss implications for training programs.

---

### Project 3: Analyzing the Effects of Policy Changes on Public Health Outcomes
**Difficulty**: 3 (Hard)

**Project Objective**: 
Examine the causal impact of health policy changes (e.g., smoking bans, vaccination campaigns) on public health outcomes, optimizing for policy effectiveness.

**Dataset Suggestions**: 
- Health outcomes data from the "Health Policy" dataset on Kaggle (https://www.kaggle.com/datasets/aaronschlegel/health-policy-data).

**Tasks**:
- Data Preprocessing:
    - Clean and format the dataset, focusing on health indicators and policy implementation dates.
  
- Exploratory Data Analysis:
    - Visualize trends in health outcomes before and after policy changes, identifying potential confounders.
  
- Causal Inference Design:
    - Use CausalPy to construct a causal graph that represents the relationships between policies, confounders, and health outcomes.
  
- Causal Effect Estimation:
    - Apply advanced causal inference techniques (e.g., difference-in-differences) to estimate the impact of policy changes on health outcomes.
  
- Comprehensive Analysis:
    - Provide a detailed interpretation of results, including policy recommendations based on the causal analysis.

**Bonus Ideas (Optional)**:
- For Project 1: Compare the estimated effects of advertising channels with a baseline model without causal inference.
- For Project 2: Implement a sensitivity analysis to assess the robustness of the causal estimates.
- For Project 3: Explore the potential long-term effects of the policies on health outcomes using time-series analysis.

