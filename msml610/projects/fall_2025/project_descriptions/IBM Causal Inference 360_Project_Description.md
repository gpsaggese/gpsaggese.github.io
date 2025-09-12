**Description**

IBM Causal Inference 360 is an open-source toolkit designed to help data scientists and researchers identify and estimate causal effects from observational data. It provides a suite of algorithms for causal inference and allows users to analyze the impact of interventions on outcomes. 

Technologies Used
IBM Causal Inference 360

- Offers a variety of causal inference methods including propensity score matching, inverse probability weighting, and regression adjustment.
- Facilitates the estimation of treatment effects and counterfactual outcomes.
- Provides tools for diagnostics and validation of causal assumptions.

---

### Project 1: Understanding the Impact of Marketing Campaigns on Sales
**Difficulty**: 1 (Easy)

**Project Objective**: 
Evaluate the causal effect of a marketing campaign on sales performance, identifying how much sales increased due to the campaign.

**Dataset Suggestions**: 
- Dataset: Marketing Campaign Data from Kaggle (e.g., “Marketing Campaigns”).
- Source: [Kaggle - Marketing Campaign Dataset](https://www.kaggle.com/datasets/)

**Tasks**:
- Data Preprocessing:
    - Clean and preprocess the dataset to handle missing values and categorical variables.
    
- Define Treatment and Control Groups:
    - Identify customers who received the marketing campaign (treatment group) and those who did not (control group).
    
- Causal Model Specification:
    - Use IBM Causal Inference 360 to specify a causal model to estimate the treatment effect on sales.
    
- Estimate Treatment Effects:
    - Implement methods such as propensity score matching to estimate the causal impact of the campaign.
    
- Interpret Results:
    - Analyze and interpret the estimated treatment effects and validate the causal assumptions.

**Bonus Ideas (Optional)**:
- Compare the impact of different marketing strategies (e.g., email vs. social media).
- Analyze long-term effects of the campaign on customer retention.

---

### Project 2: Analyzing the Effect of Education Interventions on Student Performance
**Difficulty**: 2 (Medium)

**Project Objective**: 
Investigate the causal relationship between educational interventions (e.g., tutoring programs) and student performance metrics.

**Dataset Suggestions**: 
- Dataset: Student Performance Data from Kaggle (e.g., “Student Performance Dataset”).
- Source: [Kaggle - Student Performance Dataset](https://www.kaggle.com/datasets/)

**Tasks**:
- Data Exploration:
    - Explore the dataset to understand the features and the distribution of student performance.
    
- Treatment Group Identification:
    - Identify students who participated in tutoring programs as the treatment group and those who did not as the control group.
    
- Causal Inference Analysis:
    - Utilize IBM Causal Inference 360 to apply methods such as regression adjustment to estimate the effect of tutoring on performance.
    
- Sensitivity Analysis:
    - Conduct sensitivity analysis to assess the robustness of your causal estimates against unobserved confounding.
    
- Reporting Findings:
    - Summarize the findings and provide recommendations based on the causal analysis.

**Bonus Ideas (Optional)**:
- Explore the effect of different types of interventions (e.g., group tutoring vs. one-on-one).
- Compare performance across different demographics.

---

### Project 3: Evaluating the Impact of Public Health Policies on COVID-19 Outcomes
**Difficulty**: 3 (Hard)

**Project Objective**: 
Assess the causal effects of various public health policies (e.g., lockdowns, mask mandates) on COVID-19 infection rates and mortality.

**Dataset Suggestions**: 
- Dataset: COVID-19 Public Health Policy Data from the Oxford COVID-19 Government Response Tracker.
- Source: [Oxford COVID-19 Government Response Tracker](https://www.bsg.ox.ac.uk/research/research-projects/covid-19-government-response-tracker)

**Tasks**:
- Data Integration:
    - Collect and integrate COVID-19 case data with public health policy data across multiple regions.
    
- Define Treatment and Control Regions:
    - Identify regions with different policy implementations (e.g., strict lockdown vs. no lockdown) as treatment and control groups.
    
- Causal Inference Methodology:
    - Apply IBM Causal Inference 360 to estimate treatment effects using techniques like inverse probability weighting.
    
- Model Diagnostics:
    - Perform model diagnostics to validate the assumptions of the causal models used.
    
- Policy Recommendations:
    - Analyze the results to provide evidence-based recommendations for public health policy.

**Bonus Ideas (Optional)**:
- Investigate the impact of vaccination rates on the effectiveness of policies.
- Analyze the effects of policies over time to observe changes in public health outcomes.

