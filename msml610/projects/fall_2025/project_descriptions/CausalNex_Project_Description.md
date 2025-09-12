**Description**

CausalNex is a Python library designed for causal inference and causal modeling, enabling users to construct, visualize, and analyze causal graphs. It allows data scientists to identify relationships between variables, simulate interventions, and perform counterfactual reasoning. 

Technologies Used
CausalNex

- Provides tools for building and visualizing Bayesian networks.
- Supports causal inference, enabling users to derive insights about the impact of interventions.
- Allows simulation of counterfactual scenarios to understand potential outcomes.

---

### Project 1: Understanding Factors Influencing Student Performance
**Difficulty**: 1 (Easy)  
**Project Objective**: The goal is to analyze how various factors (e.g., study habits, attendance, and socio-economic status) influence student performance in exams. The project will optimize the understanding of key determinants that affect academic success.

**Dataset Suggestions**: Use the "Student Performance Dataset" available on Kaggle: [Student Performance Data](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption).

**Tasks**:
- **Data Ingestion**: Load the dataset and perform initial exploration to understand the features.
- **Causal Graph Construction**: Use CausalNex to create a causal graph representing the relationships between student performance and influencing factors.
- **Intervention Simulation**: Simulate interventions (e.g., increased study hours) and analyze their potential impact on student performance.
- **Analysis and Visualization**: Visualize the causal graph and present findings on the most significant factors affecting performance.

---

### Project 2: Analyzing the Impact of Marketing Campaigns on Sales
**Difficulty**: 2 (Medium)  
**Project Objective**: This project aims to determine the causal impact of different marketing campaigns on product sales, optimizing marketing strategies based on data-driven insights.

**Dataset Suggestions**: Use the "Marketing Campaign Dataset" from Kaggle: [Marketing Campaigns](https://www.kaggle.com/datasets/rohanrao94/marketing-campaign).

**Tasks**:
- **Data Preprocessing**: Clean the dataset and prepare it for analysis, ensuring all variables are appropriately formatted.
- **Causal Graph Development**: Build a causal graph using CausalNex to illustrate the relationships between marketing campaigns, customer engagement, and sales.
- **Counterfactual Analysis**: Conduct a counterfactual analysis to estimate what sales would have been without specific marketing efforts.
- **Evaluation of Interventions**: Analyze the effectiveness of different campaigns by simulating various scenarios and interpreting the results.

---

### Project 3: Investigating the Effects of Air Quality on Public Health
**Difficulty**: 3 (Hard)  
**Project Objective**: The objective is to explore the causal relationships between air quality indicators (e.g., PM2.5 levels) and public health outcomes (e.g., hospital admissions for respiratory issues) to identify effective interventions for improving health.

**Dataset Suggestions**: Use the "Air Quality and Health Data" available from the UCI Machine Learning Repository: [Air Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Air+Quality).

**Tasks**:
- **Data Integration**: Combine air quality data with public health records, ensuring temporal alignment of datasets.
- **Complex Causal Graph Construction**: Develop a comprehensive causal graph using CausalNex that includes various environmental and health-related variables.
- **Advanced Causal Inference**: Utilize CausalNex to perform advanced causal inference, determining the direct and indirect effects of air quality on health outcomes.
- **Policy Simulation**: Simulate potential policy interventions (e.g., reducing emissions) and analyze their expected impact on public health metrics.

**Bonus Ideas (Optional)**: 
- For Project 1, extend the analysis to include the effect of online learning environments on student performance.
- For Project 2, consider incorporating seasonality effects into the causal graph and analyze their influence on sales.
- For Project 3, explore the interaction between air quality and socio-economic factors in the context of health outcomes.

