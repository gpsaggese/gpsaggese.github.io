**Description**

CausalNex is a Python library designed for causal inference and causal modeling, enabling users to construct, visualize, and analyze causal graphs. It allows data scientists to identify relationships between variables, simulate interventions, and perform counterfactual reasoning. 

**Technologies Used**

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

## Project 2: Analyzing the Impact of Marketing Campaigns on Sales  
**Difficulty**: 2 (Medium)  

**Project Objective**: Determine the causal impact of different marketing campaigns on customer responses, optimizing strategies based on data-driven insights.  

**Dataset Suggestions**:  
- **Dataset**: "Bank Marketing Dataset" from the UCI Machine Learning Repository  
- **Link**: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  

**Tasks**:  
- **Data Preprocessing**: Clean and format the dataset, ensuring categorical variables (e.g., campaign type, contact method) are handled properly.  
- **Causal Graph Development**: Build a causal graph using CausalNex to illustrate relationships between campaign strategies, customer demographics, and responses.  
- **Counterfactual Analysis**: Conduct a counterfactual analysis to estimate outcomes if certain campaigns had not been run.  
- **Evaluation of Interventions**: Analyze the effectiveness of different strategies by simulating interventions (e.g., changing contact method).  

**Bonus Idea (Optional)**: Consider incorporating economic conditions (e.g., employment rates) into the causal graph to assess their influence on campaign success.  

---

### Project 3: Investigating the Effects of Air Quality on Pollutant Levels
**Difficulty**: 3 (Hard)  

**Project Objective**:  
Explore the causal relationships between different air quality indicators (e.g., CO, NOx, PM2.5) to understand how changes in one pollutant affect others, using causal graphs to simulate interventions.  

**Dataset Suggestions**:  
- **Dataset**: "Air Quality Data Set" from the UCI Machine Learning Repository  
- **Link**: [Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)  

**Tasks**:  
- **Data Preprocessing**: Clean and preprocess the dataset, handling missing values and aligning temporal measurements.  
- **Causal Graph Construction**: Use CausalNex to construct a causal graph linking pollutants and environmental variables (e.g., temperature, humidity).  
- **Intervention Simulation**: Simulate interventions (e.g., reduction in CO levels) and analyze downstream effects on other pollutants.  
- **Interpretation**: Identify key variables driving air quality changes and discuss potential implications for pollution control policies.  

**Bonus Idea (Optional)**: Extend the analysis by comparing causal structures across different seasons (e.g., summer vs. winter) to see how relationships change.  

