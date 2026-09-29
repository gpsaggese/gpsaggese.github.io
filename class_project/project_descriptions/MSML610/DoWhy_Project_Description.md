**Description**

DoWhy is a Python library designed for causal inference that allows users to model
causal relationships and perform causal reasoning. It provides a unified framework
for defining causal graphs, estimating causal effects, and testing for causal
assumptions. By leveraging DoWhy, users can analyze observational data to understand
the impact of interventions and make informed decisions.

Technologies Used
DoWhy

- Facilitates causal graph creation and manipulation
- Supports various causal inference methods including propensity score matching and
  instrumental variables
- Provides tools for testing causal assumptions and robustness checks

### Fall2026

#### Tutorial

- Usual tutorial "Learn DoWhy in 60 mins", following
  `.claude/skills/tutorials_in_60_mins.rules.md`
- Improve the existing `tutorials/dowhy/`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `dowhy_utils.py`
  - `dowhy.API.ipynb`
  - `dowhy.example.ipynb`

#### Project: Causal Analysis in Finance

- TODO(ai_gp): Create 3 examples of application of causal analysis in finance

#### Milestones

- TODO(ai_gp): Add milestones related to the project

### Project 1: Analyzing the Impact of Education on Income

- **Difficulty**: 1 (Easy)
- **Project Objective**: To estimate the causal effect of educational attainment on
  individual income levels using observational data
- **Dataset Suggestions**: Use the "Adult Income Dataset" available on Kaggle
  ([Adult Census Income](https://www.kaggle.com/uciml/adult-census-income))
- **Tasks**:
  - **Define the Causal Graph**: Create a causal graph representing the relationship
    between education, income, and other confounding variables
  - **Estimate Causal Effect**: Utilize DoWhy to estimate the causal effect of
    education on income using methods like propensity score matching
  - **Test Assumptions**: Conduct robustness checks to validate the assumptions made
    in the causal model
  - **Interpret Results**: Analyze the results and discuss the implications of the
    findings
- **Bonus Ideas (Optional)**:
  - Compare the causal effect across different demographics (age, gender)
  - Investigate the role of other factors such as work experience or location

### Project 2: Evaluating the Effect of Marketing Campaigns on Sales

- **Difficulty**: 2 (Medium)
- **Project Objective**: To assess the causal impact of a marketing campaign on sales
  revenue for a retail store
- **Dataset Suggestions**: Use the "Retail Store Sales Data" available on Kaggle
  ([Retail Store Sales Data](https://www.kaggle.com/datasets/irfanasrullah/retail-store-sales-data))
- **Tasks**:
  - **Construct the Causal Graph**: Develop a causal graph showing the relationship
    between marketing campaigns, sales, and potential confounders (e.g.,
    seasonality, economic factors)
  - **Estimate Treatment Effect**: Apply DoWhy to estimate the treatment effect of
    the marketing campaign on sales revenue using regression discontinuity or
    matching techniques
  - **Conduct Sensitivity Analysis**: Perform sensitivity analysis to assess how
    robust the causal estimates are to violations of assumptions
  - **Visualize Findings**: Create visualizations to present the estimated causal
    effects and their confidence intervals
- **Bonus Ideas (Optional)**:
  - Explore the effects of different types of marketing campaigns (digital vs. print)
  - Investigate the long-term impact of marketing on customer retention

### Project 3: Understanding the Effect of Air Quality on Health Outcomes

- **Difficulty**: 3 (Hard)
- **Project Objective**: To investigate the causal relationship between air quality
  (measured by PM2.5 levels) and respiratory health outcomes in urban populations
- **Dataset Suggestions**: Use the "Air Quality and Health Data" available from the
  U.S. Environmental Protection Agency (EPA)
  ([EPA Outdoor Air Quality Data](https://www.epa.gov/outdoor-air-quality-data))
- **Tasks**:
  - **Develop Causal Framework**: Create a complex causal graph that includes air
    quality, health outcomes, socio-economic factors, and other confounders
  - **Estimate Causal Effects**: Use DoWhy to estimate the causal effect of air
    quality on health outcomes employing advanced techniques like instrumental
    variable analysis
  - **Account for Confounding Variables**: Implement methods to control for
    confounding variables and assess their impact on the causal estimates
  - **Evaluate Model Robustness**: Perform robustness checks and sensitivity
    analyses to ensure the reliability of the causal inferences
- **Bonus Ideas (Optional)**:
  - Analyze the effects of policy changes aimed at improving air quality on health
    outcomes
  - Explore non-linear relationships between air quality and health effects using
    machine learning techniques
