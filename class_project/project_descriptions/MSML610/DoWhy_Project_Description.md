# Description

DoWhy is a Python library for causal inference that models causal relationships and
estimates the effect of an intervention from observational data. It solves the
problem of turning a causal question into a defined graph, an identified estimand, an
estimate, and a set of robustness checks. It is worth a 60-minute tutorial because
its four-step API (model, identify, estimate, refute) makes the causal assumptions
explicit and testable.

## Technologies Used

DoWhy

- Facilitates causal graph creation and manipulation
- Supports various causal inference methods including propensity score matching and
  instrumental variables
- Provides tools for testing causal assumptions and robustness checks

# Tutorial

- Implement the tutorial "Learn DoWhy in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Read `tutorials/dowhy/README.md`
  - Read the `README.md` of the Spring2025 DoWhy project of DATA605, and reuse what
    is good
    - `class_project/data605/Spring2025/projects/TutorTask119_Spring2025_Real-Time_Bitcoin_Causal_Analysis_with_DoWhy/`
  - Read the `README.md` of `tutorials/CausalML_Diabetes_Study/` and
    `tutorials/causalnex/` for the related causal tools
- Start from the existing `tutorials/dowhy/` and make it better
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `dowhy_utils.py`
  - `dowhy.API.ipynb`
  - `dowhy.example.ipynb`

# Project

## Project 1: Effect of Credit Limit on Loan Default

- **Difficulty**: 2 (Medium)
- **Project Objective**: Estimate the causal effect of a high credit limit on the
  probability of default in the next month, and measure how robust the estimate is to
  violated assumptions
- **Dataset Suggestions**:
  [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Tasks**:
  - **Preprocess the Data**: Encode `SEX`, `EDUCATION`, and `MARRIAGE`, and define a
    binary treatment "credit limit above the median" and the default outcome
  - **Define the Causal Graph**: Write a graph where age, sex, education, marital
    status, and past repayment status are common causes of the limit and of default,
    and pass it to `CausalModel`
  - **Estimate the Effect**: Call `identify_effect` and `estimate_effect` with
    `backdoor.propensity_score_stratification` and `backdoor.linear_regression`, and
    compare the estimated ATE of the two methods
  - **Refute the Estimate**: Call `refute_estimate` with `placebo_treatment_refuter`,
    `random_common_cause`, and `data_subset_refuter`, and report the new estimates
  - **Interpret the Results**: Plot the ATE with confidence intervals per estimator
    and discuss whether an unobserved bank risk score can break the estimate
- **Bonus Ideas (Optional)**: Estimate the effect by age group with an EconML
  estimator called through DoWhy; add a sensitivity analysis to unobserved
  confounding

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Data
  - Result: `tutorials/dowhy/` container running, and a clean table with the
    treatment, the outcome, and the confounders
- Milestone 2: API notebook
  - Project tasks: Define the Causal Graph, Estimate the Effect, Refute the Estimate
  - Result: `dowhy.API.ipynb` covering `CausalModel`, `identify_effect`,
    `estimate_effect`, and `refute_estimate` on a synthetic dataset with a known
    effect
- Milestone 3: Example notebook
  - Project tasks: Define the Causal Graph, Estimate the Effect, Refute the Estimate,
    Interpret the Results
  - Result: `dowhy.example.ipynb` running end to end

## Project 2: Effect of Fed Rate Hikes on Equity Returns

- **Difficulty**: 2 (Medium)
- **Project Objective**: Estimate the causal effect of a Federal Reserve rate hike on
  the next-month return of the S&P 500, controlling for inflation, unemployment, and
  market volatility
- **Dataset Suggestions**:
  [FRED - Federal Funds Effective Rate](https://fred.stlouisfed.org/series/FEDFUNDS)
  - [FRED - Consumer Price Index](https://fred.stlouisfed.org/series/CPIAUCSL) and
    [FRED - Unemployment Rate](https://fred.stlouisfed.org/series/UNRATE)
  - S&P 500 (`^GSPC`) and VIX (`^VIX`) prices from
    [yfinance](https://pypi.org/project/yfinance/)
- **Tasks**:
  - **Build the Monthly Panel**: Merge the funds rate, inflation, unemployment,
    S&P 500 returns, and VIX into one monthly table
  - **Define the Treatment**: Define a rate-hike indicator as a positive monthly
    change of the funds rate, and the outcome as the next-month S&P 500 return
  - **Estimate the Effect**: Build the graph with lagged confounders and estimate the
    ATE with `backdoor.linear_regression` and `backdoor.propensity_score_weighting`
  - **Refute the Estimate**: Run `placebo_treatment_refuter` and
    `data_subset_refuter`, and check `add_unobserved_common_cause`
  - **Visualize the Effect**: Plot the hike indicator over the return series, and the
    ATE with confidence intervals for each estimator
- **Bonus Ideas (Optional)**: Repeat the analysis for rate cuts or for sector ETFs

## Project 3: Effect of Oil Price Shocks on Airline Stocks

- **Difficulty**: 3 (Hard)
- **Project Objective**: Estimate the causal effect of a weekly oil price surge on
  the weekly return of airline stocks, and test how robust the estimate is to
  unobserved confounding
- **Dataset Suggestions**:
  [FRED - WTI Crude Oil Price](https://fred.stlouisfed.org/series/DCOILWTICO)
  - Prices of the airline ETF `JETS`, the S&P 500, and the VIX from
    [yfinance](https://pypi.org/project/yfinance/)
- **Tasks**:
  - **Merge the Data Sources**: Align weekly oil prices, airline ETF, S&P 500, and
    VIX on week-end dates
  - **Define the Causal Graph**: Encode market return and volatility as common causes
    of the oil shock and of the airline return
  - **Estimate the Effects**: Estimate the ATE with `backdoor.linear_regression` and
    `backdoor.propensity_score_matching`, with bootstrap confidence intervals
  - **Test the Robustness**: Run `add_unobserved_common_cause`,
    `placebo_treatment_refuter`, and `data_subset_refuter`, and plot how the estimate
    changes with the strength of the hidden confounder
  - **Compare Regimes**: Estimate the effect separately in calm and high-volatility
    weeks and plot the two estimates with their intervals
- **Bonus Ideas (Optional)**: Compare the effect on airline stocks with the effect on
  energy stocks; add a second treatment for large oil price drops
