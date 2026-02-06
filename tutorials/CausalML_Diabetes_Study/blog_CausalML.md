# Beyond Averages: Using CausalML to Discover Who Actually Benefits from Exercise

*How we used causal machine learning to move past "one-size-fits-all" health recommendations*

---

You've probably heard it a thousand times: *exercise reduces your risk of diabetes*. It's solid advice backed by decades of research. But here's the thing, that statement hides something important. It tells you about the *average* person, but you're not average. Nobody is.

What if we could ask a different question: **Who benefits the most from physical activity?** A 25-year-old athlete? A 60-year-old with high blood pressure? Someone already in poor health?

This is exactly what we set out to explore using `CausalML`, a Python library for causal machine learning. Along the way, we built a tool called `CausalNavigator` to make these analyses more accessible and uncovered some genuinely surprising patterns in the data.

---

## The Problem with Traditional Approaches

Let's say you're a health policy researcher, and you want to understand whether physical activity reduces diabetes risk. The classic approach? Run a regression:

```python
# The traditional way
model = LogisticRegression()
model.fit(X, y)
print(f"Exercise coefficient: {model.coef_[0]}")
```

You get back a single number. Maybe it's -0.3, suggesting exercise is protective. Great! But this number represents the **Average Treatment Effect (ATE)**, it assumes the effect is roughly the same for everyone.

In reality, treatment effects are almost always **heterogeneous**. The benefit of exercise might be enormous for some people and negligible for others. A blanket recommendation like "everyone should exercise more" ignores this nuance. What if we could identify high-responders and prioritize interventions where they'll have the most impact?

This is where **Heterogeneous Treatment Effect (HTE)** estimation comes in, and where `CausalML` shines.

---

## Enter CausalML: A Quick Tour

[CausalML](https://github.com/uber/causalml) is an open-source Python library developed by Uber's data science team. It provides a suite of **meta-learners** algorithms designed to estimate individualized treatment effects from observational data.

The key insight behind meta-learners is clever: instead of building one model to predict outcomes, you build multiple models and combine them strategically to isolate the causal effect of a treatment.

Here's the lineup:

| Learner | How It Works | Best For |
|---------|--------------|----------|
| **S-Learner** | One model predicts outcome using treatment as a feature | Simple baselines |
| **T-Learner** | Separate models for treated/control groups | Balanced datasets |
| **X-Learner** | Two-stage approach with propensity weighting | Imbalanced treatments |
| **R-Learner** | Residualize outcome and treatment, then model the residuals | Doubly robust estimation |
| **DR-Learner** | Combines propensity scores with outcome modeling | Maximum robustness |

The native `CausalML` API is powerful but requires significant boilerplate. You need to preprocess your data, check assumptions, train models, and visualize results all with separate code. We wanted something more streamlined.

---

## Building CausalNavigator: A Friendlier Interface

To make causal inference workflows more accessible, we built `CausalNavigator`, a wrapper class that handles the entire pipeline from assumption checking to visualization.

Here's what the interface looks like:

```python
from causalml_utils import CausalNavigator

# Initialize with your preferred meta-learner
navigator = CausalNavigator(
    learner_type='X',        # Use the X-Learner
    control_name='Sedentary',
    treatment_name='Active'
)

# Check assumptions before modeling
navigator.check_overlap(X, T)

# Estimate individualized treatment effects
cate = navigator.fit_estimate(X, T, Y)

# Visualize heterogeneity across subgroups
df_with_effects = navigator.get_cate_df(df)
navigator.plot_heterogeneity(df_with_effects, col='Age', bins=5)
```

Three lines to go from raw data to individualized causal estimates. Let's break down what's happening under the hood.

---

## The Analysis: Physical Activity and Diabetes

For our case study, we used the **CDC BRFSS (Behavioral Risk Factor Surveillance System)** dataset, a survey of over 250,000 Americans containing health indicators, lifestyle factors, and demographic information.

**Our research question:** Does the protective effect of physical activity on diabetes vary by age, income, or existing health status?

### Setting Up the Data

The BRFSS dataset includes a binary indicator for physical activity (`PhysActivity`: 1 if the respondent exercised in the past 30 days) and diabetes status (`Diabetes_binary`). We also have rich covariates: age, income, BMI, general health, and more.

```python
from causalml_utils import load_cdc_data, preprocess_for_causal

# Load and preprocess
df = load_cdc_data('data/diabetes_indicators.csv')
X, T, Y = preprocess_for_causal(
    df,
    treatment_col='PhysActivity',
    outcome_col='Diabetes_binary',
    covariate_cols=['Age', 'Income', 'BMI', 'GenHlth', 'HighBP', 'HighChol']
)
```

One immediate challenge: our dataset is **imbalanced**. About 74% of respondents reported being physically active, while only 26% were sedentary. This imbalance is precisely why we chose the X-Learner. It's designed to handle situations where one group dominates.

### Step 1: Checking Assumptions

Before we estimate any causal effects, we need to verify that causal inference is even valid for this data. The critical assumption is **overlap** (also called positivity): for every combination of covariates, there must be both treated and untreated individuals.

```python
navigator.check_overlap(X, T)
```

This generates a propensity score distribution plot. Propensity scores represent the probability of receiving treatment given covariates. If the distributions for treated and control groups don't overlap, we can't reliably estimate causal effects, we'd be extrapolating into regions with no data.

Our data passed this check. The distributions overlapped substantially, meaning we have active and sedentary people across all demographic strata.

### Step 2: Estimating Individual Treatment Effects

Now for the main event. The X-Learner works in two stages:

1. **Stage 1:** Train separate outcome models for the treated and control groups
2. **Stage 2:** Use each model to impute the counterfactual outcomes, then model the treatment effect directly

```python
# Fit the model and get individual-level effects
cate = navigator.fit_estimate(X, T, Y)

# Merge effects back into our dataframe
df_results = navigator.get_cate_df(df)
print(f"Average Treatment Effect: {cate.mean():.4f}")
print(f"Effect Range: [{cate.min():.4f}, {cate.max():.4f}]")
```

The output? A **CATE (Conditional Average Treatment Effect)** for every single person in our dataset. Not one number—253,680 numbers, each representing how much that individual would benefit from physical activity.

---

## What We Found: The Heterogeneity Story

Here's where things get interesting. When we visualized treatment effects across subgroups, clear patterns emerged.

### Finding 1: Sickest Patients Benefit Most

```python
navigator.plot_heterogeneity(df_results, col='GenHlth', bins=5)
```

People who rated their general health as "Poor" (GenHlth=5) showed the **largest protective effect** from physical activity. Meanwhile, those already in excellent health showed effects close to zero.

This makes intuitive sense if you're already healthy, the marginal benefit of exercise is smaller. There's a floor effect. But for someone managing multiple health conditions, physical activity can be transformative.

**Policy implication:** Instead of generic "everyone should exercise" campaigns, targeted interventions for high-risk populations would yield greater returns.

### Finding 2: The Age Anomaly

Here's where we hit something counterintuitive. Young adults (18-34) showed a near-zero or even *slightly positive* treatment effect. Wait, does exercise somehow *increase* diabetes risk for young people?

Not so fast. This is a textbook example of **reverse causality** contaminating observational data.

Think about it: Type 2 diabetes is rare in young adults. Those who *do* have it often have severe risk factors and have been medically advised to exercise. So in our data, the "Active" group among young people is enriched with individuals who are exercising *because* they're managing a condition, not the other way around.

Meanwhile, plenty of sedentary young people are metabolically healthy simply because they're young. Youth provides a buffer that masks the consequences of inactivity.

The model correctly recovers the expected protective signal in older populations, where accumulated lifestyle choices outweigh these selection biases.

**Lesson:** Always interrogate surprising results. Causal inference from observational data requires careful reasoning about the data-generating process.

---

## Validating Our Results

Finding interesting patterns is one thing. Trusting them is another. We implemented three robustness checks to stress-test our conclusions.

### A. Placebo Test: Is This Just Noise?

The idea is simple: if we randomly shuffle treatment assignments (breaking any true causal link), the estimated effect should collapse to zero.

```python
navigator.run_placebo_test(X, T, Y, n_simulations=10)
```

We ran 10 simulations with shuffled treatments. The "placebo" effects clustered tightly around 0.001 (essentially noise), while our actual estimated effect was clearly separated at -0.002.

**Verdict:** Our effect is statistically distinguishable from random chance.

### B. Estimator Tournament: Does the Method Matter?

Different meta-learners make different assumptions. If they all agree, we have more confidence in our findings.

```python
navigator.compare_estimators(X, T, Y)
```

This runs a "horse race" between S, T, X, R, and DR-Learners, comparing them using **Uplift Curves** on held-out test data. (Since we can't observe ground-truth individual effects, we measure how well each model sorts people from "high responder" to "low responder.")

Results:
- **X, T, R, and DR-Learners** performed nearly identically, clustering together at the top
- **S-Learner** underperformed significantly, closer to random guessing

This confirms two things: (1) the heterogeneous signal is real and structural, and (2) our choice of X-Learner was justified. The S-Learner's failure illustrates "regularization bias", it smoothed over the weak treatment signal instead of detecting it.

### C. Sensitivity Analysis: Which Variables Drive the Effect?

What happens if we remove covariates one at a time?

```python
navigator.run_sensitivity_analysis(X, T, Y)
```

This iteratively drops each variable and re-estimates the ATE. If the effect is robust, it should remain stable.

Our findings:
- The effect stayed **protective (negative)** across all scenarios, good news for robustness
- Removing **BMI** caused the largest shift, increasing the protective effect by roughly **5.5x**

That BMI result is particularly revealing. It tells us that physical activity reduces diabetes risk primarily *through* lowering body weight. When we control for BMI, we're estimating only the **direct effect** of exercise (independent of weight). When we remove BMI, we see the full **total effect** which is much larger.

---

## Comparing to Gold-Standard Evidence

Our observational analysis found a protective effect of about **0.2% absolute risk reduction**. Compare that to the **Diabetes Prevention Program (DPP)**, a landmark randomized controlled trial that found a **58% relative risk reduction** from lifestyle interventions.

That's a huge gap! But it's expected. Cross-sectional data like ours is plagued by survivor bias, reverse causality, and unmeasured confounding. We're working with a snapshot in time, not a carefully controlled experiment.

Still, the fact that our model:
- Recovers the correct *direction* of the effect
- Identifies that high-risk patients benefit most (consistent with DPP subgroup analyses)
- Passes multiple robustness checks

...demonstrates the utility of `CausalML` when RCTs aren't feasible. Not every research question can wait for a multi-year clinical trial.

---

## Key Takeaways

**For practitioners:**
- `CausalML` provides production-ready implementations of meta-learners for heterogeneous treatment effect estimation
- The X-Learner is particularly well-suited for imbalanced treatment scenarios
- Always validate with multiple robustness checks, placebo tests, sensitivity analysis, and estimator comparisons

**For the curious:**
- "Average effects" hide important variation. The same intervention can be transformative for some and useless for others
- Observational causal inference is powerful but requires careful reasoning about assumptions and biases
- Counterintuitive results often reveal selection effects or reverse causality, dig into them

**For health policy:**
- Targeting interventions at high-risk populations (poor general health, older adults) may yield better returns than blanket recommendations
- Exercise works, but the mechanism matters, much of the benefit flows through weight reduction

---

## Try It Yourself

We've packaged everything into a Docker container so you can reproduce our analysis in minutes:

```bash
# Build the container
docker build -t causalml_project .

# Run Jupyter
docker run -p 8888:8888 -v "$(pwd)":/app causalml_project
```

Open [`CausalML.API.ipynb`](https://github.com/gpsaggese/umd_classes/blob/master/tutorials/CausalML_Diabetes_Study/CausalML.API.ipynb) to explore the `CausalNavigator` interface, or dive into [`CausalML.example.ipynb`](https://github.com/gpsaggese/umd_classes/blob/master/tutorials/CausalML_Diabetes_Study/CausalML.example.ipynb) for the full diabetes analysis.

The code is designed to be self-contained, documented, and most importantly guaranteed to work. No more fighting with package versions or missing dependencies.

---

## What's Next?

Causal machine learning is a rapidly evolving field. Some directions we're excited about:

- **Causal forests** (via `econml` or `grf`) for non-parametric effect estimation
- **Double machine learning** for more flexible nuisance function estimation
- **Causal discovery** algorithms that infer causal graphs from data

The tools are maturing fast. What was research-only five years ago is now accessible to any data scientist willing to learn. And as healthcare, policy, and business increasingly demand *causal* answers, not just correlational patterns, these skills will only become more valuable.

If you're looking to apply causal inference in production environments, check out [Causify](https://causify.ai/) they're building tools to make causal AI accessible for real-world decision making.

