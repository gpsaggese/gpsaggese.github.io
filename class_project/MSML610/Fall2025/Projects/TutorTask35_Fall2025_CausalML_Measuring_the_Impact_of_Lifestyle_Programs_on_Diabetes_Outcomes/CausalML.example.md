# Causal Analysis of Lifestyle Interventions on Diabetes Risk

## 1. Project Objective
This project estimates the **Heterogeneous Treatment Effect (HTE)** of physical activity on diabetes prevalence using Observational Data. Unlike traditional regression, which gives a single "average" coefficient, we use Causal Machine Learning (X-Learner) to understand **who** benefits most from lifestyle changes.

**Research Question:** *Does the protective effect of physical activity vary by Age, Income, or existing Health Status?*

## 2. Data Source
*   **Dataset:** CDC Diabetes Health Indicators (BRFSS 2015)
*   **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
*   **Scale:** ~253,680 records (Full Dataset)
*   **Type:** Cross-sectional survey data

### Key Variables
| Variable | Role | Description |
|----------|------|-------------|
| `PhysActivity` | Treatment (T) | 1 = Reported physical activity in past 30 days, 0 = No. |
| `Diabetes_binary` | Outcome (Y) | 1 = Has Diabetes, 0 = No Diabetes. |
| `Age` | Covariate (X) | 13-level age category (1 = 18-24, 13 = 80+). |
| `Income` | Covariate (X) | 8-level income category. |
| `GenHlth` | Covariate (X) | Self-reported health (1=Excellent, 5=Poor). |
| `BMI` | Covariate (X / M) | Body mass index (used as confounder in estimation). |

## 3. Identification Strategy & Assumptions
Since we cannot randomize people into "exercise" vs "sedentary" groups, we rely on observational causal inference methods.

### Causal Graphs (DAGs)

We present two DAGs: the **ideal structure** we would have with longitudinal intervention data, and the **actual structure** reflecting our cross-sectional reality.

---

## DAG 1: Assumed Causal Structure (If We Had Longitudinal Data)

This DAG represents the causal structure we **would** estimate with longitudinal intervention data, where physical activity is measured before diabetes onset and confounders are baseline characteristics.

```mermaid
%% DAG 1: Assumed Causal Structure (Longitudinal, Ideal Case)
graph LR
    X[Confounders<br/>Age, Income,<br/>GenHlth, BMI]
    T[Physical Activity<br/>Baseline]
    Y[Diabetes Risk<br/>Follow-up]

    X --> T
    X --> Y
    T --> Y

    style X fill:#f9f,stroke:#333,stroke-dasharray: 5 5,color:#000
    style T fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style Y fill:#bfb,stroke:#333,stroke-width:2px,color:#000
````

---

## DAG 2: Actual Cross-Sectional Reality (What We Have)

This DAG represents our actual cross-sectional data structure, showing:
(1) unmeasured confounding (U),
(2) simultaneous measurement of treatment and outcome, and
(3) potential reverse causality.

```mermaid
%% DAG 2: Actual Cross-Sectional Reality (What We Have)
graph LR
    U[Unmeasured U<br/>Motivation, Genetics,<br/>Access, Support]
    X[Measured X<br/>Age, Income,<br/>GenHlth, BMI]
    T[Physical Activity<br/>Measured Now]
    Y[Diabetes Risk<br/>Measured Now]

    U --> T
    U --> Y
    X --> T
    X --> Y

    T --> Y
    Y -. Reverse causality (bias) .-> T

    style U fill:#000,stroke:#333,color:#fff
    style X fill:#f9f,stroke:#333,stroke-dasharray: 5 5,color:#000
    style T fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style Y fill:#bfb,stroke:#333,stroke-width:2px,color:#000
```

> ⚠️ **Critical Limitation:** The solid arrow from Physical Activity to Diabetes Risk represents the intended causal effect, while the dashed arrow back from Diabetes Risk to Physical Activity represents reverse causality.Diabetes may initially increase exercise adherence (post-diagnosis 
behavioral response to medical advice) but eventually reduce it 
(due to complications like neuropathy, fatigue). Cross-sectional 
data captures both patterns simultaneously across different disease 
stages, making temporal ordering impossible to establish. Cross-sectional data cannot distinguish these mechanisms. Our implausible age-stratified results (positive effects in young adults) provide empirical evidence of this violation.


---

### Key Assumptions

1. **Unconfoundedness (Selection on Observables):** We assume that by controlling for 15+ variables (Income, Education, General Health, etc.), we isolate the effect of activity.
2. **Overlap (Positivity):** We verify that there are sedentary and active people in every demographic stratum using Propensity Score checks.
3. **SUTVA:** We assume no interference (one person's exercise doesn't prevent another's diabetes).

## 4. Methodology: The X-Learner

We use the **X-Learner** (implemented via our `CausalNavigator` wrapper).

* **Why X-Learner?** It is superior to standard S-Learners when the treatment groups are imbalanced (more people exercise than not).
* **Base Learner:** We use **XGBoost** for the nuisance models to capture non-linear relationships (e.g., the risk of diabetes accelerates non-linearly with Age).

### How X-Learner Calculates Heterogeneous Treatment Effects (CATE)

The **X-Learner** estimates the **Conditional Average Treatment Effect (CATE)** τ(x) = E[Y(1) - Y(0) | X = x] through a sophisticated 3-stage process:

#### Stage 1: Outcome Regression
Train separate models for treated and control groups:
- **μ₀(x)**: Predicts outcome for control group (sedentary people)  
- **μ₁(x)**: Predicts outcome for treated group (active people)

#### Stage 2: Imputed Treatment Effects  
Calculate "pseudo-outcomes" by imputing missing counterfactuals:

**For treated units (active people):**
- Observed: Y₁(x) (their actual diabetes outcome)  
- Imputed: μ₀(x) (predicted outcome if they were sedentary)
- **D₁(x) = Y₁(x) - μ₀(x)** (individual treatment effect estimate)

**For control units (sedentary people):**  
- Observed: Y₀(x) (their actual diabetes outcome)
- Imputed: μ₁(x) (predicted outcome if they were active)  
- **D₀(x) = μ₁(x) - Y₀(x)** (individual treatment effect estimate)

#### Stage 3: Treatment Effect Regression
Train models to predict treatment effects as a function of covariates:
- **τ₀(x)**: CATE model from control perspective using pseudo-outcomes D₀
- **τ₁(x)**: CATE model from treated perspective using pseudo-outcomes D₁
- **Final CATE**: Propensity-weighted combination of both perspectives

#### Key Advantages:
1. **Handles Imbalanced Data**: Works well when treatment/control groups have different sizes (73% active vs 27% sedentary)
2. **Captures Heterogeneity**: CATE varies by individual characteristics (age, health status, income)  
3. **Robust Estimation**: Combines evidence from both treatment and control groups

#### Interpretation:
- **CATE > 0**: Exercise increases diabetes risk for this person (counterintuitive - suggests confounding)
- **CATE < 0**: Exercise decreases diabetes risk for this person (expected biological effect)
- **CATE ≈ 0**: Exercise has no effect for this person (already at optimal risk level)

## 5. Key Findings & Interpretation

Our analysis reveals critical insights that a standard regression would miss:

### A. The "High Risk" Benefit

The protective effect of physical activity is **strongest for individuals in poor general health** (GenHlth=5).

* *Insight:* Healthy individuals (GenHlth=1) show near-zero CATE (they are already at low risk). Sick individuals show a large negative CATE (protective). This suggests interventions should target at-risk populations rather than the general public.

### B. The "Age" Anomaly: Understanding the Counterintuitive Results

We observed a positive treatment effect (higher diabetes risk for active people) in young adults (Age groups 1–3, corresponding to ages 18–34), which is biologically implausible.

* *Diagnostic Results:* Our analysis revealed that diabetics exercise **LESS** than non-diabetics across all age groups:
  - Age Group 1 (18-24): 80.8% of diabetics vs 86.5% of non-diabetics exercise  
  - Age Group 2 (25-29): 72.1% of diabetics vs 83.5% of non-diabetics exercise
  - All differences are negative (-5.7% to -12.0%), showing the expected biological pattern

* *Root Cause Analysis:* The positive CATE in young adults is **not** due to reverse causality but likely due to:
  1. **Unmeasured confounding**: Variables not in our dataset (genetics, socioeconomic factors, motivation) that affect both exercise propensity and diabetes risk
  2. **Selection bias**: Young adults who develop diabetes may have severe underlying conditions that make them fundamentally different from the general population
  3. **Model limitations**: The X-Learner assumes unconfoundedness, but our 17 covariates may be insufficient to capture all relevant confounders

* *Implication:* The counterintuitive results highlight the fundamental challenge of causal inference with observational data. Even sophisticated methods like X-Learner cannot overcome violations of key assumptions.

## 6. Limitations & Future Work

### Key Limitations

1. **Unmeasured Confounding:** Critical variables like genetics, detailed socioeconomic status, motivation, healthcare quality, and social support are not captured in the dataset. The counterintuitive age results suggest our 17 covariates are insufficient for unconfoundedness.
2. **Cross-Sectional Design:** Treatment and outcome are measured simultaneously, preventing establishment of temporal ordering and causal direction.
3. **Self-Reported Data:** Both physical activity and diabetes status are self-reported, introducing measurement error and potential social desirability bias.
4. **Population Heterogeneity:** Young adults with diabetes represent a small, potentially very different population with severe underlying conditions not captured by standard health indicators.



