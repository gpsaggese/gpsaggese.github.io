
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

## 3. Identification Strategy & Assumptions
Since we cannot randomize people into "exercise" vs "sedentary" groups, we rely on observational causal inference methods.

### Causal Graph (DAG)
We assume the following causal structure. We control for **Confounders** ($X$) to close the "backdoor paths" that create spurious correlations between Activity ($T$) and Diabetes ($Y$).

```mermaid
graph LR
    %% Nodes
    T(Treatment: Physical Activity)
    Y(Outcome: Diabetes Risk)
    X(Confounders: Age, Income, GenHlth)
    M(Mediator: BMI)

    %% Edges - Confounding (The Backdoor Paths)
    X --> T
    X --> Y
    X --> M

    %% Edges - Causal Paths
    T --> Y
    T --> M
    M --> Y

    %% Style
    style T fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style Y fill:#bfb,stroke:#333,stroke-width:2px,color:#000
    style X fill:#f9f,stroke:#333,stroke-dasharray: 5 5,color:#000
    style M fill:#000,stroke:#333,stroke-width:2px,color:#fff
````

### Key Assumptions

1. **Unconfoundedness (Selection on Observables):** We assume that by controlling for 15+ variables (Income, Education, General Health, etc.), we isolate the effect of activity.
2. **Overlap (Positivity):** We verify that there are sedentary and active people in every demographic stratum using Propensity Score checks.
3. **SUTVA:** We assume no interference (one person's exercise doesn't prevent another's diabetes).

## 4. Methodology: The X-Learner

We use the **X-Learner** (implemented via our `CausalNavigator` wrapper).

* **Why X-Learner?** It is superior to standard S-Learners when the treatment groups are imbalanced (more people exercise than not).
* **Base Learner:** We use **XGBoost** for the nuisance models to capture non-linear relationships (e.g., the risk of diabetes accelerates non-linearly with Age).

## 5. Key Findings & Interpretation

Our analysis reveals critical insights that a standard regression would miss:

### A. The "High Risk" Benefit

The protective effect of physical activity is **strongest for individuals in poor general health** (GenHlth=5).

* *Insight:* Healthy individuals (GenHlth=1) show near-zero CATE (they are already at low risk). Sick individuals show a large negative CATE (protective). This suggests interventions should target at-risk populations rather than the general public.

### B. The "Age" Anomaly (Reverse Causality)

We observed a positive treatment effect (higher diabetes risk for active people) in young adults (Age < 30).

* *Diagnosis:* This is likely **Reverse Causality**. Young adults typically do not develop diabetes unless they have specific risk factors. Once diagnosed, they are prescribed exercise. Thus, in this cross-sectional slice, "Activity" is a proxy for "Diagnosis Management" in the young.
* *Validation:* The effect becomes correctly protective (deeply negative) in older populations (Age 60+), where the cumulative biological benefit of exercise outweighs this selection bias.

```
::contentReference[oaicite:0]{index=0}
```