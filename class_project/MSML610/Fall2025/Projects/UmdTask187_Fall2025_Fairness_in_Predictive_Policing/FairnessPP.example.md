# End-to-End Predictive Policing Workflow (UmdTask187)

# FairnessPP Example: Mitigating Bias in Crime Prediction

## 1. Problem Statement
Predictive policing algorithms are often criticized for reinforcing historical biases. A standard model trained on arrest data may learn to over-police specific demographic groups (e.g., Low-Income neighborhoods or Minority communities) simply because they are historically over-represented in the training data.

This example demonstrates how **FairnessPP** can be used to:
1.  **Diagnose** bias in a standard Gradient Boosting model.
2.  **Mitigate** that bias using Equalized Odds constraints.
3.  **Evaluate** the trade-off between public safety (Accuracy) and civil rights (Fairness).

## 2. Workflow Overview
The example notebook (`FairnessPP.example.ipynb`) follows a strict temporal validation approach, simulating a real-world deployment:
* **Training Data:** Chicago Crime Data (2020–2022).
* **Testing Data:** Future Crime Data (2023).

We compare two models:
* **Baseline Model:** Standard Gradient Boosting (Unaware of demographics).
* **Fair Model:** The same architecture wrapped in the `FairnessPredictor` with mitigation enabled.

## 3. Usage Example
Below is a simplified view of how the `FairnessPP` API simplifies this workflow:

```python
from FairnessPP_utils import FairnessPredictor, load_chicago_data

# 1. Load Data
X, y, A, dates = load_chicago_data()

# 2. Train Standard Model (Unmitigated)
baseline = FairnessPredictor()
baseline.train(X, y, mitigate=False)

# 3. Train Fair Model (Mitigated) - Just one flag change!
fair_model = FairnessPredictor()
fair_model.train(X, y, A=A, mitigate=True)

# 4. Compare
res_base = baseline.evaluate(X, y, A)
res_fair = fair_model.evaluate(X, y, A)

## 5 Results & Analysis
The following results were observed on the full 80,000-row dataset:

| Model | Accuracy | Balanced Accuracy | Fairness Disparity |
| :--- | :--- | :--- | :--- |
| **Baseline (Unmitigated)** | ~88.5% | ~62.0% | **High (0.39)** |
| **Fair (Mitigated)** | ~87.8% | ~50.0% | **Near-Zero (0.003)** |

### Key Findings
1.  **Majority Class Collapse:** The Fair Model successfully minimized disparate impact to near-zero. However, the `group_metrics` analysis reveals that it achieved this by predicting "No Arrest" for almost everyone (Selection Rate ~0.0).
2.  **The Trade-off:** We achieved perfect fairness, but at the cost of utility (Balanced Accuracy dropped significantly).
3.  **Conclusion:** The tool successfully enforced the constraints, highlighting the critical real-world trade-off between strict fairness definitions and predictive power.