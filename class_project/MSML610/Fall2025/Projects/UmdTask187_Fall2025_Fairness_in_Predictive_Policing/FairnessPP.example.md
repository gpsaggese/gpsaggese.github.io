# End-to-End Predictive Policing Workflow (UmdTask187)

## 1. Workflow Diagram

The project follows a standard machine learning workflow, with a critical intervention layer:


## 2. Core Technical Decisions

### A. Model Selection: Gradient Boosting Trees (GBT)
We chose GBT over standard Random Forests because GBT generally achieves higher predictive accuracy by sequentially correcting errors. This is crucial as achieving high performance is the *trade-off* we must analyze against fairness.

### B. The Fairness Constraint: Equalized Odds
We selected **Equalized Odds** because it is the most relevant metric for policing. It requires that the model's performance metrics—specifically the True Positive Rate (correctly predicting a hotspot) and False Positive Rate (falsely flagging an area as a hotspot)—are equal across all sensitive groups. This directly combats disparate allocation of policing resources.

### C. Addressing Intersectional Bias
Our data preparation step creates the sensitive attribute ($\mathbf{A}$) by combining Race and Income. The evaluation notebook will present metrics for groups like 'Black\_Low' and 'Hispanic\_Low'. The key finding for the final project will be analyzing whether the mitigation strategy successfully reduces the disparity for the most disadvantaged **intersectional group**.

## 3. Impact Analysis and Trade-Off
The final analysis will focus on the trade-off curve:
* **Performance vs. Fairness:** We will compare the overall AUC/Accuracy of the mitigated model against the unmitigated baseline.
* **Implications for Policing:** We will discuss the findings, concluding whether enforcing Equalized Odds on intersectional groups results in a negligible performance hit while significantly improving equity in predicted hotspot locations.