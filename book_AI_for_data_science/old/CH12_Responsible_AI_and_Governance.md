# Chapter 12: Responsible AI and Governance

## 1. Bias Detection and Fairness

### 1.1 Types of Bias in ML

- **Historical bias**: bias embedded in the training data due to past societal
  inequities
  - Example: hiring model trained on data where women were historically
    underrepresented in senior roles
- **Representation bias**: training data overrepresents certain groups
- **Measurement bias**: different measurement quality or proxy variables for
  different groups
- **Aggregation bias**: using a single model for groups with fundamentally
  different statistical relationships

### 1.2 Fairness Metrics

- Group fairness metrics (parity-based):
  - **Demographic parity**: equal positive prediction rates across groups
  - **Equal opportunity**: equal true positive rates (recall) across groups
  - **Equalized odds**: equal true positive and false positive rates
  - **Predictive parity**: equal precision across groups
- Individual fairness: similar individuals should receive similar predictions
- Tension: many group fairness criteria are mathematically incompatible;
  choosing one may violate another

### 1.3 AI-Assisted Bias Auditing

- AI interprets fairness reports and translates them for non-technical
  stakeholders:
  - "The model has 15% lower recall for applicants over 50. In plain language,
    the model misses 15% more qualified older applicants than younger ones."
- AI suggests bias mitigation approaches based on the source of bias:
  - Pre-processing: resampling or reweighting training data
  - In-processing: fairness constraints during training
  - Post-processing: adjust decision thresholds per group

- TUTORIAL: AI Fairness 360 (AIF360) - Audit a trained classifier for
  demographic bias using AIF360; apply a bias mitigation algorithm; use AI to
  explain the trade-off between fairness and accuracy

- TUTORIAL: Fairlearn - Compute fairness metrics across demographic groups with
  Fairlearn; apply the ExponentiatedGradient mitigation; use AI to interpret the
  fairness-accuracy Pareto frontier

- TUTORIAL: What-If Tool - Use the What-If Tool to interactively explore
  model performance disparities across demographic groups; use AI to write a
  structured fairness assessment report from the exploration

---

## 2. Explainability with AI Assistance

### 2.1 The Explainability Requirement

- Why explainability matters:
  - **Regulatory compliance**: EU AI Act, GDPR right to explanation, US CFPB
    fair lending regulations require explanations for automated decisions
  - **Trust and debugging**: understand why the model makes a decision before
    trusting it in production
  - **Bias detection**: identify if the model uses protected attributes or their
    proxies
  - **Stakeholder communication**: translate model behavior to business
    language

### 2.2 Local vs. Global Explanations

- Global explanations: describe overall model behavior
  - Feature importance rankings, partial dependence plots
  - Who the model "generally" relies on
- Local explanations: explain a single prediction
  - SHAP values for one observation
  - LIME: locally linear approximation around one point
  - Counterfactuals: "What would need to change for this prediction to flip?"

### 2.3 AI Translation of Explanations

- Technical explanation outputs (SHAP values, LIME weights) are not
  self-explanatory to non-technical stakeholders
- AI translates:
  - Input: SHAP summary plot data (feature, importance value, direction)
  - Output: "The three most important factors in predicting churn are:
    (1) the customer has not logged in for more than 30 days,
    (2) their last support ticket was rated 1 star,
    (3) they are on the monthly plan rather than annual"

- TUTORIAL: SHAP - Generate SHAP explanation plots for a production model; use
  AI to translate the SHAP output into plain-language explanations suitable for
  non-technical stakeholders

- TUTORIAL: LIME - Apply LIME to a text classification model; use AI to identify
  which input features are driving the model's decisions and whether they
  represent legitimate signals or spurious correlations

- TUTORIAL: What-If Tool - Use the What-If Tool to generate counterfactual
  explanations for loan decisions; use AI to summarize the minimum changes
  required for an applicant to receive a positive decision

---

## 3. Governance Frameworks

### 3.1 AI Governance Components

- AI governance = policies, processes, and tools that ensure AI systems are
  safe, fair, and compliant
  - Model risk management: evaluate and mitigate risks before deployment
  - Access controls: who can deploy models, access training data, or change
    decision thresholds?
  - Audit trails: immutable records of model decisions and the data used to
    make them
  - Incident response: defined process for handling model failures

### 3.2 Data Quality as Governance

- Data quality gates enforced before each training run:
  - Schema validation: does the incoming data match the expected schema?
  - Distribution validation: are feature distributions within expected bounds?
  - Completeness validation: are required columns fully populated?
- Tools: Great Expectations, pandera
- AI generates the validation suite from a natural language description of the
  data contract

### 3.3 Regulatory Context

- Key regulations affecting ML:
  - **GDPR** (EU): right to explanation, data minimization, consent requirements
  - **EU AI Act**: risk-based regulation; high-risk AI requires human oversight
    and documentation
  - **FCRA / ECOA** (US): fair lending requirements; explainability for credit
    decisions
- AI can generate a compliance checklist from a regulatory document:
  - "Read this section of the EU AI Act and generate a checklist for a
    high-risk ML system"

- TUTORIAL: Great Expectations - Build a data governance checkpoint that
  validates data against defined expectations before each model training run;
  use AI to generate expectation suites from data quality requirements

- TUTORIAL: pandera - Define a schema contract for a model's training data; use
  AI to generate pandera validation rules from a regulatory compliance document
  describing data requirements

- TUTORIAL: Snorkel - Build a programmatic data labeling and governance pipeline
  with Snorkel; use AI to write labeling functions that encode regulatory
  compliance rules as training data constraints

---

## 4. Auditing AI-Assisted Decisions

### 4.1 The Audit Trail Problem

- When AI makes or assists a consequential decision, the system must be able to
  answer:
  - What data was used to train the model?
  - What version of the model was used for this decision?
  - What features drove this specific prediction?
  - Who approved the model for production use?
  - Has the model's performance been monitored since deployment?
- These questions require a complete, immutable audit trail

### 4.2 Building Audit Infrastructure

- Log every prediction with:
  - Input features (or a hash if PII must not be stored)
  - Model version and experiment ID
  - SHAP values for the top-K features
  - Timestamp and the identity of the requesting system
- MLflow Model Registry: links model version to experiment, data snapshot, and
  approval record
- Differential privacy: protects individual training records from being
  recovered by auditing queries

### 4.3 Privacy-Preserving AI

- Differential privacy (DP): add calibrated noise to model training so that
  individual records cannot be inferred from the model
  - Trade-off: stronger privacy = more noise = lower accuracy
  - DP budget (epsilon): controls the privacy-accuracy trade-off
- AI assists in selecting the DP budget:
  - "This dataset contains medical records. Given a privacy budget of epsilon=1,
    how much accuracy loss should we expect for a logistic regression model?"

- TUTORIAL: mlflow - Build an audit trail for an AI-assisted decision pipeline
  by logging all model inputs, outputs, and explanations to MLflow; use AI to
  generate a compliance summary report from the logged data

- TUTORIAL: pysyft - Apply differential privacy to a model training workflow
  using PySyft; use AI to select the privacy budget based on the sensitivity of
  the training data and the regulatory requirements

- TUTORIAL: opacus - Train a PyTorch model with differential privacy using
  Opacus; use AI to analyze the privacy-accuracy trade-off and recommend an
  epsilon value appropriate for the use case
