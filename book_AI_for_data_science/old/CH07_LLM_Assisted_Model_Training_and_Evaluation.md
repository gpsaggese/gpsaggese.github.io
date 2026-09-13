# Chapter 7: LLM-Assisted Model Training and Evaluation

## 1. Experiment Design with AI

### 1.1 Experiment-Driven ML Development

- Systematic experimentation = the scientific method applied to ML:
  1. Formulate a hypothesis (e.g., "adding interaction features will reduce
     RMSE by 5%")
  2. Design the experiment: controlled change, fixed seed, identical evaluation
  3. Run the experiment and log all parameters and metrics
  4. Analyze results against hypothesis
  5. Iterate
- Without structure, experiments become ad hoc and unreproducible

### 1.2 AI-Assisted Experiment Design

- AI roles in experiment design:
  - Suggests next experiment based on current results (exploration strategy)
  - Generates hyperparameter grids based on prior literature
  - Identifies confounds in the experimental design
  - Helps formulate falsifiable hypotheses
- Prompting pattern: share the experiment log summary -> ask "what should the
  next experiment be and why?"

### 1.3 Experiment Tracking Systems

- MLflow: open-source experiment tracking
  - `mlflow.log_param()`, `mlflow.log_metric()`, `mlflow.log_artifact()`
  - Model registry: promote models from experiment to staging to production
- Weights & Biases: cloud-based experiment tracking with rich visualization
  - Sweeps: parallel hyperparameter search with automatic aggregation
  - Reports: shareable experiment summaries

- TUTORIAL: mlflow - Design a multi-run experiment with AI-suggested
  hyperparameter grids; log all parameters, metrics, and artifacts; use AI to
  analyze the results and recommend the next experiment

- TUTORIAL: wandb - Use Weights and Biases sweeps to run AI-suggested
  hyperparameter searches; visualize parallel coordinate plots to identify the
  best configuration region

- TUTORIAL: Ray - Distribute a hyperparameter search across multiple CPU/GPU
  nodes using Ray Tune; use AI to configure the search algorithm and resource
  allocation strategy

---

## 2. Training Monitoring and Diagnostics

### 2.1 Training Instability Patterns

- Common training problems that monitoring detects:
  - **Divergence**: loss increases monotonically; usually caused by too high
    learning rate
  - **Plateau**: loss stops decreasing; may indicate wrong architecture, bad
    feature scaling, or stuck in a local minimum
  - **Overfitting**: train loss decreases but validation loss increases; gap
    between train and validation metrics
  - **Gradient explosion**: gradients grow to infinity; use gradient clipping
  - **Gradient vanishing**: gradients shrink to zero in early layers; use
    batch normalization or skip connections

### 2.2 Monitoring Tools

- TensorBoard: real-time visualization of training curves, weight histograms,
  and computational graphs
- Weights & Biases: richer UI; supports distributed training logging; model
  artifact versioning
- Practical monitoring checklist:
  - Log train and validation loss every epoch
  - Log gradient norms to detect explosion/vanishing
  - Log learning rate schedule to verify it is decreasing correctly
  - Alert if validation loss has not improved for N epochs

### 2.3 AI Interpretation of Training Curves

- Paste training curve data (loss, accuracy per epoch) into LLM prompt:
  - "The validation loss plateaued at epoch 20. What should I try next?"
  - AI responses: reduce learning rate, add dropout, increase regularization,
    add more training data

- TUTORIAL: tensorboard - Monitor training runs in real time with TensorBoard;
  use AI to diagnose learning curve anomalies such as divergence, plateaus, and
  overfitting

- TUTORIAL: Weights & Biases - Track gradient norms, activation distributions,
  and learning rate schedules; use AI to detect training instabilities from the
  logged metrics

- TUTORIAL: Darts - Train and monitor a time series forecasting model with
  Darts; use AI to interpret the training and validation loss curves and
  recommend early stopping criteria

---

## 3. Evaluation and Error Analysis

### 3.1 Evaluation Metrics Selection

- Choose metrics that align with the business objective, not just technical
  convenience:
  - Binary classification: accuracy, precision, recall, F1, AUC-ROC, AUC-PR
  - Multiclass: macro vs. micro F1, confusion matrix analysis
  - Regression: MAE, RMSE, R², MAPE
  - Ranking: NDCG, MAP
- AI helps match metrics to business goals:
  - "In fraud detection, false negatives (missed fraud) are more costly than
    false positives. Which metric should I optimize?"
  - Answer: recall or a weighted F1 with higher weight on the positive class

### 3.2 Error Analysis

- Systematic error analysis process:
  1. Identify failure modes: which examples does the model get wrong?
  2. Slice by subgroups: is performance worse for a specific segment?
  3. Find patterns in errors: what do the wrong predictions have in common?
  4. Hypothesize root cause: feature engineering gap, data quality issue, or
     distribution mismatch
- AI assists with steps 3 and 4: paste the list of misclassified examples ->
  ask "what pattern do you see in these errors?"

### 3.3 Model Explanation for Evaluation

- SHAP values explain individual predictions:
  - Global SHAP: which features matter most across all predictions
  - Local SHAP: why did the model predict X for this specific row
- LIME: approximates the model locally with a linear model for each prediction
- Use explanation tools during evaluation to verify that the model is using
  expected features, not spurious correlations

- TUTORIAL: SHAP - Compute SHAP values for a trained classifier; use AI to
  interpret global and local explanations; identify systematic error patterns in
  model predictions

- TUTORIAL: LIME - Apply LIME to explain individual predictions from a complex
  model; use AI to translate LIME output into actionable recommendations for
  model improvement

- TUTORIAL: What-If Tool - Use the What-If Tool to slice model performance by
  demographic subgroups; use AI to write a structured evaluation report
  summarizing performance disparities

---

## 4. Model Cards and Documentation

### 4.1 What Model Cards Are

- Model card = structured documentation for a trained model that describes:
  - Intended use cases and known limitations
  - Evaluation results across demographic groups and data slices
  - Training data description and potential biases
  - Ethical considerations and fairness analysis
- Required by responsible AI frameworks (Google, Hugging Face, EU AI Act)

### 4.2 AI-Generated Model Documentation

- AI drafts model card sections from logged metadata:
  - "Here is the model's evaluation report. Write the Intended Use and
    Limitations sections of a model card."
  - "Here are the demographic group performance metrics. Summarize the fairness
    findings in plain language."
- Human review: validate factual claims; add context not visible in logged data
- Integrate model card generation into the CI/CD pipeline:
  - Auto-generate after every training run using MLflow metadata

### 4.3 Living Documentation

- Model cards must stay in sync with model versions
- W&B Weave: automates model evaluation tracking; builds a living artifact
  updated after each training run
- Versioned model documentation:
  - Link each model card version to the model version in the registry
  - Track how performance changes across model versions

- TUTORIAL: mlflow - Generate a model card from logged MLflow metadata; use AI
  to draft the intended use, limitations, and evaluation results sections

- TUTORIAL: W&B Weave - Use Weave to track model evaluation results and build
  a living model documentation artifact updated automatically after each
  training run

- TUTORIAL: gradio - Build a model demo interface with Gradio; use AI to
  generate the interface code and a model card HTML page; share the demo as
  part of the model documentation package
