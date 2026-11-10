---
title: "AutoGluon Tabular in 60 Minutes"
draft: true
authors:
    - Sohail-5678
    - gpsaggese
date: 2026-11-09
description:
categories:
    - Machine Learning
    - Data Science
---

TL;DR: AutoGluon trains and ensembles a dozen models on a table in one call, and
this tutorial shows the decisions it still leaves to you: the baseline, the
metric, the decision threshold, and the explanation.

<!-- more -->

## What AutoGluon Does

Training a good model on a table takes many steps. You encode the categorical
and date columns, try several model families, validate without touching the test
set, and combine the best models. AutoGluon runs all of these steps from one
`fit()` call, within a time budget you set.

The tutorial lives in
[`tutorials/AutoGluon`](https://github.com/gpsaggese/gpsaggese.github.io/tree/master/tutorials/AutoGluon).
It pins `autogluon.tabular==1.6.3` with the CatBoost, LightGBM, and XGBoost
models. It leaves out the neural networks, so the Docker image needs no
`torch` and each notebook runs in about a minute on a laptop CPU.

## The API in Five Calls

The
[API notebook](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/AutoGluon/AutoGluon.API.ipynb)
uses a synthetic churn table whose label depends on four known columns. Because
the true answer is known, you can check every output against it.

```python
import autogluon.tabular as agt

predictor = agt.TabularPredictor(
    label="churned", eval_metric="roc_auc", path="tmp.autogluon/churn"
)
predictor.fit(train_df, presets="medium_quality", time_limit=60)

predictor.leaderboard(test_df)       # One row per trained model.
predictor.predict_proba(test_df)     # One column per class.
predictor.evaluate(test_df)          # Dictionary of test metrics.
predictor.feature_importance(test_df)
```

From the label alone, AutoGluon infers a binary problem. It turns the date
column into year, month, day, and weekday features, and it drops a constant
column. It then trains nine models and a weighted ensemble in about five
seconds. The permutation importance gets the generator right: the four real
drivers come first and the two noise columns score close to zero.

The same five calls cover regression and multiclass problems. Only the label
changes. The notebook also covers bagging and stacking (`num_bag_folds`,
`num_stack_levels`) and what each preset sets. It ends with
`clone_for_deployment()`, which keeps only the models the best model needs: in
the example, 2 models out of 8, and 0.7 MB instead of 46 MB.

## A Real Problem: Employee Attrition

The
[example notebook](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/AutoGluon/AutoGluon.example.ipynb)
uses the IBM HR Analytics dataset, with 1,470 fictional employees and 16% who
left. It reads the file from a pinned commit of IBM's repository and checks it
against a SHA-256 digest, so it needs no Kaggle account.

### Keep a Baseline

A logistic regression with one-hot encoding, written by hand in a few lines,
reaches a test ROC-AUC of 0.81. The AutoGluon ensemble (`good_quality`, 8-fold
bagging) reaches 0.80. A bootstrap of the 294 test rows gives 95% intervals of
0.73 to 0.88 and 0.72 to 0.87: the test set cannot tell the two models apart.

This result is the most useful lesson of the notebook. On a small table with
mostly monotonic effects, a linear model is hard to beat. AutoGluon matched it
with no feature engineering and searched enough model families to show there
was little left to gain. Without the baseline, a 0.80 would look like a win
for AutoML.

### Move the Threshold

At the default threshold of 0.5, the model flags only 8 of the 47 test
employees who left. The spec of the project asks for high recall, so the
threshold has to move. Choosing it on the test set would leak the test set into
the model. The notebook uses the out-of-fold predictions of the bagged models
instead: each training row gets a probability from models that never saw it.

```python
f1_threshold = predictor.calibrate_decision_threshold(metric="f1")
oof_proba = predictor.predict_proba_oof()["Yes"]
```

`calibrate_decision_threshold()` finds 0.27 for the best F1. A cost model gives
a second answer. If losing an employee costs five times a retention
conversation, the lowest average cost on the out-of-fold data is at 0.16. On
the test set the three thresholds compare as follows:

| Threshold | Precision | Recall | Cost per employee |
| :-------- | --------: | -----: | ----------------: |
| 0.50      |      0.73 |   0.17 |              0.67 |
| 0.27      |      0.49 |   0.53 |              0.46 |
| 0.16      |      0.38 |   0.68 |              0.44 |

`set_decision_threshold(0.16)` stores the choice in the predictor, so
`predict()` and any deployed copy use it. The 5:1 cost ratio is an assumption,
so replace it with your own numbers.

### Explain the Predictions

Permutation importance shows that the model relies on `OverTime` far more than
on any other column. Employees who work overtime leave three times as often.
For single predictions, the notebook wraps `predict_proba()` for SHAP's
`KernelExplainer`, so the explanation covers the whole ensemble and not one
model inside it. For the employee with the highest risk (0.90 against a
baseline of 0.26), the top contributions are overtime, an age of 19, one year
of experience, and a low income.

Both methods describe what the model uses, not why people leave. The data also
contains age, gender, and marital status. A real HR model would need a fairness
review before anyone acts on its scores.

## Takeaways

- AutoGluon removes the mechanical work of tabular modeling: feature types,
  model search, validation, and ensembling
- The choices that decide whether the model is useful stay with you: remove
  identifiers, pick the metric, keep a baseline, and report confidence
  intervals on small test sets
- A probability is not a decision: choose the threshold on out-of-fold
  predictions, with a metric or a cost that reflects the business

To run it yourself, clone the repo, `cd tutorials/AutoGluon`, and run
`./docker_build.sh` and `./docker_jupyter.sh`.
