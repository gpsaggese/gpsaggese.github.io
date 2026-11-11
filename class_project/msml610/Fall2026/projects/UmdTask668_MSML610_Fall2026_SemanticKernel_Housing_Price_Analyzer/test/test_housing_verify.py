"""
Tests for the labeled-metric check of agent answers.

The answers in the `REAL_*` constants were produced by the agent in live
runs (Groq `openai/gpt-oss-120b` and Ollama `qwen3:4b`); the tool results are
the values returned by the plugin on the UCI dataset.
"""

import json

import pytest

import housing_trace
import housing_verify

SUMMARY = {
    "rows": 414,
    "features": [
        "X1 transaction date",
        "X2 house age",
        "X3 distance to the nearest MRT station",
        "X4 number of convenience stores",
        "X5 latitude",
        "X6 longitude",
    ],
    "target": "Y house price of unit area",
    "missing_values": 0,
}
SCORES = {
    "Linear Regression": {
        "MAE": 5.3053556900741405,
        "RMSE": 7.3147535245216915,
        "R2": 0.681058055509592,
    },
    "Random Forest": {
        "MAE": 3.9651934308663237,
        "RMSE": 5.670520379199749,
        "R2": 0.8083282645495542,
    },
}

REAL_GROQ_ROWS = "The dataset contains **414 rows** and **6 features**."
REAL_GROQ_COMPARE = (
    "The Random Forest model has lower MAE (3.9652 vs 5.3054) and RMSE "
    "(5.6705 vs 7.3148) and a higher R² (0.8083 vs 0.6811) on the test set, "
    "so it performs better than Linear Regression."
)
REAL_GROQ_BOTH = """**Dataset overview**
- Rows: 414
- Features: X1 transaction date, X2 house age, X3 distance to the nearest MRT station, X4 number of convenience stores, X5 latitude, X6 longitude
- Target: Y house price of unit area
- Missing values: 0

**Model comparison (test set)**

| Model            | MAE   | RMSE  | R² |
|------------------|------|-------|-----|
| Linear Regression| 5.3054 | 7.3148 | 0.6811 |
| Random Forest    | 3.9652 | 5.6705 | 0.8083 |

The Random Forest model outperforms Linear Regression on all three metrics, showing lower errors and a higher R²."""
REAL_OLLAMA_COMPARE = (
    "Random Forest performs better on the test set with MAE: 3.9652, "
    "RMSE: 5.6705, R2: 0.8083."
)
REAL_OLLAMA_BOTH = """Dataset overview: 414 rows, features ["X1 transaction date", "X2 house age", "X3 distance to nearest MRT", "X4 number of convenience stores", "X5 latitude", "X6 longitude"], target: "Y house price of unit area", 0 missing values.

Model comparison:
- Linear Regression: MAE 5.3054, RMSE 7.3148, R² 0.6811
- Random Forest: MAE 3.9652, RMSE 5.6705, R² 0.8083"""

MODEL_METRICS = [
    f"{model} {metric}"
    for model in SCORES
    for metric in ("MAE", "RMSE", "R²")
]


def _trace(summary=True, scores=True, scores_status="ok"):
    trace = housing_trace.ExecutionTrace()
    if summary:
        trace.records.append(housing_trace.FunctionCallRecord(
            step=1, request_index=0, plugin="HousingAnalysis",
            function="dataset_summary", arguments={}, status="ok",
            result=json.dumps(SUMMARY),
        ))
    if scores:
        trace.records.append(housing_trace.FunctionCallRecord(
            step=2, request_index=0, plugin="HousingAnalysis",
            function="compare_models", arguments={}, status=scores_status,
            result=json.dumps(SCORES) if scores_status == "ok" else "error",
        ))
    return trace


def _statuses(report):
    return {check.metric: check.status for check in report.checks}


def test_real_groq_dataset_answer():
    report = housing_verify.check_reported_metrics(
        REAL_GROQ_ROWS, _trace(scores=False)
    )

    assert _statuses(report) == {
        "rows": "verified",
        "features": "verified",
        "missing_values": "not_reported",
    }
    assert report.ok


def test_real_groq_comparison_sentence():
    """Values in a sentence naming both models are checked per metric."""
    report = housing_verify.check_reported_metrics(
        REAL_GROQ_COMPARE, _trace(summary=False)
    )

    statuses = _statuses(report)
    assert all(statuses[m] == "verified" for m in MODEL_METRICS)
    assert report.issues == []
    assert report.ok


def test_real_groq_table_answer():
    report = housing_verify.check_reported_metrics(REAL_GROQ_BOTH, _trace())

    statuses = _statuses(report)
    assert statuses["rows"] == "verified"
    assert statuses["missing_values"] == "verified"
    # The features are listed by name, not counted.
    assert statuses["features"] == "not_reported"
    assert all(statuses[m] == "verified" for m in MODEL_METRICS)
    assert report.ok


@pytest.mark.parametrize("answer", [REAL_OLLAMA_COMPARE, REAL_OLLAMA_BOTH])
def test_real_ollama_answers(answer):
    report = housing_verify.check_reported_metrics(answer, _trace())

    assert report.ok
    assert _statuses(report)["Random Forest R²"] == "verified"


def test_rounded_and_percentage_values_match():
    answer = "Random Forest: MAE 3.97, RMSE 5.7, R² 0.81 (81%)."

    report = housing_verify.check_reported_metrics(answer, _trace(summary=False))

    statuses = _statuses(report)
    assert statuses["Random Forest MAE"] == "verified"
    assert statuses["Random Forest RMSE"] == "verified"
    assert statuses["Random Forest R²"] == "verified"
    assert report.ok


def test_wrong_value_is_a_mismatch():
    answer = "Random Forest: MAE 4.2000, RMSE 5.6705, R² 0.8083."

    report = housing_verify.check_reported_metrics(answer, _trace(summary=False))

    assert _statuses(report)["Random Forest MAE"] == "mismatch"
    assert not report.ok


def test_swapped_model_values_are_mismatches():
    answer = (
        "- Linear Regression: MAE 3.9652, RMSE 5.6705, R² 0.8083\n"
        "- Random Forest: MAE 5.3054, RMSE 7.3148, R² 0.6811"
    )

    report = housing_verify.check_reported_metrics(answer, _trace(summary=False))

    statuses = _statuses(report)
    assert all(statuses[m] == "mismatch" for m in MODEL_METRICS)
    assert not report.ok


def test_swapped_values_in_table_are_mismatches():
    answer = (
        "| Model | MAE | RMSE | R² |\n"
        "|---|---|---|---|\n"
        "| Linear Regression | 3.9652 | 5.6705 | 0.8083 |\n"
        "| Random Forest | 5.3054 | 7.3148 | 0.6811 |"
    )

    report = housing_verify.check_reported_metrics(answer, _trace(summary=False))

    assert all(_statuses(report)[m] == "mismatch" for m in MODEL_METRICS)


def test_transposed_table():
    answer = (
        "| Metric | Linear Regression | Random Forest |\n"
        "|---|---|---|\n"
        "| MAE | 5.3054 | 3.9652 |\n"
        "| RMSE | 7.3148 | 5.6705 |\n"
        "| R² | 0.6811 | 0.8083 |"
    )

    report = housing_verify.check_reported_metrics(answer, _trace(summary=False))

    assert all(_statuses(report)[m] == "verified" for m in MODEL_METRICS)


def test_unattributed_wrong_value_is_an_issue():
    answer = "The two models reach an MAE of 3.9652 and 4.5000."

    report = housing_verify.check_reported_metrics(answer, _trace(summary=False))

    assert report.issues == ["MAE 4.5000 matches no model's MAE"]
    assert not report.ok


def test_metrics_without_supporting_result_are_issues():
    """Metrics reported when compare_models did not run are flagged."""
    answer = "The dataset has 414 rows. Random Forest has an R² of 0.81."

    report = housing_verify.check_reported_metrics(answer, _trace(scores=False))

    assert report.issues == [
        "R² 0.81 is reported, but no compare_models result supports it"
    ]
    assert not report.ok


def test_failed_function_results_are_not_used():
    answer = "Random Forest has an R² of 0.81."

    report = housing_verify.check_reported_metrics(
        answer, _trace(summary=False, scores_status="error")
    )

    assert report.checks == []
    assert len(report.issues) == 1


@pytest.mark.parametrize(
    "answer, metric, status",
    [
        ("The dataset has 400 rows.", "rows", "mismatch"),
        ("There are no missing values.", "missing_values", "verified"),
        ("It has 3 missing values.", "missing_values", "mismatch"),
        ("It has 6 input features.", "features", "verified"),
        ("Number of features: 5", "features", "mismatch"),
    ],
)
def test_dataset_facts(answer, metric, status):
    report = housing_verify.check_reported_metrics(answer, _trace(scores=False))

    assert _statuses(report)[metric] == status


def test_feature_names_are_not_counts():
    """Digits inside feature names such as `X6` are not feature counts."""
    report = housing_verify.check_reported_metrics(
        "Features: X1 transaction date, X6 longitude", _trace(scores=False)
    )

    assert _statuses(report)["features"] == "not_reported"


def test_no_results_and_no_metrics():
    report = housing_verify.check_reported_metrics(
        "Hello.", housing_trace.ExecutionTrace()
    )

    assert report.checks == []
    assert report.ok
    assert report.format() == "No labeled metrics to check."
