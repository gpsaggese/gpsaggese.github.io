"""
Minimal Flask app for interactive employee attrition predictions with LIME.

Run from the project root:

    python lime_attrition_flask_app.py

Then open http://127.0.0.1:5000 in a browser.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

import lime_attrition_utils as u


app = Flask(__name__)


def add_engineered_features(X_in: pd.DataFrame) -> pd.DataFrame:
    X2 = X_in.copy()
    if {"MonthlyIncome", "YearsAtCompany"}.issubset(X2.columns):
        X2["IncomePerYearAtCompany"] = X2["MonthlyIncome"] / (X2["YearsAtCompany"] + 1)
    if {"YearsAtCompany", "TotalWorkingYears"}.issubset(X2.columns):
        X2["TenureRatio"] = X2["YearsAtCompany"] / (X2["TotalWorkingYears"] + 1)
    if "DistanceFromHome" in X2.columns:
        X2["LongCommute"] = (X2["DistanceFromHome"] >= 10).astype(int)
    if "Age" in X2.columns:
        X2["EarlyCareer"] = (X2["Age"] <= 30).astype(int)
    return X2


# Load and prepare data
DATA_DIR = Path("data")
csvs = sorted(DATA_DIR.glob("*.csv"))
if not csvs:
    raise FileNotFoundError("No CSV found in 'data/'. Please place the IBM HR Attrition CSV there.")
CSV_PATH = csvs[0]

data_cfg = u.AttritionDataConfig()
df_raw = u.load_raw_attrition_data(str(CSV_PATH))
df = u.clean_attrition_data(df_raw, data_cfg)
X, y = u.split_features_target(df, data_cfg)
X_fe = add_engineered_features(X)
X_train, X_test, y_train, y_test = u.train_test_split_attrition(X_fe, y, data_cfg)
AVAILABLE_FEATURES = list(X_fe.columns)


def train_model_for_subset(feature_subset):
    if not feature_subset:
        raise ValueError("Feature subset is empty.")
    Xtr = X_train[feature_subset]
    pre = u.build_preprocessor(Xtr)
    mcfg = u.ModelConfig(
        use_xgboost=False,
        use_lightgbm=False,
        use_random_forest=False,
    )
    models = u.train_attrition_models(Xtr, y_train, pre, mcfg)
    model_pipeline = models["gradient_boosting"]
    explainer = u.build_lime_explainer(pre, Xtr, class_names=["Stay", "Leave"])
    return model_pipeline, explainer, pre


TEMPLATE = """
<!doctype html>
<title>Employee Attrition + LIME Demo</title>
<h1>Employee Attrition + LIME Demo</h1>

<p>Available features (original + engineered):</p>
<pre style="white-space: pre-wrap;">{{ feature_list }}</pre>

<form method="post">
  <label>Feature subset (comma-separated):</label><br>
  <input type="text" name="features" size="100"
         value="{{ default_subset }}"><br><br>

  <label>Test-set row index (0 to {{ max_idx }}):</label><br>
  <input type="number" name="row_index" value="0" min="0" max="{{ max_idx }}"><br><br>

  <input type="submit" value="Explain prediction">
</form>

{% if error %}
  <h2 style="color:red;">Error</h2>
  <pre>{{ error }}</pre>
{% endif %}

{% if prediction is not none %}
  <h2>Prediction</h2>
  <p>Predicted P(leave) = {{ prediction }}</p>

  <h2>LIME explanation (feature, weight)</h2>
  <pre>
{% for f, w in explanation %}
{{ f }}: {{ "%.3f"|format(w) }}
{% endfor %}
  </pre>
{% endif %}
"""


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None
    explanation = None

    default_subset = ",".join(AVAILABLE_FEATURES[:10])

    if request.method == "POST":
        subset_str = request.form.get("features", "")
        row_index_str = request.form.get("row_index", "0")
        try:
            row_index = int(row_index_str)
        except ValueError:
            error = f"Invalid row_index: {row_index_str}"
            row_index = 0

        subset = [s.strip() for s in subset_str.split(",") if s.strip()]

        if not subset:
            error = "Please provide at least one feature."
        else:
            missing = [f for f in subset if f not in AVAILABLE_FEATURES]
            if missing:
                error = f"Unknown features: {missing}"
            elif row_index < 0 or row_index >= len(X_test):
                error = f"row_index must be between 0 and {len(X_test) - 1}"
            else:
                try:
                    model_pipeline, explainer, pre = train_model_for_subset(subset)
                    row = X_test.iloc[row_index][subset]
                    row_df = row.to_frame().T
                    pred_proba = model_pipeline.predict_proba(row_df)[0, 1]
                    lcfg = u.LimeConfig(num_features=min(10, len(subset)), num_samples=5000)
                    exp = u.explain_single_employee(
                        explainer=explainer,
                        model_pipeline=model_pipeline,
                        raw_row=row,
                        preprocessor=pre,
                        lime_config=lcfg,
                    )
                    prediction = float(pred_proba)
                    explanation = exp.as_list()
                except Exception as e:
                    error = str(e)

    return render_template_string(
        TEMPLATE,
        feature_list=", ".join(AVAILABLE_FEATURES),
        default_subset=default_subset,
        max_idx=len(X_test) - 1,
        error=error,
        prediction=prediction,
        explanation=explanation,
    )


if __name__ == "__main__":
    app.run(debug=True)
