import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(PROCESSED / "hourly_congestion.csv")

    # Encode labels
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    df["label"] = df["congestion_level"].map(label_map)

    numeric = ["departures", "arrivals", "total_flights", "HOUR"]
    df["AIRPORT"] = df["AIRPORT"].astype(str)

    X = df[numeric + ["AIRPORT"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["AIRPORT"])
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss"
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rev_label = {0: "Low", 1: "Medium", 2: "High"}
    preds_text = [rev_label[p] for p in preds]
    y_test_text = [rev_label[y] for y in y_test]

    print("=== Classification Report ===")
    print(classification_report(y_test_text, preds_text))

    joblib.dump(model, MODELS / "model.pkl")
    print("[SUCCESS] Saved trained model.")

if __name__ == "__main__":
    main()