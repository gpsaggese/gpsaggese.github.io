import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    artifacts = Path("artifacts")
    metrics_path = artifacts / "metrics" / "last_run.json"
    out_dir = artifacts / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}. Run the pipeline once to create it.")

    d = json.loads(metrics_path.read_text())
    results = d.get("results", {})

    rows = []
    for model, payload in results.items():
        if isinstance(payload, str):
            continue
        val = (payload or {}).get("val_metrics", {}) or {}
        test = (payload or {}).get("test_metrics", {}) or {}
        rows.append(
            {
                "model": model,
                "val_rmse": val.get("RMSE"),
                "test_rmse": test.get("RMSE"),
                "test_mae": test.get("MAE"),
                "test_mape": test.get("MAPE"),
                "test_r2": test.get("R2"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No model results found in last_run.json")

    df_sorted_test = df.sort_values("test_rmse", na_position="last")
    (artifacts / "metrics").mkdir(parents=True, exist_ok=True)
    df_sorted_test.to_csv(artifacts / "metrics" / "rmse_table.csv", index=False)

    # Test RMSE bar chart.
    plt.figure(figsize=(12, 5))
    plt.bar(df_sorted_test["model"].astype(str), df_sorted_test["test_rmse"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Test RMSE (lower is better)")
    plt.title("Model Comparison: Test RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_comparison_test.png", dpi=200)
    plt.close()

    # Val RMSE bar chart.
    df_sorted_val = df.dropna(subset=["val_rmse"]).sort_values("val_rmse")
    plt.figure(figsize=(12, 5))
    plt.bar(df_sorted_val["model"].astype(str), df_sorted_val["val_rmse"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Validation RMSE (lower is better)")
    plt.title("Model Comparison: Validation RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_comparison_val.png", dpi=200)
    plt.close()

    print("Wrote:")
    print(out_dir / "rmse_comparison_test.png")
    print(out_dir / "rmse_comparison_val.png")
    print(artifacts / "metrics" / "rmse_table.csv")


if __name__ == "__main__":
    main()


