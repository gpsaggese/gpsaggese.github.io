import os
from azua_utils import load_melbourne, train_select_best, save_artifacts

if __name__ == "__main__":
    candidates = [
        os.environ.get("DATA_PATH"),
        "data/melb_data.csv",
    ]
    csv = next((p for p in candidates if p and os.path.exists(p)), None)
    if not csv:
        raise SystemExit("No dataset found. Put data/melb_data.csv or set DATA_PATH.")
    print(f"Using dataset: {csv}")

    df = load_melbourne(csv)
    bundle = train_select_best(df)

    print("CV results:")
    for row in bundle["all_results"]:
        print(row)
    print(f"Best: {bundle['best']['name']} | RMSE={bundle['best']['rmse']:.0f}, R2={bundle['best']['r2']:.3f}")

    save_artifacts(bundle)
    print("Saved to artifacts/")

