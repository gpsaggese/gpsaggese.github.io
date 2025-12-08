from pathlib import Path

from src.features import load_and_clean, build_customer_features, feature_matrix
from src.tpot_pipeline import run_tpot_preprocessing, save_preprocess
from src.clustering import search_models, fit_best, pca_plot
from src.plot_clusters_size import plot_cluster_sizes

OUT_DIR = Path("outputs")
DATA_PATH = Path("data/Online Retail.xlsx")

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()

    print("Loading & cleaning data…")
    df = load_and_clean(str(DATA_PATH))

    print("Building customer features…")
    cust = build_customer_features(df)
    X, feat_cols = feature_matrix(cust)

    print("Running TPOT to learn preprocessing…")
    preprocess, tpot_obj = run_tpot_preprocessing(X, cust["total_spend"])

    print("Transforming features with TPOT preprocessing…")
    X_t = preprocess.fit_transform(X)
    save_preprocess(preprocess, OUT_DIR / "tpot_preprocess_transformer.joblib")

    print("Searching clustering models…")
    res = search_models(X_t, 2, 10)
    res.to_csv(OUT_DIR / "clustering_search.csv", index=False)

    best = res.iloc[0]
    print("Best model:", best.to_dict())
    model, labels = fit_best(X_t, best)

    # save customer segments
    cust_out = cust[["CustomerID", "total_spend"]].copy()
    cust_out["cluster"] = labels

    # simple naming heuristic
    mean_spend = cust_out["total_spend"].mean()
    segment_names = {}
    for c in sorted(cust_out["cluster"].unique()):
        seg_mean = cust_out.loc[cust_out["cluster"] == c, "total_spend"].mean()
        if seg_mean > mean_spend * 1.5:
            segment_names[c] = "High-Value Customers"
        elif seg_mean < mean_spend * 0.7:
            segment_names[c] = "Low-Value / At-Risk"
        else:
            segment_names[c] = "Mid-Tier Customers"
    cust_out["segment_name"] = cust_out["cluster"].map(segment_names)

    cust_out.to_csv(OUT_DIR / "customer_segments.csv", index=False)
    plot_cluster_sizes(cust_out, OUT_DIR / "cluster_sizes.png")
    pca_plot(X_t, labels, OUT_DIR / "clusters_pca.png")

    # with open(OUT_DIR / "REPORT.txt", "w") as f:
    #     f.write("Best clustering model:\n")
    #     f.write(str(best.to_dict()) + "\n\n")
    #     f.write("Segments:\n")
    #     for c, name in segment_names.items():
    #         f.write(f"Cluster {c}: {name}\n")

    print("Done! Outputs written to:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
