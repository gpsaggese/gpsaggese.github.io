#!/usr/bin/env python3
"""Verify ABC/FNSPID tests contain identical targets; create source comparison."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def compare(path_a: Path, path_b: Path, out: Path):
    directories = {"abc": path_a, "fnspid": path_b}
    preds = {s: pd.read_csv(p/"test_predictions.csv") for s,p in directories.items()}
    available = {src: set(df.loc[df.model.eq("price_ridge"), "asset"]) for src,df in preds.items()}
    if available["abc"] != available["fnspid"] or not available["abc"]:
        raise ValueError(f"Sources do not contain identical nonempty sets of assets: {available}")
    for asset in sorted(available["abc"]):
        rows = []
        for src, df in preds.items():
            p = df.loc[(df.asset == asset) & (df.model == "price_ridge"),
                       ["date", "actual_return"]].sort_values("date").reset_index(drop=True)
            if p.empty:
                raise ValueError(f"No price-only model for {src}/{asset}")
            rows.append(p)
        if rows[0].date.tolist() != rows[1].date.tolist():
            raise ValueError(f"TEST DATES DIFFER for {asset}: fix date window and splits before comparing")
        if not np.allclose(rows[0].actual_return, rows[1].actual_return, atol=1e-12, rtol=0):
            raise ValueError(f"TEST TARGETS DIFFER for {asset}: price series mismatch")
    results = []
    for src, directory in directories.items():
        frame = pd.read_csv(directory/"test_leaderboard.csv")
        for asset, g in frame.groupby("asset"):
            base = float(g.loc[g.model == "price_ridge", "rmse"].iloc[0])
            for _, r in g.iterrows():
                results.append({"source": src, "asset": asset, "model": r.model,
                                "n_test": int(r.n_test), "rmse": r.rmse,
                                "oos_r2": r.oos_r2,
                                "pct_news_days_test": r.pct_news_days_test,
                                "rmse_gain_vs_price": base - r.rmse,
                                "relative_rmse_gain_pct": 100 * (base-r.rmse)/base})
    result = pd.DataFrame(results)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"[PASS] ABC vs FNSPID: identical test dates and returns for {sorted(available['abc'])}")
    print(result.sort_values(["asset", "source", "rmse"]).to_string(index=False,
                                                                   float_format=lambda x:f"{x:.6f}"))
    print("Saved:",out)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--abc-dir", default="results/news_fusion/abc/evaluation")
    p.add_argument("--fnspid-dir", default="results/news_fusion/fnspid/evaluation")
    p.add_argument("--output", default="results/news_fusion/source_comparison.csv")
    args = p.parse_args()
    compare(Path(args.abc_dir), Path(args.fnspid_dir), Path(args.output))

if __name__ == "__main__":
    main()
