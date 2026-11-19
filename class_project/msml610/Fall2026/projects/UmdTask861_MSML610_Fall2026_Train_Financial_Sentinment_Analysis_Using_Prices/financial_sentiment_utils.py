"""Portable audit of published experiments, without broker or private corpus access.

This module recomputes aggregate checks, not the original model training. The
separate research sources and their input contracts are shipped alongside it.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

RESULT_NAMES = ("financial_sentiment_price_bert.json", "financial_sentiment_strategy_comparison.json")


def evidence_directory():
    here = Path(__file__).resolve().parent
    bundled = here / "public_results"
    return bundled if bundled.is_dir() else here.parents[2] / "reports"


def load_evidence(directory=None):
    directory = Path(directory) if directory else evidence_directory()
    values = {}
    hashes = {}
    for name in RESULT_NAMES:
        raw = (directory / name).read_bytes()
        values[name] = json.loads(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    manifest = directory / "manifest.json"
    if manifest.is_file() and json.loads(manifest.read_text())["sha256"] != hashes:
        raise ValueError("Published evidence does not match the export manifest")
    return values[RESULT_NAMES[0]], values[RESULT_NAMES[1]], hashes


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("A finite numeric observation is required")
    return float(value)


def model_comparison(bert):
    """All nine models and all companies, with error in percentage points."""
    rows = []
    for case in bert["cases"]:
        for name, metrics in case["metrics"].items():
            if metrics["samples"] != case["test_rows"]:
                raise ValueError("Models were not evaluated on equal sample counts")
            rows.append({"company": case["company"], "model": name,
                         "train_rows": case["training_rows"], "test_rows": metrics["samples"],
                         "MAE_pp": 100 * _finite(metrics["mae"]),
                         "RMSE_pp": 100 * _finite(metrics["rmse"]),
                         "direction_accuracy": _finite(metrics["direction_accuracy"])})
    return rows


def audit_strategy(comparison):
    """Recompute PnL, drawdown and paired text increment from published curves."""
    initial = _finite(comparison["starting_capital"])
    curves = comparison["curves"]
    dates = [row["date"] for row in curves]
    if initial <= 0 or not curves or dates != sorted(set(dates)):
        raise ValueError("Require positive initial capital and ordered distinct sessions")
    if len(curves) != comparison["sessions"]:
        raise ValueError("Session count differs from the published curve")
    rows = []
    for model in comparison["models"]:
        name = model["id"]
        equity = [initial] + [_finite(row[name]) for row in curves]
        peak, drawdown = initial, 0.0
        for value in equity:
            peak = max(peak, value)
            drawdown = min(drawdown, value / peak - 1)
        pnl = equity[-1] - initial
        for actual, expected in [(pnl, model["net_pnl"]), (pnl / initial, model["net_return"])]:
            if not math.isclose(actual, _finite(expected), rel_tol=1e-9, abs_tol=1e-7):
                raise ValueError(f"Curve and published summary disagree for {name}")
        # Published max drawdown uses intraday equity; daily closes cannot
        # reproduce it. Their drawdown must be no worse than the intraday value.
        reported_drawdown = _finite(model["max_drawdown"])
        if drawdown < reported_drawdown - 1e-7:
            raise ValueError("Daily drawdown exceeds the reported intraday drawdown")
        rows.append({"model": name, "gross_simulated_pnl": pnl + _finite(model["costs"]),
                     "assumed_execution_friction": model["costs"], "net_simulated_pnl": pnl,
                     "daily_close_drawdown": drawdown, "reported_intraday_drawdown": reported_drawdown,
                     "fill_count": model["fill_count"]})
    by_id = {row["model"]: row for row in rows}
    return {"models": rows, "text_increment_B_minus_C": by_id["B"]["net_simulated_pnl"] - by_id["C"]["net_simulated_pnl"],
            "cost_kind": "modeled execution friction, not verified broker charges",
            "is_training_reproduction": False, "is_independent_validation": False}


def price_label(stock_open, benchmark_open, entry_index, horizon):
    """Raw and excess opening-price returns; cash dividends are not included.

    Caller supplies an aligned exchange-session series and a verified entry
    timestamp. This arithmetic alone does not establish text availability.
    """
    if len(stock_open) != len(benchmark_open) or type(entry_index) is not int or type(horizon) is not int:
        raise ValueError("Aligned session arrays and integer indices are required")
    if horizon < 1 or entry_index < 0 or entry_index + horizon >= len(stock_open):
        raise ValueError("Outcome has not matured in the supplied session series")
    values = [_finite(series[index]) for series in (stock_open, benchmark_open)
              for index in (entry_index, entry_index + horizon)]
    if min(values) <= 0:
        raise ValueError("Opening prices must be positive")
    stock_return, market_return = values[1] / values[0] - 1, values[3] / values[2] - 1
    return {"raw_return": stock_return, "benchmark_return": market_return,
            "excess_return": stock_return - market_return, "horizon_sessions": horizon}


def audit():
    bert, strategy, hashes = load_evidence()
    rows = model_comparison(bert)
    return {"scope": "published-result integrity audit, not model retraining",
            "evidence_sha256": hashes, "model_company_comparisons": len(rows),
            "reported_training_rows": sum(case["training_rows"] for case in bert["cases"]),
            "reported_test_rows": sum(case["test_rows"] for case in bert["cases"]),
            "reported_optimizer_steps": bert["actual_optimizer_steps"],
            "strategy": audit_strategy(strategy), "alpha_established": bert["alpha_established"]}


def refit_baselines(packet_path=None):
    """Actually refit four models from observed numeric features, using NumPy.

    The dual centered ridge solution is independent of the original sklearn
    LSQR fit. Preprocessing remains frozen; no raw text is bundled or invented.
    """
    import numpy as np
    packet_path = Path(packet_path) if packet_path else Path(__file__).with_name("data") / "historical_features.json"
    packet = json.loads(packet_path.read_text())
    if packet["data_kind"] != "observed_retrospective_derived_features":
        raise ValueError("This example requires the observed feature packet")
    designs, targets = {}, {}
    ids = set()
    cutoff = packet["test_start"] + "T00:00:00Z"
    for part in ("train", "test"):
        p = packet["partitions"][part]
        for sample in p["samples"]:
            if sample["sample_id"] in ids:
                raise ValueError("Duplicate sample across partitions")
            ids.add(sample["sample_id"])
            if part == "train" and sample["exit_at"] >= cutoff:
                raise ValueError("Training outcome crosses test boundary")
            if part == "test" and sample["entry_at"] < cutoff:
                raise ValueError("Test entry precedes its boundary")
        text = np.zeros((len(p["samples"]), packet["text_columns"]))
        coo = p["text_coo"]
        text[np.asarray(coo["row"], dtype=int), np.asarray(coo["column"], dtype=int)] = coo["value"]
        price = np.asarray(p["price_features"], dtype=float)
        y = np.asarray(p["targets"], dtype=float)
        if price.shape != (len(text), 2) or y.shape != (len(text),) or not all(np.isfinite(a).all() for a in [text, price, y]):
            raise ValueError("Invalid feature matrix")
        designs[part] = {"price_ridge": price, "text_tfidf_ridge": text,
                         "price_text_ridge": np.concatenate([price, text], axis=1)}
        targets[part] = y
    y = targets["train"]
    predictions = {"train_mean": np.full(len(targets["test"]), y.mean())}
    for name, x in designs["train"].items():
        means = x.mean(axis=0)
        centered = x - means
        dual = np.linalg.solve(centered @ centered.T + packet["ridge_alpha"] * np.eye(len(y)), y - y.mean())
        predictions[name] = (designs["test"][name] - means) @ centered.T @ dual + y.mean()
    results = []
    for name, values in predictions.items():
        error = values - targets["test"]
        difference = float(np.max(np.abs(values - np.asarray(packet["expected_predictions"][name]))))
        if difference > 1e-6:
            raise ValueError(f"Independent ridge solution differs from recorded predictions: {name}")
        results.append({"model": name, "train_rows": len(y), "test_rows": len(error),
                        "MAE_pp": float(np.abs(error).mean()) * 100,
                        "RMSE_pp": float(np.sqrt(np.mean(error ** 2))) * 100,
                        "direction_accuracy": float(np.mean(np.sign(values) == np.sign(targets["test"]))),
                        "max_prediction_difference": difference})
    return results


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
