"""Evaluate approximate Deal/No Deal strategies on held-out shows.

From project root:
  python -m src.pipelines.evaluate_optimal_strategy
  python -m src.pipelines.evaluate_optimal_strategy --max-decisions 60 --simulations 16

Default settings are deliberately modest; increase gradually after profiling.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from src.strategies.optimal_strategy import (
    EVRatioPredictor, OfferPredictor, evaluate_decision, parse_board, STANDARD_COUNTS,
)

SEED = 42


def load_data(root):
    """Load the same Part A feature table, with minimal required validation."""
    path = root / "data" / "processed" / "zbw_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Part A feature table: {path}")
    df = pd.read_csv(path)
    required = {"show_id", "round", "board_ev", "bank_offer", "decision"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    for col in ["round", "board_ev", "bank_offer", "decision"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.loc[df.show_id.notna() & df["round"].notna() &
                  df.board_ev.notna() & df.bank_offer.notna() &
                  df.decision.isin([0, 1])].copy()


def grouped_split(df):
    """Match Part A's 80/20 outer and 75/25 inner group splits."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    dev_idx, test_idx = next(outer.split(df, groups=df.show_id))
    dev, test = df.iloc[dev_idx].copy(), df.iloc[test_idx].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
    train_idx, val_idx = next(inner.split(dev, groups=dev.show_id))
    train, val = dev.iloc[train_idx].copy(), dev.iloc[val_idx].copy()
    assert set(train.show_id).isdisjoint(val.show_id)
    assert set(train.show_id).isdisjoint(test.show_id)
    assert set(val.show_id).isdisjoint(test.show_id)
    return train, val, test


def choose_states(eligible, limit, seed, near_ev_fraction=0.25):
    """Seeded, round-stratified representative cohort + separate near-EV diagnostic.

    Near-EV states are NOT added to representative metrics; doing so would bias
    population-level summaries toward ambiguous decisions.
    """
    rng = np.random.default_rng(seed)
    if limit == 0 or limit >= len(eligible):
        selected = list(eligible)
    else:
        groups = {r: [] for r in range(1, 10)}
        for entry in eligible:
            groups[int(entry[0]["round"])].append(entry)
        for group in groups.values():
            rng.shuffle(group)
        selected = []
        while len(selected) < limit and any(groups.values()):
            for r in range(1, 10):
                if groups[r] and len(selected) < limit:
                    selected.append(groups[r].pop())
    # Additional diagnostic cohort is explicitly labeled, not mixed with representative.
    n_near = min(len(eligible), round(len(selected) * near_ev_fraction))
    def closeness(entry):
        row, board = entry
        ev = float(np.mean(board))
        return abs(float(row["bank_offer"]) - ev) / max(ev, 1.0)
    diagnostic = sorted(eligible, key=closeness)[:n_near]
    return [("representative", x) for x in selected] + [("near_ev_diagnostic", x) for x in diagnostic]


def comparison_tables(results):
    representative = results.loc[results.cohort == "representative"].copy()
    summary = (results.groupby(["cohort", "model"], as_index=False)
               .agg(states=("show_id", "size"),
                    unique_shows=("show_id", "nunique"),
                    deal_rate=("recommended_decision", "mean"),
                    contestant_agreement=("matches_contestant", "mean"),
                    ev_agreement=("matches_simple_ev", "mean"),
                    uncertain_rate=("decision_uncertain", "mean"),
                    mean_mc_se=("mc_se", "mean"),
                    mean_continuation=("continuation_value", "mean")))
    pairs = representative.pivot_table(index=["state_id", "show_id", "round"],
                                       columns="model", values="recommended_decision",
                                       aggfunc="first")
    if pairs.shape[1] == 2:
        left, right = pairs.columns.tolist()
        disagreements = pairs.loc[pairs[left] != pairs[right]].reset_index()
        disagreements["model_a"] = left
        disagreements["model_b"] = right
    else:
        disagreements = pd.DataFrame(columns=["state_id", "show_id", "round", "model_a", "model_b"])
    by_round = (representative.groupby(["model", "round"], as_index=False)
                .agg(states=("show_id", "size"),
                     deal_rate=("recommended_decision", "mean"),
                     ev_agreement=("matches_simple_ev", "mean"),
                     uncertain_rate=("decision_uncertain", "mean"),
                     mean_mc_se=("mc_se", "mean")))
    return summary, disagreements, by_round


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-decisions", type=int, default=36,
                        help="Representative sample size; 0 = all eligible (slow)")
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=12)
    parser.add_argument("--near-ev-fraction", type=float, default=0.25,
                        help="Size of separate near-EV diagnostic as fraction of representative sample")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.max_decisions < 0 or args.lookahead < 1 or args.branches < 1 or args.simulations < 2:
        parser.error("max-decisions must be >= 0, lookahead/branches >= 1, simulations >= 2")
    if not 0 <= args.near_ev_fraction <= 1:
        parser.error("near-ev-fraction must be between 0 and 1")
    root = Path(__file__).resolve().parents[2]
    output = root / "outputs" / "optimal_strategy"
    output.mkdir(parents=True, exist_ok=True)
    artifact = root / "outputs" / "banker_offer" / "banker_offer_model.joblib"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing Part A model: {artifact}")
    saved = joblib.load(artifact)  # Only load a trusted artifact you trained.
    predictor = OfferPredictor(saved["model"], saved["features"], saved.get("model_name", "Part A"))
    if hasattr(predictor.estimator, "named_steps"):
        for step in predictor.estimator.named_steps.values():
            if hasattr(step, "n_jobs"):
                step.n_jobs = 1

    data = load_data(root)
    train, validation, test = grouped_split(data)
    dev = pd.concat([train, validation], ignore_index=True)
    ratios = (dev.loc[dev.board_ev > 0]
              .assign(ratio=lambda x: x.bank_offer / x.board_ev)
              .groupby("round")["ratio"].median().dropna().to_dict())
    baseline = EVRatioPredictor(ratios)

    source = "board_list" if "board_list" in test.columns else "remaining_board"
    if source not in test:
        raise ValueError("Need board_list or remaining_board for simulation")
    eligible, skipped = [], 0
    for index, row in test.iterrows():
        try:
            board = parse_board(row[source])
            r = int(row["round"])
            if r not in range(1, 10) or len(board) != STANDARD_COUNTS[r]:
                skipped += 1
                continue
            if any(k not in ratios for k in range(r + 1, min(10, r + args.lookahead + 1))):
                skipped += 1
                continue
            eligible.append((row, board))
        except (ValueError, TypeError, SyntaxError):
            skipped += 1
    if not eligible:
        raise ValueError("No eligible standard-game test states")
    selected = choose_states(eligible, args.max_decisions, args.seed, args.near_ev_fraction)
    print(f"Held-out shows: {test.show_id.nunique()} | eligible: {len(eligible)} | skipped: {skipped}")
    print(f"Selected states: {len(selected)} | lookahead={args.lookahead}, branches={args.branches}, simulations={args.simulations}")

    rows = []
    for i, (cohort, (row, board)) in enumerate(selected):
        # Both offer models use identical random draws for a fair paired comparison.
        state_seed = args.seed + i * 1009
        state_id = f"{cohort}:{row['show_id']}:{int(row['round'])}:{i}"
        for model in (predictor, baseline):
            result = evaluate_decision(board, int(row["round"]), float(row["bank_offer"]),
                                       model, seed=state_seed, branches=args.branches,
                                       lookahead=args.lookahead, simulations=args.simulations)
            rows.append({"state_id": state_id, "cohort": cohort,
                         "show_id": row["show_id"], "round": int(row["round"]),
                         "actual_decision": int(row["decision"]),
                         "model": model.name, **result})
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i+1}/{len(selected)} states", flush=True)
    results = pd.DataFrame(rows)
    results["matches_contestant"] = results.recommended_decision == results.actual_decision
    results["simple_ev_decision"] = (results.current_offer >= results.board_ev).astype(int)
    results["matches_simple_ev"] = results.recommended_decision == results.simple_ev_decision
    results.to_csv(output / "decisions.csv", index=False)
    summary, disagreements, by_round = comparison_tables(results)
    summary.to_csv(output / "model_comparison.csv", index=False)
    disagreements.to_csv(output / "model_disagreements.csv", index=False)
    by_round.to_csv(output / "metrics_by_round.csv", index=False)
    config = {"source": "Part A held-out Australian shows", "eligible_states": len(eligible),
              "skipped_invalid_states": skipped,
              "representative_states": sum(x[0] == "representative" for x in selected),
              "diagnostic_states": sum(x[0] == "near_ev_diagnostic" for x in selected),
              "lookahead": args.lookahead, "branches": args.branches,
              "simulations": args.simulations, "seed": args.seed,
              "models": [predictor.name, baseline.name],
              "limitations": ["Approximate finite-depth policy with optimistic max(offer, board EV) boundary",
                              "Monte Carlo CI describes sampling noise only, not banker-model error",
                              "No counterfactual realized-payoff or causal optimality claim",
                              "Diagnostic near-EV cohort must not be pooled with representative cohort"]}
    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    print("\n=== STRATEGY COMPARISON ===")
    print(summary.round(3).to_string(index=False))
    print(f"\nRepresentative paired disagreements: {len(disagreements)}")
    print(f"Saved outputs to {output}")
    return results


if __name__ == "__main__":
    main()
