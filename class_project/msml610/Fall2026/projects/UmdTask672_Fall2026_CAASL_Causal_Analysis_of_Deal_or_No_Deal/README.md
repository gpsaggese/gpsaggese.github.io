# Causal Analysis of Deal or No Deal

**MSML610 · Fall 2026 · Group 28**  
**Researchers:** Stuti Patel and Riya Puri

A reproducible research project studying **how contestants decide between Deal and No Deal**, how banker offers respond to the remaining prize board, and whether modeling future offers changes a risk-neutral decision strategy.

> **Project status (November 2026):** Australian data preparation, offer-level feature engineering, an exploratory workflow, the simple expected-value (EV) benchmark, banker-offer prediction (**Part A**), and approximate sequential strategy evaluation (**Part B**) have been implemented. Risk-preference estimation, causal identification, and financial counterfactuals remain future work. Predictive associations are not causal effects.

## Quick start

Run commands **from the project root** on macOS/Linux (Windows: activate the virtual environment with `.venv\Scripts\activate`). Python 3.10+ is recommended; use a compatible scikit-learn version for loading an existing saved model.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q --strict-markers
```

**Required input:** `data/raw/zbw/deal-data.txt` (and its accompanying source documentation). The archived data pipeline reconstructs `data/processed/zbw_reconstructed.csv`. The later modeling pipelines require **`data/processed/zbw_features.csv`**. If you do not have that file, use the feature-generation command under [Data preparation](#data-preparation); the archived reconstruction CLI alone does **not** produce it.

After generating the feature CSV and training the banker model:

```bash
python -m src.pipelines.train_banker
python -m src.pipelines.evaluate_optimal_strategy --max-decisions 0 --lookahead 3 --branches 3 --simulations 12
```

`--max-decisions 0` selects every eligible held-out state; a separate near-EV diagnostic cohort is **also** evaluated, so the number of strategy evaluations may exceed the number of held-out states. Run the Part B command after the saved model exists at `outputs/banker_offer/banker_offer_model.joblib`.

## Research questions

1. Which game-state factors are associated with contestants deviating from a decision benchmark?
2. Are previous prize outcomes associated with decisions after accounting for the current offer and board?
3. How much expected value might be forgone under different risk-preference assumptions? *(Not yet estimated.)*
4. How well can banker offers be predicted, and does the offer model affect recommended actions?
5. How do risk preferences differ across contestants, rounds, or game editions? *(Future work.)*

## Repository layout

```text
.
├── README.md
├── CHANGELOG.md
├── .gitignore
├── requirements.txt
├── pytest.ini
├── data/
│   ├── raw/
│   │   ├── zbw/                   # Australian source data and documentation
│   │   └── kaggle/                # Secondary Deal or No Deal dataset
│   └── processed/                 # Reconstructed and engineered CSVs (generated)
├── src/
│   ├── load_zbw.py               # Read Australian source records
│   ├── load_kaggle.py            # Read secondary dataset
│   ├── cleaner.py                # Series and game filtering
│   ├── build_dataset.py          # Reconstruct remaining prize boards
│   ├── features.py               # Current-board and historical features
│   ├── analysis.py               # Existing decision logit analysis
│   ├── models/
│   │   └── banker_offer.py       # Part A model specifications and predictions
│   ├── strategies/
│   │   ├── ev_strategy.py        # Simple EV decision rule
│   │   └── optimal_strategy.py   # Part B approximate sequential policy
│   └── pipelines/
│       ├── train_banker.py       # Part A training and model selection
│       └── evaluate_optimal_strategy.py # Part B held-out evaluation
├── notebooks/                    # EDA and strategy exploration
├── tests/                        # Unit and integration tests
└── outputs/
    ├── banker_offer/             # Saved model, predictions, metrics
    └── optimal_strategy/         # Decision-level strategy evaluation
```

**Version note:** The original uploaded repository archive contains the core data and logit code; the `models/`, `strategies/`, `pipelines/`, and newer tests above are from the subsequently developed Parts A/B scripts. Copy those scripts into the corresponding paths before running the modeling commands. Notebooks may vary by local checkout.

## Data sources and preparation

### Primary: Australian ZBW data

The original project uses the dataset accompanying de Roos and Sarafidis (2010), *Decision Making Under Risk in Deal or No Deal*. Each raw record describes a round; the source has 20 whitespace-separated fields and uses `-999` as a missing-value marker.

| Processing stage | Rows | Shows |
|---|---:|---:|
| Raw Australian data | 4,664 | 424 |
| Cleaned main-series sample (2.00, 3.00, 3.10) | 3,421 | 311 |
| Successfully reconstructed cleaned games | 3,410 | 310 |
| Offer-level engineered feature data | **2,136** | **291** |
| Prior-luck logit sample (defined luck) | 1,845 | 291 |

The cleaned sample excludes special-show games; one cleaned show (336) fails reconstruction. Other reconstructed shows without usable banker offers do not enter the offer-level sample. Full-raw reconstruction succeeds for 422 shows; two shows (336 and 1003) fail configured board checks. These are different sample definitions and should not be conflated.

**Board reconstruction:** Starting from a series-specific 26-prize board, `build_dataset.py` removes opened prizes in sequence and computes remaining prizes, count, board expected value (`board_ev`), and `offer_ev_ratio = bank_offer / board_ev`. Mismatched prizes are recorded separately rather than silently accepted.

### Secondary: Kaggle-style Deal or No Deal data

The separately supplied `dond_game_data.csv` contains **825 rows across 107 games**. Its remaining-prize lists appear sufficient to derive the seven Part A board features, but different prize scales, game rules, round numbering, and postgame rows require validation. **It has not been incorporated into the reported Australian Part A/B results.** A future external evaluation should use this data without retraining the Australian model first.

### Data preparation

Reconstruct the raw data:

```bash
python src/build_dataset.py
```

Outputs: `data/processed/zbw_reconstructed.csv` and `data/processed/zbw_board_mismatches.csv`.

The archived code exposes an in-memory cleaned feature chain. To save the feature file required by Parts A/B, run:

```bash
python - <<'PYCODE'
from src.load_zbw import load_zbw
from src.cleaner import clean
from src.build_dataset import build_dataset
from src.features import build_features
from pathlib import Path

raw = load_zbw()
cleaned = clean(raw)
reconstructed, mismatches = build_dataset(cleaned)
features = build_features(reconstructed)
path = Path('data/processed/zbw_features.csv')
path.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(path, index=False)
print(f'Saved {len(features)} rows to {path}')
PYCODE
```

This is an explicit reproducibility command, **not** a claim that the archived repository has a single combined feature-generation CLI. Confirm that the resulting CSV contains the expected 2,136 rows and 291 shows before comparing with the reported model metrics.

### Key engineered features

| Feature | Definition / role |
|---|---|
| `board_ev` | Arithmetic mean of remaining prizes |
| `round` | Current offer round |
| `n_remaining` | Number of remaining prizes |
| `board_min`, `board_max` | Smallest and largest remaining prizes |
| `prob_above_ev` | Fraction of remaining prizes greater than board EV |
| `board_skew` | Skewness of remaining prize distribution |
| `board_list` / `remaining_board` | Remaining prizes used in simulation |
| `offer_ev_ratio` | Actual offer / board EV; **not** used to predict that same offer |
| `normalized_luck` | Prior-to-current EV change scaled by prior EV; may be undefined |
| `offers_declined` | Earlier No Deal choices, excluding current decision |

The feature builder computes historical lags before selecting offer rounds. It preserves accepted Deal decisions and removes subsequent game states. `decision=1` denotes Deal and `decision=0` denotes No Deal.

## Simple EV benchmark

The initial risk-neutral rule is:

\[\text{Deal if current offer} \geq \text{board EV; otherwise No Deal.}\]

It is a **myopic** benchmark: it ignores potential future banker offers and is not an exact optimal policy.

## Part A — Banker-offer prediction

**Goal:** Predict `bank_offer` from the observable game state. Candidate models include a mean baseline, EV-only linear regression, EV + round regression, full-feature linear regression, and Random Forest.

**Predictors:** `board_ev`, `round`, `n_remaining`, `board_min`, `board_max`, `prob_above_ev`, `board_skew`.

**Evaluation:** Split by `show_id` to prevent rounds from the same game leaking across train/validation/test. The reported split contains **174 training, 58 validation, and 59 held-out test shows**. Choose the model by validation MAE, then assess the chosen model on held-out shows.

```bash
python -m src.pipelines.train_banker
```

**Recorded held-out test results (Random Forest):**

| Metric | Result |
|---|---:|
| Test shows | 59 |
| Test offer observations | 423 |
| Mean absolute error (MAE) | **$1,547.04** |
| Root mean squared error (RMSE) | **$2,574.20** |
| R² | **0.9299** |

Outputs expected in `outputs/banker_offer/`: `banker_offer_model.joblib`, `validation_comparison.csv`, `test_predictions.csv`, `test_metrics_by_round.csv`, and `summary.json`.

**Interpretation:** The model predicts offers accurately on held-out Australian shows. This does not establish predictive performance for hypothetical boards, other countries' game versions, or decisions under changed rules.

## Part B — Approximate sequential strategy

**Goal:** Compare taking the actual current offer with the estimated value of continuing, using a banker-offer model for future states.

At each state the approximate policy evaluates:

\[V(s) \approx \max\{\text{current offer},\;\widehat{\mathbb E}[V(s')]\}\]

The implementation simulates legal prize eliminations using the Australian standard-game board schedule, predicts offers on future boards, and applies finite-depth backward induction. At the lookahead boundary it uses the **optimistic** approximation `max(predicted_offer, board_ev)`. Two offer predictors are compared: the Part A Random Forest and a simple round-specific historical median offer/EV ratio.

Run the **full held-out evaluation**:

```bash
python -m src.pipelines.evaluate_optimal_strategy \
  --max-decisions 0 \
  --lookahead 3 \
  --branches 3 \
  --simulations 12
```

`--max-decisions 0` means all eligible held-out decisions. The run also creates a separately labeled near-EV diagnostic cohort; those rows may overlap with the representative cohort and **must not be pooled** into overall statistics.

### Recorded results: full held-out evaluation

**Representative cohort:** 423 states across 59 held-out shows.

| Metric | Random Forest offers | Round EV-ratio offers |
|---|---:|---:|
| States | 423 | 423 |
| Recommended Deal rate | 8.0% | 8.7% |
| Agreement with observed contestant choices | 86.8% | 87.0% |
| Agreement with simple EV rule | 98.3% | 99.1% |
| Decisions flagged uncertain | 4.5% | 6.9% |
| Mean Monte Carlo standard error | $1,191.10 | $1,195.13 |
| Mean estimated continuation value | $17,741.65 | $17,717.40 |

**Paired model disagreements:** 3 of 423 representative states. Both offer-model-based strategies generally agree with the simple EV rule; **agreement does not establish greater realized earnings or optimality**.

**Near-EV diagnostic cohort:** 106 states across 47 shows (selected for offers close to board EV, not a representative population sample).

| Metric | Random Forest offers | Round EV-ratio offers |
|---|---:|---:|
| Recommended Deal rate | 27.4% | 30.2% |
| Agreement with simple EV rule | 94.3% | 95.3% |
| Decisions flagged uncertain | 17.9% | 20.8% |

The higher uncertain-decision rate in near-EV cases indicates that small differences between the current offer and estimated continuation value are harder to resolve with limited simulations.

**Output files** in `outputs/optimal_strategy/`:

| File | Contents |
|---|---|
| `decisions.csv` | Decision-level results, both models, simulation uncertainty, and cohorts |
| `model_comparison.csv` | Cohort-by-model summary |
| `model_disagreements.csv` | Representative states where the two offer models disagree |
| `metrics_by_round.csv` | Round-specific representative metrics |
| `run_config.json` | Evaluation configuration and explicit limitations |

### Part B limitations

- **Approximate, not exact:** finite three-round lookahead, limited Monte Carlo branches, and an optimistic terminal approximation.
- **Predictor shift:** the banker model was evaluated on historical boards, not independently validated on simulated future boards.
- **Uncertainty:** reported Monte Carlo intervals quantify simulation variability, **not** model misspecification or full policy uncertainty.
- **No causal payoff claim:** agreement with contestants or the simple EV rule does not estimate money gained, money left on the table, or causal impact.
- **Risk neutrality:** this strategy maximizes estimated expected money, not a contestant-specific utility function.

## Existing prior-outcome decision model

The earlier `src/analysis.py` implements a pooled logistic regression and per-series checks of Deal decisions against `normalized_luck`, `offer_ev_ratio`, `log_board_ev`, `n_remaining`, `round`, and series indicators. It uses game-clustered standard errors and model diagnostics. Its sample of **1,845 observations** excludes undefined prior luck. **These coefficients describe associations, not causal identification.**

## Tests and reproducibility

```bash
python -m pytest tests -q --strict-markers
python -m pytest tests/unit -q
python -m pytest -m integration -v
python -m pytest tests/test_optimal_strategy.py -q
```

The archived data pipeline reported **87 tests** before the later Part A/B additions; the current total depends on which new scripts have been copied into your checkout. The Part B scripts were also developed with automated tests, but reproduce their results in your own environment before reporting a final total.

For reproducibility, retain the random seed (42 in the reported experiments), show-grouped splits, installed dependency versions, exact input data, model artifact, and `run_config.json`. Saved `.joblib` files should only be loaded from trusted sources; scikit-learn version differences can cause incompatibilities.

## Roadmap

| Component | Status |
|---|---|
| Australian reconstruction and offer features | Implemented |
| Descriptive EDA and simple EV rule | Implemented |
| Part A: banker-offer prediction | Completed initial held-out evaluation |
| Part B: approximate risk-neutral strategy | Completed full held-out evaluation |
| Risk-averse/risk-seeking utility models | Planned |
| Decision deviations and expected monetary regret | Planned |
| Previous-outcome causal identification and sensitivity | Planned |
| External game-edition validation | Planned |

## Research integrity and interpretation

A model that predicts observed choices does not prove those choices are rational; a strategy that differs from contestants does not prove the contestant made a mistake. The project's sequential strategy is a **benchmark under stated assumptions**. Claims about counterfactual winnings, personal risk preferences, and causal effects require additional assumptions and analysis.

## References and data attribution

- de Roos and Sarafidis (2010), *Decision Making Under Risk in Deal or No Deal* (Australian ZBW source data).
- Secondary Kaggle-style Deal or No Deal dataset (`data/raw/kaggle/dond_game_data.csv`); document the original download URL and license before redistribution.

---

## Additional validation details (original pipeline)

The original implementation validates raw record widths, IDs, series, duplicate rounds, board membership, offer/EV consistency, missing or malformed boards, undefined luck, post-Deal filtering, empty outputs, and logit convergence. A missing-value marker (`-999`) in a correctly sized record is a missing value, not evidence of an absent field. A round gap makes lagged luck undefined; it does not reconstruct the missing observation.

The full raw reconstruction CLI clears stale output files and writes CSV headers even when a reconstruction result is empty. In the original feature tests, supplied finite offer/EV ratios were checked against their inputs with absolute tolerance `1e-6`. Series-specific logistic fits require enough observations and valid clustered covariance estimates.

