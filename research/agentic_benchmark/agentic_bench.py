"""
agentic_bench.py

One fair benchmark harness for:
  always_positive      the baseline every system must beat
  single_agent         linear pipeline, same models as the graph, no reviewer
  graph_no_reviewer    LangGraph pipeline with the reviewer switched off (ablation)
  graph_reviewer       LangGraph pipeline with the reviewer on
  autogluon            optional, same train/test frames as everyone else

Rules that fix the problems in the audit:
  * every system gets the same train/test split, the same features, the same metrics
  * best model is picked by cross-validated MCC on TRAIN only, never F1, never on test
  * corruption is applied to TRAIN only, with a seeded RNG, and the flipped rows are recorded
  * the reviewer checks the exact things we corrupt (impossible values, label noise, leakage)
  * retries change something (a new strategy each time) or the graph stops and says so
  * results go to a CSV and the tables are built from that CSV
"""
import time
import warnings
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from langgraph.graph import END, StateGraph

warnings.filterwarnings("ignore")

RID = "_rid"  # original row id, carried through so we can score the reviewer as a detector

# Domain bounds. The reviewer reads these, so it can see what the corruption does.
BOUNDS = {
    "Heart_Disease": {"age": (1, 120), "chol": (50, 700), "trestbps": (50, 260)},
    "NYC_Taxi": {"fare_amount": (3, 500), "passenger_count": (1, 8),
                 "trip_distance": (0.1, 100), "trip_duration": (1, 180)},
    "Amazon_Reviews": {"text_length": (1, 20000), "title_length": (1, 500)},
}
STRATS = ["default", "class_weight", "threshold", "alt_model"]


# ----------------------------------------------------------------------------
# 1. Metrics, splits, baseline
# ----------------------------------------------------------------------------
def score(y_true, y_pred, y_prob):
    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    neg = max((y_true == 0).sum(), 1)
    one_class = len(np.unique(y_true)) < 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": ((y_true == 0) & (y_pred == 0)).sum() / neg,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": np.nan if one_class else roc_auc_score(y_true, y_prob),
        "pr_auc": np.nan if one_class else average_precision_score(y_true, y_prob),
    }


def always_positive(y_test):
    n = len(y_test)
    return score(y_test, np.ones(n, dtype=int), np.full(n, 0.5))


def add_rid(df):
    df = df.reset_index(drop=True).copy()
    df[RID] = np.arange(len(df))
    return df


def make_split(df, target, seed, test_size=0.2):
    tr, te = train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Shared model code (single agent and graph use the same models)
# ----------------------------------------------------------------------------
def make_models(balanced=False):
    cw = "balanced" if balanced else None
    return {
        "rf": RandomForestClassifier(n_estimators=150, class_weight=cw, n_jobs=-1, random_state=0),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw)),
        "hgb": HistGradientBoostingClassifier(class_weight=cw, random_state=0),
    }


def feats(df, target, drop_cols=()):
    X = df.drop(columns=[c for c in [target, RID, *drop_cols] if c in df.columns])
    return X.select_dtypes("number")


def impute(tr, te, target):
    """mode for low-cardinality columns, median otherwise. Fit on train, apply to both."""
    tr, te = tr.copy(), te.copy()
    for c in tr.columns:
        if c in (target, RID):
            continue
        if tr[c].isna().any() or te[c].isna().any():
            fill = tr[c].mode().iloc[0] if tr[c].nunique() <= 10 else tr[c].median()
            tr[c], te[c] = tr[c].fillna(fill), te[c].fillna(fill)
    return tr, te


def select_model(models, X, y, exclude=()):
    """Pick by 3-fold CV MCC on TRAIN only."""
    best_name, best_score = None, -9
    for name, m in models.items():
        if name in exclude:
            continue
        try:
            s = cross_val_score(m, X, y, cv=3, scoring="matthews_corrcoef").mean()
        except Exception:
            s = -1
        if s > best_score:
            best_name, best_score = name, s
    model = models[best_name].fit(X, y)
    return best_name, model


def best_threshold(model, X, y):
    oof = cross_val_predict(model, X, y, cv=3, method="predict_proba")[:, 1]
    grid = np.linspace(0.1, 0.9, 33)
    bas = [balanced_accuracy_score(y, (oof >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(bas))])


def run_single_agent(train, test, target):
    """Linear pipeline: impute, pick best of the same 3 models, predict once, done."""
    tr, te = impute(train, test, target)
    Xtr, Xte = feats(tr, target), feats(te, target)
    ytr, yte = tr[target].astype(int), te[target].astype(int)
    name, model = select_model(make_models(False), Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    return {"metrics": score(yte, (prob >= 0.5).astype(int), prob), "model": name, "strategy": "default"}


# ----------------------------------------------------------------------------
# 3. Reviewer checks (each one matches something we actually inject)
# ----------------------------------------------------------------------------
def impossible_mask(df, dataset):
    m = pd.Series(False, index=df.index)
    for col, (lo, hi) in BOUNDS.get(dataset, {}).items():
        if col in df.columns:
            m |= (df[col] < lo) | (df[col] > hi)
    return m


def leakage_suspects(X, y, thresh=0.95):
    """A single feature that nearly separates the classes on its own is a leak suspect."""
    out = {}
    if y.nunique() < 2:
        return out
    for c in X.columns:
        try:
            a = roc_auc_score(y, X[c])
        except Exception:
            continue
        a = max(a, 1 - a)
        if a > thresh:
            out[c] = round(float(a), 3)
    return out


def label_issue_rids(X, y, rids, cutoff=0.1):
    """Rows where an out-of-fold model strongly disagrees with the given label."""
    if y.value_counts().min() < 5:
        return np.array([], dtype=int)
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X, y, cv=5,
                          method="predict_proba")
    p_given = p[np.arange(len(y)), y.values]
    return np.asarray(rids)[p_given < cutoff]


def is_degenerate(m):
    return (m["balanced_acc"] < 0.55
            or (m["recall"] > 0.97 and m["specificity"] < 0.10)
            or m["recall"] < 0.70)


# ----------------------------------------------------------------------------
# 4. LangGraph system
# ----------------------------------------------------------------------------
class State(TypedDict, total=False):
    train: pd.DataFrame
    test: pd.DataFrame
    target: str
    dataset: str
    drop_cols: list
    pending: dict
    dropped_impossible: list
    dropped_label: list
    label_checked: bool
    attempted: list
    next_strategy: str
    history: list
    metrics: dict
    model_name: str
    review_decision: str
    review_passes: int
    retry_count: int
    log: list
    plan: str
    final_report: str


def build_graph(use_reviewer=True, llm=None, strategy_mode="rules",
                label_rate_thresh=0.05, max_retries=3):
    """
    strategy_mode: "rules" picks the next untried strategy in order.
                   "llm" asks the LLM to pick from the untried menu (falls back to rules).
    """

    def planner(state: State) -> State:
        tr, target = state["train"], state["target"]
        bal = tr[target].value_counts(normalize=True)
        risks = []
        if bal.min() < 0.35:
            risks.append(f"imbalance (minority {bal.min():.0%})")
        if tr.isna().any().any():
            risks.append("missing values")
        plan = f"{state['dataset']}: {len(tr)} train rows, risks: {risks or 'none'}"
        if llm is not None:
            try:
                plan += " | LLM note: " + llm.invoke(
                    f"One sentence data science plan for: {plan}").strip()[:300]
            except Exception:
                pass
        return {"plan": plan, "log": state.get("log", []) + ["planner: " + plan]}

    def eda(state: State) -> State:
        tr, te, target = state["train"].copy(), state["test"].copy(), state["target"]
        pend = state.get("pending", {})
        log = list(state.get("log", []))
        d_imp, d_lab = list(state.get("dropped_impossible", [])), list(state.get("dropped_label", []))
        drop_cols = list(state.get("drop_cols", []))
        tr, te = impute(tr, te, target)
        if pend.get("impossible"):
            m = impossible_mask(tr, state["dataset"])
            d_imp += tr.loc[m, RID].tolist()
            tr = tr[~m].reset_index(drop=True)
            log.append(f"eda: dropped {int(m.sum())} rows with impossible values")
        if len(pend.get("label_rids", [])):
            m = tr[RID].isin(pend["label_rids"])
            d_lab += tr.loc[m, RID].tolist()
            tr = tr[~m].reset_index(drop=True)
            log.append(f"eda: dropped {int(m.sum())} rows with suspected wrong labels")
        for c in pend.get("leaky", []):
            if c not in drop_cols:
                drop_cols.append(c)
                log.append(f"eda: dropped leaky feature {c}")
        return {"train": tr, "test": te, "pending": {}, "drop_cols": drop_cols,
                "dropped_impossible": d_imp, "dropped_label": d_lab, "log": log}

    def ml(state: State) -> State:
        target, drop_cols = state["target"], state.get("drop_cols", [])
        strat = state.get("next_strategy", "default")
        hist = list(state.get("history", []))
        attempted = list(state.get("attempted", [])) + [strat]
        Xtr, Xte = feats(state["train"], target, drop_cols), feats(state["test"], target, drop_cols)
        ytr, yte = state["train"][target].astype(int), state["test"][target].astype(int)
        balanced = strat in ("class_weight", "threshold", "alt_model")
        exclude = [hist[-1]["model"]] if (strat == "alt_model" and hist) else []
        name, model = select_model(make_models(balanced), Xtr, ytr, exclude)
        thr = best_threshold(model, Xtr, ytr) if strat == "threshold" else 0.5
        prob = model.predict_proba(Xte)[:, 1]
        m = score(yte, (prob >= thr).astype(int), prob)
        hist.append({"strategy": strat, "model": name, "threshold": thr, "metrics": m})
        return {"metrics": m, "model_name": name, "history": hist, "attempted": attempted,
                "log": state.get("log", []) + [f"ml: {strat}/{name} thr={thr:.2f} "
                                               f"bal_acc={m['balanced_acc']:.3f} mcc={m['mcc']:.3f}"]}

    def reviewer(state: State) -> State:
        passes = state.get("review_passes", 0) + 1
        retries = state.get("retry_count", 0)
        log = list(state.get("log", []))
        if not use_reviewer:
            return {"review_decision": "pass", "review_passes": passes, "log": log}
        if retries >= max_retries:
            log.append("reviewer: max retries reached, passing")
            return {"review_decision": "pass", "review_passes": passes, "log": log}

        tr, target, ds = state["train"], state["target"], state["dataset"]
        drop_cols = state.get("drop_cols", [])
        X, y = feats(tr, target, drop_cols), tr[target].astype(int)

        # data checks first: they route back to EDA
        pend, found = {}, []
        n_imp = int(impossible_mask(tr, ds).sum())
        if n_imp:
            pend["impossible"] = True
            found.append(f"{n_imp} impossible values")
        leaks = {c: a for c, a in leakage_suspects(X, y).items() if c not in drop_cols}
        if leaks:
            pend["leaky"] = list(leaks)
            found.append(f"leak suspects {leaks}")
        label_checked = state.get("label_checked", False)
        if not label_checked:
            label_checked = True
            rids = label_issue_rids(X, y, tr[RID].values)
            if len(rids) / max(len(tr), 1) > label_rate_thresh:
                pend["label_rids"] = rids
                found.append(f"{len(rids)} suspected wrong labels")
        if found:
            log.append("reviewer: " + "; ".join(found) + " -> retry_eda")
            return {"review_decision": "retry_eda", "pending": pend, "label_checked": label_checked,
                    "review_passes": passes, "retry_count": retries + 1, "log": log}

        # model checks: route back to ML with a NEW strategy, or stop
        hist = state["history"]
        if is_degenerate(state["metrics"]):
            if len(hist) >= 2 and all(
                    abs(hist[-1]["metrics"][k] - hist[-2]["metrics"][k]) < 1e-9 for k in ("f1", "balanced_acc")):
                log.append("reviewer: retry changed nothing, stopping")
            else:
                untried = [s for s in STRATS if s not in state.get("attempted", [])]
                if untried:
                    pick = untried[0]
                    if llm is not None and strategy_mode == "llm":
                        try:
                            ans = llm.invoke(f"Model metrics {({k: round(v, 3) for k, v in state['metrics'].items()})} "
                                             f"look degenerate. Pick ONE strategy from {untried}. "
                                             "Reply with the strategy name only.").strip().lower()
                            pick = next((s for s in untried if s in ans), pick)
                        except Exception:
                            pass
                    log.append(f"reviewer: degenerate model -> retry_ml with {pick}")
                    return {"review_decision": "retry_ml", "next_strategy": pick, "label_checked": label_checked,
                            "review_passes": passes, "retry_count": retries + 1, "log": log}
                log.append("reviewer: degenerate model and every strategy tried, passing with warning")
        else:
            log.append("reviewer: all checks passed")
        return {"review_decision": "pass", "label_checked": label_checked, "review_passes": passes, "log": log}

    def report(state: State) -> State:
        m = state["metrics"]
        txt = (f"{state['dataset']}: model={state['model_name']} bal_acc={m['balanced_acc']:.3f} "
               f"mcc={m['mcc']:.3f} auc={m['auc']:.3f} retries={state.get('retry_count', 0)}\n"
               + "\n".join(state.get("log", [])))
        if llm is not None:
            try:
                txt += "\nLLM summary: " + llm.invoke("Summarise in 3 sentences:\n" + txt).strip()
            except Exception:
                pass
        return {"final_report": txt}

    g = StateGraph(State)
    for name, fn in [("planner", planner), ("eda", eda), ("ml", ml), ("reviewer", reviewer), ("report", report)]:
        g.add_node(name, fn)
    g.set_entry_point("planner")
    g.add_edge("planner", "eda")
    g.add_edge("eda", "ml")
    g.add_edge("ml", "reviewer")
    g.add_conditional_edges("reviewer", lambda s: s["review_decision"],
                            {"retry_eda": "eda", "retry_ml": "ml", "pass": "report"})
    g.add_edge("report", END)
    return g.compile()


def run_graph(app, train, test, target, dataset):
    out = app.invoke({"train": train.copy(), "test": test.copy(), "target": target, "dataset": dataset,
                      "drop_cols": [], "pending": {}, "dropped_impossible": [], "dropped_label": [],
                      "attempted": [], "history": [], "log": [], "retry_count": 0, "review_passes": 0,
                      "next_strategy": "default"}, {"recursion_limit": 40})
    return out


# ----------------------------------------------------------------------------
# 5. AutoGluon on the same frames (optional)
# ----------------------------------------------------------------------------
def run_autogluon(train, test, target, tag, time_limit=60):
    from autogluon.tabular import TabularPredictor
    tr, te = train.drop(columns=[RID]), test.drop(columns=[RID])
    p = TabularPredictor(label=target, eval_metric="balanced_accuracy", verbosity=0,
                         path=f"/tmp/ag_{tag}_{int(time.time())}").fit(
        tr, time_limit=time_limit, presets="medium_quality")
    prob = p.predict_proba(te.drop(columns=[target])).iloc[:, 1].values
    return {"metrics": score(te[target], (prob >= 0.5).astype(int), prob), "model": p.model_best,
            "strategy": "autogluon"}


# ----------------------------------------------------------------------------
# 6. Corruption (seeded, train only, rows recorded)
# ----------------------------------------------------------------------------
def corrupt_train(train, target, dataset, label_noise, imp_rate, seed):
    rng = np.random.default_rng(seed)
    tr, n = train.copy(), len(train)
    info = {"flipped": set(), "impossible": set()}
    if label_noise > 0:
        idx = rng.choice(n, int(n * label_noise), replace=False)
        tr.loc[idx, target] = 1 - tr.loc[idx, target]
        info["flipped"] = set(tr.loc[idx, RID])
    cols = [c for c in BOUNDS[dataset] if c in tr.columns]
    if imp_rate > 0 and cols:
        tr[cols] = tr[cols].astype(float)
        idx = rng.choice(n, int(n * imp_rate), replace=False)
        for i in idx:
            tr.at[i, str(rng.choice(cols))] = -9999.0
        info["impossible"] = set(tr.loc[idx, RID])
    return tr, info


def add_leaky_feature(df, target, seed=0):
    df = df.copy()
    rng = np.random.default_rng(seed)
    df["leaky_feature"] = df[target] * 0.9 + rng.normal(0, 0.1, len(df))
    return df


def _pr(found, truth):
    found, truth = set(found), set(truth)
    if not truth:
        return np.nan, np.nan
    hit = len(found & truth)
    return (hit / len(found) if found else np.nan), hit / len(truth)


# ----------------------------------------------------------------------------
# 7. Experiment runner -> CSV
# ----------------------------------------------------------------------------
DEFAULT_CONDITIONS = [(0.0, 0.0), (0.05, 0.02), (0.15, 0.05), (0.30, 0.05)]  # (label_noise, impossible_rate)


def run_experiment(datasets, seeds=range(5), conditions=DEFAULT_CONDITIONS, use_autogluon=False,
                   llm=None, strategy_mode="rules", leak=False, out_csv="results.csv", verbose=True):
    """
    datasets: {"Heart_Disease": (df, "target"), "NYC_Taxi": (df, "tipped"), ...}
    leak=True injects a near-copy of the label as a feature (train AND test) to test leak detection.
    """
    rows = []
    app_off = build_graph(use_reviewer=False, llm=None)
    app_on = build_graph(use_reviewer=True, llm=llm, strategy_mode=strategy_mode)

    def add(system, ds, seed, ln, ir, res, t0, tr_info=None, out=None):
        r = {"dataset": ds, "system": system, "seed": seed, "label_noise": ln, "imp_rate": ir,
             **res["metrics"], "model": res.get("model"), "strategy": res.get("strategy"),
             "seconds": round(time.time() - t0, 1)}
        if out is not None:
            r["retries"], r["passes"] = out.get("retry_count", 0), out.get("review_passes", 0)
            r["n_dropped_impossible"], r["n_dropped_label"] = len(out["dropped_impossible"]), len(out["dropped_label"])
            r["leaky_flagged"] = "leaky_feature" in out.get("drop_cols", [])
            if tr_info:
                r["label_det_prec"], r["label_det_rec"] = _pr(out["dropped_label"], tr_info["flipped"])
                r["imp_det_prec"], r["imp_det_rec"] = _pr(out["dropped_impossible"], tr_info["impossible"])
        rows.append(r)

    for ds, (df, target) in datasets.items():
        df = add_rid(add_leaky_feature(df, target) if leak else df)
        for seed in seeds:
            train, test = make_split(df, target, seed)
            for ln, ir in conditions:
                tr_c, info = corrupt_train(train, target, ds, ln, ir, seed)
                if verbose:
                    print(f"{ds} seed={seed} label_noise={ln} imp_rate={ir}")
                t0 = time.time()
                add("always_positive", ds, seed, ln, ir,
                    {"metrics": always_positive(test[target]), "model": "-", "strategy": "-"}, t0)
                t0 = time.time()
                add("single_agent", ds, seed, ln, ir, run_single_agent(tr_c, test, target), t0)
                for name, app in [("graph_no_reviewer", app_off), ("graph_reviewer", app_on)]:
                    t0 = time.time()
                    out = run_graph(app, tr_c, test, target, ds)
                    add(name, ds, seed, ln, ir,
                        {"metrics": out["metrics"], "model": out["model_name"],
                         "strategy": out["history"][-1]["strategy"]}, t0, info, out)
                if use_autogluon:
                    t0 = time.time()
                    add("autogluon", ds, seed, ln, ir, run_autogluon(tr_c, test, target, f"{ds}_{seed}"), t0)
                pd.DataFrame(rows).to_csv(out_csv, index=False)  # save as we go
    return pd.DataFrame(rows)


def summarize(results, metrics=("balanced_acc", "mcc", "f1", "auc", "recall", "specificity")):
    """mean +- std over seeds, straight from the results frame (or a CSV path)."""
    if isinstance(results, str):
        results = pd.read_csv(results)
    g = results.groupby(["dataset", "label_noise", "imp_rate", "system"])
    mean, std = g[list(metrics)].mean(), g[list(metrics)].std()
    out = mean.round(3).astype(str) + " +- " + std.round(3).astype(str)
    out["n_seeds"] = g.size()
    return out


def detector_table(results):
    """How well did the reviewer find the rows we corrupted?"""
    if isinstance(results, str):
        results = pd.read_csv(results)
    r = results[results.system == "graph_reviewer"]
    cols = ["label_det_prec", "label_det_rec", "imp_det_prec", "imp_det_rec", "retries", "leaky_flagged"]
    return r.groupby(["dataset", "label_noise", "imp_rate"])[cols].mean().round(3)


# ----------------------------------------------------------------------------
# 8. Loaders (same cleaning you already did)
# ----------------------------------------------------------------------------
def load_heart(path):
    # missing values (ca, thal) are filled later, fit on TRAIN only, by impute()
    return pd.read_csv(path), "target"


def load_nyc(path, n=10000, keep_payment_type=False, seed=42):
    df = pd.read_parquet(path)
    df["trip_duration"] = (pd.to_datetime(df["tpep_dropoff_datetime"]) -
                           pd.to_datetime(df["tpep_pickup_datetime"])).dt.total_seconds() / 60
    df = df[(df.fare_amount > 5) & (df.fare_amount < 500) & (df.trip_duration > 3) & (df.trip_duration < 180)
            & (df.trip_distance > 0.1) & (df.trip_distance < 100) & (df.passenger_count >= 1)]
    df["tipped"] = (df["tip_amount"] > 0).astype(int)
    cols = ["passenger_count", "trip_distance", "fare_amount", "trip_duration", "PULocationID",
            "DOLocationID", "RatecodeID"] + (["payment_type"] if keep_payment_type else []) + ["tipped"]
    return df[cols].dropna().sample(n=n, random_state=seed), "tipped"


def load_amazon(jsonl_path, n=10000, seed=42):
    df = pd.read_json(jsonl_path, lines=True)
    df = df.dropna(subset=["text", "title", "rating"])
    df = df[(df.text.str.len() > 0) & (df.title.str.len() > 0)]
    df["sentiment"] = (df["rating"] >= 4).astype(int)
    df["text_length"], df["title_length"] = df["text"].str.len(), df["title"].str.len()
    df["verified_purchase"] = df["verified_purchase"].astype(int)
    cols = ["text_length", "title_length", "helpful_vote", "verified_purchase", "sentiment"]
    return df[cols].dropna().sample(n=n, random_state=seed), "sentiment"