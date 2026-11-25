"""Plots and tables for the full study. Everything is built from the results CSV or a graph run."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import agentic_bench as ab

SYS_ORDER = ["always_positive", "single_agent", "graph_no_reviewer", "graph_reviewer_v1",
             "graph_reviewer_fixed", "autogluon"]
SYS_LABEL = {"always_positive": "Always positive", "single_agent": "Single agent",
             "graph_no_reviewer": "Graph, no reviewer", "graph_reviewer_v1": "Graph + reviewer (old)",
             "graph_reviewer_fixed": "Graph + reviewer (fixed)", "autogluon": "AutoGluon"}
COL = {"always_positive": "#b0b0b0", "single_agent": "#4c78a8", "graph_no_reviewer": "#72b7b2",
       "graph_reviewer_v1": "#f58518", "graph_reviewer_fixed": "#54a24b", "autogluon": "#b279a2"}


def eda_plot(df, target, name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    df[target].value_counts().sort_index().plot(kind="bar", ax=ax[0], color=["steelblue", "coral"])
    ax[0].set_title(f"{name}: class distribution"); ax[0].set_xlabel("class"); ax[0].set_ylabel("count")
    sns.heatmap(df.select_dtypes("number").corr(), ax=ax[1], cmap="coolwarm", center=0)
    ax[1].set_title(f"{name}: correlation")
    plt.tight_layout(); plt.savefig(f"{name}_eda.png", dpi=110, bbox_inches="tight"); plt.show()


def shap_plot(out, name, max_rows=150):
    """SHAP for the model the graph finally used, on its TEST rows."""
    import shap
    model, cols = out["model_obj"], out["feature_names"]
    Xte = out["test"][cols].sample(min(max_rows, len(out["test"])), random_state=0)
    Xbg = out["train"][cols].sample(min(100, len(out["train"])), random_state=0)
    f = lambda X: model.predict_proba(pd.DataFrame(X, columns=cols))[:, 1]
    sv = shap.Explainer(f, Xbg, algorithm="permutation")(Xte, max_evals=2 * len(cols) + 1)
    shap.summary_plot(sv.values, Xte, show=False)
    plt.title(f"{name}: SHAP ({out['model_name']})"); plt.tight_layout()
    plt.savefig(f"{name}_shap.png", dpi=110, bbox_inches="tight"); plt.show()


def _agg(res, metric, cond_cols=("label_noise",)):
    g = res.groupby(list(cond_cols) + ["system"])[metric]
    return g.mean().unstack("system"), g.std().unstack("system")


def bar_by_condition(res, dataset, metric="mcc", title=None):
    d = res[res.dataset == dataset]
    m, s = _agg(d, metric)
    cols = [c for c in SYS_ORDER if c in m.columns]
    ax = m[cols].plot(kind="bar", yerr=s[cols], figsize=(11, 4.5), capsize=3,
                      color=[COL[c] for c in cols], width=0.85)
    ax.set_xlabel("label noise in training data"); ax.set_ylabel(metric.upper() + " (test, mean ± sd)")
    ax.set_title(title or f"{dataset}: {metric.upper()} on the held-out test set")
    ax.legend([SYS_LABEL[c] for c in cols], fontsize=8, ncol=3, loc="upper right")
    plt.xticks(rotation=0); plt.tight_layout(); plt.show()


def train_vs_test(res, dataset, metric="mcc"):
    d = res[(res.dataset == dataset) & res.system.isin(["graph_no_reviewer", "graph_reviewer_v1", "graph_reviewer_fixed"])]
    if f"train_{metric}" not in d.columns:
        return
    g = d.groupby(["label_noise", "system"])[[f"train_{metric}", metric]].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    w, labels = 0.13, sorted(d.label_noise.unique())
    for i, sname in enumerate(["graph_no_reviewer", "graph_reviewer_v1", "graph_reviewer_fixed"]):
        for j, ln in enumerate(labels):
            tr, te = g.loc[(ln, sname)]
            ax.bar(j + (i * 2) * w, tr, w, color=COL[sname], alpha=0.45, label=f"{SYS_LABEL[sname]} train" if j == 0 else None)
            ax.bar(j + (i * 2 + 1) * w, te, w, color=COL[sname], label=f"{SYS_LABEL[sname]} test" if j == 0 else None)
    ax.set_xticks(np.arange(len(labels)) + 2.5 * w); ax.set_xticklabels([f"{l:.0%}" for l in labels])
    ax.set_xlabel("label noise"); ax.set_ylabel(metric.upper())
    ax.set_title(f"{dataset}: train vs test {metric.upper()} (a big gap means overfitting)")
    ax.legend(fontsize=7, ncol=3); plt.tight_layout(); plt.show()


def f1_vs_mcc(res, dataset):
    """Why F1 alone is unsafe: the always-positive model gets a high F1."""
    d = res[(res.dataset == dataset) & (res.label_noise == 0)]
    g = d.groupby("system")[["f1", "mcc", "balanced_acc"]].mean().reindex([s for s in SYS_ORDER if s in d.system.unique()])
    ax = g.plot(kind="bar", figsize=(10, 4), color=["#4c78a8", "#54a24b", "#f58518"])
    ax.set_xticklabels([SYS_LABEL[s] for s in g.index], rotation=20, ha="right")
    ax.set_title(f"{dataset} (clean data): F1 vs MCC vs balanced accuracy"); plt.tight_layout(); plt.show()


def detection_heatmap(res):
    r = res[res.system.isin(["graph_reviewer_v1", "graph_reviewer_fixed"])]
    cols = [c for c in ["imp_det_prec", "imp_det_rec", "label_det_prec", "label_det_rec"] if c in r.columns]
    g = r[r.label_noise > 0].groupby(["dataset", "system"])[cols].mean()
    if g.empty:
        return
    plt.figure(figsize=(8, 0.6 * len(g) + 1.5))
    sns.heatmap(g, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1)
    plt.title("How well the Reviewer found the corrupted rows (corrupted conditions)"); plt.tight_layout(); plt.show()


def leak_plot(leak_res):
    g = leak_res.groupby("system")[["mcc", "f1", "train_mcc"]].mean()
    g = g.reindex([s for s in SYS_ORDER if s in g.index])
    ax = g.plot(kind="bar", figsize=(9, 4), color=["#4c78a8", "#54a24b", "#f58518"])
    ax.set_xticklabels([SYS_LABEL[s] for s in g.index], rotation=20, ha="right")
    ax.axhline(0.65, ls="--", color="k", lw=0.8); ax.text(0, 0.67, "honest score (about 0.65)", fontsize=8)
    ax.set_title("Injected leaked feature: a score near 1.0 is the leak, not skill"); plt.tight_layout(); plt.show()


def summary_table(res, dataset):
    cols = ["f1", "mcc", "balanced_acc", "recall", "specificity", "auc"]
    g = res[res.dataset == dataset].groupby(["label_noise", "system"])[cols]
    out = g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)
    return out.reindex([s for s in SYS_ORDER if s in res.system.unique()], level="system")
