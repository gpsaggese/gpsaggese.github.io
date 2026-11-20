"""
Score agent findings against the benchmark labels.

Scoring works at two levels:
- Snippet level: a snippet is flagged if the agent reports any finding on it,
  which gives TP / FP / FN / TN over vulnerable and safe snippets
- Finding level: a finding localizes the bug if its line is within
  `line_window` lines of a labelled bug line and, in strict mode, its CWE is
  the labelled CWE or a related one

Import as:

import scoring_lib as scolib
"""

import logging
import math
from typing import Dict, FrozenSet, List, Optional

import pandas as pd

import helpers.hdbg as hdbg

import findings_lib as finlib

_LOG = logging.getLogger(__name__)

# Metrics that need finding locations; NaN for a run that reports none.
_LOCALIZATION_METRICS = (
    "recall_line",
    "recall_strict",
    "cwe_accuracy",
    "finding_precision",
)

# #############################################################################
# Constants
# #############################################################################

# Max distance in lines between a finding and a labelled bug line.
LINE_WINDOW = 3

# Groups of CWEs that count as the same weakness (parent / child / variant).
# E.g., a tool reporting CWE-943 (NoSQL / generic query) for a SQL bug is right.
_RELATED_CWE_GROUPS: List[FrozenSet[int]] = [
    # SQL and query injection.
    frozenset({89, 564, 943}),
    # Cross-site scripting.
    frozenset({79, 80, 83, 116}),
    # OS command injection.
    frozenset({77, 78, 88}),
    # Code and template injection.
    frozenset({94, 95, 1336}),
    # Path traversal and external control of file names.
    frozenset({22, 23, 36, 73}),
    # Deserialization of untrusted data.
    frozenset({502}),
    # XML external entities.
    frozenset({611, 776}),
    # Server-side request forgery.
    frozenset({918}),
    # Weak hashing and broken crypto.
    frozenset({327, 328, 759, 760, 916}),
    # Hard-coded credentials and keys.
    frozenset({259, 321, 798}),
    # Weak randomness.
    frozenset({330, 338}),
    # Missing or broken authorization, including IDOR.
    frozenset({284, 285, 639, 862, 863}),
    # Debug mode and debug information exposure.
    frozenset({215, 489}),
    # Plain-text or weakly protected stored passwords.
    frozenset({256, 312, 522}),
    # Sensitive data in logs.
    frozenset({200, 359, 532}),
    # Failing open on errors.
    frozenset({636, 703, 754, 755}),
]


# #############################################################################
# Matching
# #############################################################################


def cwe_matches(predicted: Optional[int], actual: int) -> bool:
    """
    Return whether a reported CWE counts as the labelled CWE.

    :param predicted: CWE reported by the agent, or `None`
    :param actual: labelled CWE
    :return: `True` if equal or in the same related group
    """
    if predicted is None or pd.isna(predicted):
        return False
    predicted = int(predicted)
    if predicted == actual:
        return True
    is_related = any(
        predicted in group and actual in group for group in _RELATED_CWE_GROUPS
    )
    return is_related


def score_snippets(
    labels: pd.DataFrame,
    findings: pd.DataFrame,
    *,
    line_window: int = LINE_WINDOW,
) -> pd.DataFrame:
    """
    Score the findings of one run, one row per snippet.

    :param labels: table from `benchmark_lib.load_labels()`
    :param findings: findings of a single agent and run
    :param line_window: max distance between a finding and a bug line
    :return: one row per labelled snippet with columns:
        - `snippet_id`, `vulnerable`, `pair_id`
        - `n_findings`: findings reported on the snippet
        - `flagged`: whether the agent reported anything on the snippet
        - `localized_line`: a finding is within the window of a bug line
        - `localized_strict`: same, and its CWE matches the label
        - `n_findings_strict`: findings that localize the bug in strict mode
    """
    hdbg.dassert_lte(0, line_window)
    rows = []
    by_snippet = {sid: df for sid, df in findings.groupby("snippet_id")}
    for label in labels.itertuples(index=False):
        snippet_findings = by_snippet.get(label.snippet_id, findings.iloc[0:0])
        n_line = 0
        n_strict = 0
        for finding in snippet_findings.itertuples(index=False):
            # A finding with no location flags the snippet but localizes
            # nothing.
            if int(finding.line) == finlib.NO_LINE:
                continue
            near = any(
                abs(int(finding.line) - bug_line) <= line_window
                for bug_line in label.bug_lines
            )
            if near:
                n_line += 1
                if cwe_matches(finding.cwe, label.cwe):
                    n_strict += 1
        rows.append(
            {
                "snippet_id": label.snippet_id,
                "vulnerable": label.vulnerable,
                "pair_id": label.pair_id,
                "n_findings": len(snippet_findings),
                "flagged": len(snippet_findings) > 0,
                "localized_line": n_line > 0,
                "localized_strict": n_strict > 0,
                "n_findings_strict": n_strict,
            }
        )
    df = pd.DataFrame(rows)
    return df


# #############################################################################
# Metrics
# #############################################################################


def _safe_div(num: float, den: float) -> float:
    """
    Divide, returning NaN when the denominator is 0.
    """
    ratio = num / den if den else float("nan")
    return ratio


def compute_metrics(scored: pd.DataFrame) -> Dict[str, float]:
    """
    Compute detection, localization, and pair metrics for one run.

    :param scored: table from `score_snippets()`
    :return: metrics:
        - `tp`, `fp`, `fn`, `tn`: snippet-level counts
        - `precision`, `recall`, `f1`
        - `fpr`: FP / (FP + TN), share of safe snippets flagged
        - `fdr`: FP / (FP + TP), share of flagged snippets that are safe (the
          project brief calls this the "false positive rate")
        - `mcc`: Matthews correlation coefficient
        - `youden_j`: recall - fpr, the OWASP Benchmark score
        - `recall_line`: share of vulnerable snippets with a finding near a bug
          line
        - `recall_strict`: same, with the CWE matching too
        - `cwe_accuracy`: among line-localized snippets, share with the right
          CWE
        - `finding_precision`: share of all findings that localize a bug in
          strict mode
        - `paired_accuracy`: share of pairs with the vulnerable snippet flagged
          and its safe twin not flagged
    """
    vulnerable = scored["vulnerable"]
    flagged = scored["flagged"]
    tp = int((vulnerable & flagged).sum())
    fp = int((~vulnerable & flagged).sum())
    fn = int((vulnerable & ~flagged).sum())
    tn = int((~vulnerable & ~flagged).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    fpr = _safe_div(fp, fp + tn)
    mcc = _safe_div(
        tp * tn - fp * fn,
        math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
    )
    # Localization only makes sense on vulnerable snippets.
    vuln = scored[vulnerable]
    n_line = int(vuln["localized_line"].sum())
    n_strict = int(vuln["localized_strict"].sum())
    # A pair is right when its vulnerable snippet is flagged and its safe twin
    # is not.
    paired = scored[scored["pair_id"] != ""]
    pair_ok = paired.assign(ok=paired["flagged"] == paired["vulnerable"])
    pair_ok = pair_ok.groupby("pair_id")["ok"].all()
    metrics = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fdr": _safe_div(fp, fp + tp),
        "mcc": mcc,
        "youden_j": recall - fpr,
        "recall_line": _safe_div(n_line, len(vuln)),
        "recall_strict": _safe_div(n_strict, len(vuln)),
        "cwe_accuracy": _safe_div(n_strict, n_line),
        "finding_precision": _safe_div(
            int(scored["n_findings_strict"].sum()), int(scored["n_findings"].sum())
        ),
        "paired_accuracy": _safe_div(int(pair_ok.sum()), len(pair_ok)),
    }
    return metrics


def score_runs(
    labels: pd.DataFrame,
    findings: pd.DataFrame,
    runs: pd.DataFrame,
    *,
    line_window: int = LINE_WINDOW,
) -> pd.DataFrame:
    """
    Compute metrics for every run, including runs with no findings.

    :param labels: table from `benchmark_lib.load_labels()`
    :param findings: table from `findings_lib.read_findings()`
    :param runs: table from `findings_lib.read_runs()`
    :param line_window: max distance between a finding and a bug line
    :return: one row per (`agent`, `run_id`) with the metrics of
        `compute_metrics()`
    """
    hdbg.dassert_lte(1, len(runs), "No runs to score")
    rows = []
    for run in runs.to_dict("records"):
        mask = (findings["agent"] == run["agent"]) & (
            findings["run_id"] == run["run_id"]
        )
        # Score a partial run only on the snippets it reviewed.
        run_labels = labels
        snippet_ids = run.get("snippet_ids")
        if isinstance(snippet_ids, list):
            run_labels = labels[labels["snippet_id"].isin(snippet_ids)]
        run_findings = findings[mask]
        scored = score_snippets(run_labels, run_findings, line_window=line_window)
        metrics = compute_metrics(scored)
        # An agent that reports no locations (e.g., Vulnhuntr) cannot be scored
        # on localization: report N/A rather than 0.
        has_findings = len(run_findings) > 0
        if has_findings and (run_findings["line"] == finlib.NO_LINE).all():
            for key in _LOCALIZATION_METRICS:
                metrics[key] = float("nan")
        rows.append(
            {
                "agent": run["agent"],
                "run_id": run["run_id"],
                "n_snippets": len(run_labels),
                **metrics,
            }
        )
    df = pd.DataFrame(rows).sort_values(["agent", "run_id"]).reset_index(drop=True)
    return df
