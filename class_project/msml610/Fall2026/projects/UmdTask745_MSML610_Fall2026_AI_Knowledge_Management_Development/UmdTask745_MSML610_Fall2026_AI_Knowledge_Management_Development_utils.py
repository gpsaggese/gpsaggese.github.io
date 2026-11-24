"""
Helpers for the API and example notebooks of the offline half.

The pipeline itself lives in the `adrqa` package under `src/`. This file only
puts `src/` on the import path, loads the JSON that `adrqa build` writes, turns
it into tables, scores it against the draft answer keys and draws the charts,
so the notebooks stay short.

Import as:

import UmdTask745_MSML610_Fall2026_AI_Knowledge_Management_Development_utils as utils
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)

# Project directory: `src/`, `data/` and `specs/` live next to this file.
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"

# The linker's name-based signals, in the order the linker applies them.
LINK_SIGNALS = (
    "explicit_class",
    "explicit_module",
    "explicit_path",
    "explicit_file",
    "adr_commit",
)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Share the notebook's logging setup with this module.

    :param notebook_log: logger of the calling notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


def add_src_to_path() -> None:
    """
    Make the `adrqa` package importable without installing it.
    """
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


class ScriptedChat:
    """
    Stand in for a chat model by returning a fixed reply.

    It lets the notebooks show the relation extractor's checks without an API
    key, exactly like the unit tests do.
    """

    def __init__(self, relations: List[Dict]) -> None:
        """
        Store the relations the fake model will claim.

        :param relations: relation dicts in the extractor's reply format
        """
        self.reply = json.dumps({"relations": relations})

    def complete(self, messages: List[Dict]) -> str:
        """
        Return the scripted reply, ignoring the prompt.

        :param messages: chat messages built by the extractor
        :return: JSON reply
        """
        _ = messages
        return self.reply


# #############################################################################
# Load the build output
# #############################################################################


def load_json(name: str, *, base: Optional[Path] = None) -> Any:
    """
    Read one JSON file written by `adrqa build`.

    :param name: file name inside `data/`, e.g. `links.json`
    :param base: directory to read from, `data/` by default
    :return: the parsed JSON value
    """
    base = base or PROJECT_DIR / "data"
    return json.loads((base / name).read_text(encoding="utf-8"))


def load_gold(name: str) -> Any:
    """
    Read one answer key from `specs/gold/`.

    :param name: `extraction.json` or `linking.json`
    :return: the `records` entry of the key
    """
    path = PROJECT_DIR / "specs" / "gold" / name
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def records_table(records: List[Dict]) -> pd.DataFrame:
    """
    Summarize the parsed records, one row per record.

    :param records: content of `records.json`
    :return: id, title, date, number of filled fields, references, supersedes
    """
    rows = []
    for r in records:
        rows.append(
            {
                "record": r["id"],
                "title": r["title"],
                "date": r["date"],
                "fields_found": 3 - len(r["missing_fields"]),
                "references": len(r["references"]),
                "supersedes": ", ".join(
                    f"{s['target']} ({s['scope']})" for s in r["supersedes"]
                ),
            }
        )
    return pd.DataFrame(rows)


def staleness_table(scores: List[Dict], threshold: float) -> pd.DataFrame:
    """
    Show the staleness score of each record, highest first.

    :param scores: content of `staleness.json`
    :param threshold: score at or above which a record is flagged
    :return: one row per record, nulls last
    """
    df = pd.DataFrame(scores)[
        [
            "record_id",
            "churn_code",
            "edits_after_first",
            "n_linked_files",
            "stale",
        ]
    ]
    df["flagged"] = df["stale"].apply(lambda s: s is not None and s >= threshold)
    return df.sort_values(
        "stale", ascending=False, na_position="last"
    ).reset_index(drop=True)


# #############################################################################
# Score against the draft answer keys
# #############################################################################


def extraction_scores(records: List[Dict], gold: List[Dict]) -> pd.DataFrame:
    """
    Count how many parsed fields match the extraction key.

    A section counts as correct when its text starts with the key's anchor,
    the opening words of that section.

    :param records: content of `records.json`
    :param gold: `specs/gold/extraction.json` records
    :return: one row per field with correct and total counts
    """
    by_id = {r["id"]: r for r in records}
    counts: Dict[str, List[int]] = {}

    def tally(field: str, ok: bool, n: int = 1) -> None:
        correct, total = counts.setdefault(field, [0, 0])
        counts[field] = [correct + (n if ok else 0), total + n]

    for g in gold:
        r = by_id[g["id"]]
        tally("title", r["title"] == g["title"])
        tally("date", r["date"] == g["date"])
        for field in ("decision", "rationale", "approach"):
            text = r["fields"][field] or ""
            tally(field, text.startswith(g[f"{field}_anchor"]))
        # Score each gold reference and each gold replacement edge on its own.
        for ref in g["references"]:
            tally("references", ref in r["references"])
        predicted = [
            {k: s[k] for k in ("target", "scope", "statement")}
            for s in r["supersedes"]
        ]
        for edge in g["supersedes"]:
            tally("supersedes", edge in predicted)
    rows = [
        {"field": f, "correct": c, "total": t} for f, (c, t) in counts.items()
    ]
    return pd.DataFrame(rows)


def expand_gold(
    gold: Dict[str, List[str]], tree: List[str]
) -> Set[Tuple[str, str]]:
    """
    Turn the linking key into (record, file) pairs.

    A directory entry in the key stands for its non-test code files, as the
    key's guideline says, so it is expanded with the same rule the linker uses.

    :param gold: `specs/gold/linking.json` records, record id -> paths
    :param tree: every path in the repository at the pinned commit
    :return: set of (`record_id`, `file_id`) pairs
    """
    from adrqa.gitstore import is_test
    from adrqa.linker import CODE_EXT

    files = set(tree)
    pairs = set()
    for rid, paths in gold.items():
        for path in paths:
            if path in files:
                pairs.add((rid, path))
                continue
            prefix = path.rstrip("/") + "/"
            pairs.update(
                (rid, f)
                for f in tree
                if f.startswith(prefix) and CODE_EXT.search(f) and not is_test(f)
            )
    return pairs


def link_scores(
    links: List[Dict],
    gold_pairs: Set[Tuple[str, str]],
    *,
    signals: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Compute micro precision and recall of record-to-file links.

    :param links: content of `links.json`
    :param gold_pairs: output of `expand_gold()`
    :param signals: keep only links from these signals (all by default)
    :return: predicted, correct, gold size, precision and recall
    """
    keep = set(signals) if signals is not None else None
    predicted = {
        (l["record_id"], l["file_id"])
        for l in links
        if keep is None or l["signal"] in keep
    }
    correct = len(predicted & gold_pairs)
    return {
        "predicted": len(predicted),
        "correct": correct,
        "gold": len(gold_pairs),
        "precision": round(correct / len(predicted), 2) if predicted else 0.0,
        "recall": round(correct / len(gold_pairs), 2) if gold_pairs else 0.0,
    }


def signal_ablation(
    links: List[Dict], gold_pairs: Set[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Switch each linker signal off in turn and score the remaining links.

    :param links: content of `links.json`
    :param gold_pairs: output of `expand_gold()`
    :return: one row for all signals, then one row per signal left out
    """
    rows = [{"setting": "all signals", **link_scores(links, gold_pairs)}]
    for signal in LINK_SIGNALS:
        rest = [s for s in LINK_SIGNALS if s != signal]
        rows.append(
            {
                "setting": f"without {signal}",
                **link_scores(links, gold_pairs, signals=rest),
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(
    scores: List[Dict], thresholds: Iterable[float]
) -> pd.DataFrame:
    """
    Count how many records each staleness threshold would flag.

    :param scores: content of `staleness.json`
    :param thresholds: candidate thresholds
    :return: threshold, number flagged and their ids
    """
    rows = []
    for t in thresholds:
        flagged = [
            s["record_id"]
            for s in scores
            if s["stale"] is not None and s["stale"] >= t
        ]
        rows.append(
            {
                "threshold": t,
                "flagged": len(flagged),
                "records": ", ".join(flagged),
            }
        )
    return pd.DataFrame(rows)


# #############################################################################
# Plots
# #############################################################################


def plot_staleness(scores: List[Dict], threshold: float) -> None:
    """
    Draw one bar per scored record, with the warning threshold as a line.

    :param scores: content of `staleness.json`
    :param threshold: warning threshold
    """
    scored = sorted(
        (s for s in scores if s["stale"] is not None), key=lambda s: s["stale"]
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        [s["record_id"] for s in scored],
        [s["stale"] for s in scored],
        color="#4C72B0",
    )
    ax.axvline(
        threshold,
        color="#C44E52",
        linestyle="--",
        label=f"threshold = {threshold:g}",
    )
    ax.set_xlabel("stale = later commits to linked code / (1 + record edits)")
    ax.set_title("Staleness per decision record (null scores not shown)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()


def plot_gaps(gaps: List[Dict], top: int = 10) -> None:
    """
    Draw the modules with the largest documentation gap.

    :param gaps: content of `gap_report.json`
    :param top: number of modules to show
    """
    rows = list(reversed(gaps[:top]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        [g["module"] for g in rows], [g["gap"] for g in rows], color="#55A868"
    )
    ax.set_xlabel("gap = complexity x (1 - coverage)")
    ax.set_title(f"Top {top} modules with complex code and no decision record")
    fig.tight_layout()
    plt.show()
