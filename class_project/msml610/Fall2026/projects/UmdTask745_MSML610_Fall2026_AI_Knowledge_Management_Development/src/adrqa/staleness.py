"""Staleness score and dead identifiers (specs/01-corpus.md explains the corrected formula)."""
import re

from .gitstore import GitStore
from .linker import SHARED_CONFIG, extract_tokens


def dead_identifiers(store: GitStore, text: str, classes: dict, dirs_by_name: dict) -> list[str]:
    """Tokens that existed in history and appear nowhere outside docs/ at the pinned SHA."""
    tokens = extract_tokens(text)
    dead = []
    for token in sorted(tokens["camel"] - set(classes)):
        if store.last_touch(f"**/{token}.java") and not store.used_outside_docs(token):
            dead.append(token)
    for token in sorted(tokens["kebab"] - set(dirs_by_name)):
        if store.last_touch(f"**/{token}/**") and not store.used_outside_docs(token):
            dead.append(token)
    return sorted(dead)


def score_record(store: GitStore, record: dict, files: list[str], dead: list[str]) -> dict:
    history = store.history(record["path"])
    first = history[0][0]
    edits = len(history) - 1
    result = {"record_id": record["id"], "edits_after_first": edits, "n_linked_files": len(files),
              "dead_identifiers": dead}
    if not files:
        # Not 0: 0 would read as "fresh" when there is nothing to measure.
        return {**result, "stale": None, "reason": "no_linked_files", "churn_code": None}
    commits: set[str] = set()
    for f in files:
        if not SHARED_CONFIG.match(f):
            commits |= store.commits_since(first, f)
    return {**result, "stale": round(len(commits) / (1 + edits), 2), "reason": None, "churn_code": len(commits)}


_REMOVAL = re.compile(r"\b(remov\w*|deprecat\w*|retir\w*|stop publishing|drop\w*|delet\w*)\b", re.I)


def is_removal_decision(record: dict | None) -> bool:
    """Does the decision itself say it removes or retires something? Then a vanished name is expected."""
    return bool(record and _REMOVAL.search(record["fields"].get("decision") or ""))


def warnings_for(score: dict, threshold: float, record: dict | None = None) -> list[dict]:
    """Each warning has a technical `detail` and a `plain` sentence for a reader who doesn't know the code."""
    rid, warnings = score["record_id"], []
    removal = is_removal_decision(record)
    if score["stale"] is None:
        plain = ("Nothing in the code matches this decision any more, so there is no staleness score. "
                 "For a decision about removing something, that is expected." if removal else
                 "No code in the repository matches this decision, so there is no staleness score to show.")
        warnings.append({"record_id": rid, "kind": "no_linked_code", "plain": plain,
                         "detail": "no linked code found at this commit, so no staleness score"})
    elif score["stale"] >= threshold:
        n = score["churn_code"]
        warnings.append({"record_id": rid, "kind": "stale_score", "detail": f"linked code changed in {n} commits since this record was written",
                         "plain": f"The code this decision covers has changed in {n} commits since it was written, so the decision may no longer "
                                  "describe it exactly. Check the code before relying on it. Some of those commits may simply be the decision being carried out."})
    for token in score["dead_identifiers"]:
        plain = (f"{token} no longer exists. This decision is about removing or retiring things, so that is expected: it shows the decision was carried out."
                 if removal else
                 f"{token} no longer exists. If the decision was about replacing it, that is expected. Otherwise this part of the decision's wording may be out of date.")
        warnings.append({"record_id": rid, "kind": "dead_identifier", "identifier": token, "plain": plain,
                         "detail": f"names {token}, which no longer exists in the code"})
    return warnings
