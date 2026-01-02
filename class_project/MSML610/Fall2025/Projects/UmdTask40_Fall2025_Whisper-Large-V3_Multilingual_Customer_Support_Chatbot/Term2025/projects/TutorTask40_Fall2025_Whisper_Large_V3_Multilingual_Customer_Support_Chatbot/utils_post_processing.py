"""
utils_post_processing.py

Helpers for summarizing logged chatbot interactions.

These functions operate on the JSONL logs written by utils_data_io.append_interaction().
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

from utils_data_io import read_interactions


def summarize_by_intent(log_path: str | Path) -> Dict[str, int]:
    """
    Count how many times each intent_label appears in the log.
    """
    records = read_interactions(log_path)
    counter: Counter[str] = Counter()

    for rec in records:
        label = rec.get("intent_label") or "UNKNOWN"
        counter[label] += 1

    # return as a normal dict for easy display
    return dict(counter)


def summarize_by_language(log_path: str | Path) -> Dict[str, int]:
    """
    Count how many interactions were detected in each language.
    """
    records = read_interactions(log_path)
    counter: Counter[str] = Counter()

    for rec in records:
        lang = rec.get("detected_language") or rec.get("language_hint") or "unknown"
        counter[lang] += 1

    return dict(counter)


def load_records(log_path: str | Path) -> List[Dict[str, Any]]:
    """
    Convenience wrapper to expose the raw records, so notebooks do not
    need to import read_interactions directly.
    """
    return read_interactions(log_path)
