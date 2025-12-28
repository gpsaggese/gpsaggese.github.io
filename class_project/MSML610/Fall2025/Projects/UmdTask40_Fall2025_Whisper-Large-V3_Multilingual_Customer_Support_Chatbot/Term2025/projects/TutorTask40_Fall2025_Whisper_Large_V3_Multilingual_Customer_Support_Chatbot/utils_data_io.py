"""
utils_data_io.py

Small helper functions for logging and reading chatbot interactions.

The notebooks (especially example.ipynb) will use these helpers instead of
writing logging logic inline.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from api_contract import ChatRequest, ChatResponse


def _serialize_request_response(
    request: ChatRequest,
    response: ChatResponse,
) -> Dict[str, Any]:
    """
    Convert ChatRequest + ChatResponse into a flat dict that is easy to
    store as one JSON object per line (JSONL format).
    """
    # Determine request type
    if request.audio_path is not None:
        request_type = "audio"
        request_text = None
        audio_path_str = str(request.audio_path)
    else:
        request_type = "text"
        request_text = request.text
        audio_path_str = None

    intent_label = None
    intent_score = None
    if response.intent is not None:
        intent_label = response.intent.label
        intent_score = response.intent.score

    record: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": request.user_id,
        "request_type": request_type,
        "request_text": request_text,
        "audio_path": audio_path_str,
        "language_hint": request.language_hint,
        "metadata": request.metadata or {},
        "reply_text": response.reply_text,
        "detected_language": response.detected_language,
        "intent_label": intent_label,
        "intent_score": intent_score,
        "debug_info": response.debug_info or {},
    }

    return record


def append_interaction(
    log_path: str | Path,
    request: ChatRequest,
    response: ChatResponse,
) -> None:
    """
    Append a single interaction (request + response) as one JSON object
    on its own line in a JSONL log file.

    If the directory does not exist, it will be created.
    """
    log_path = Path(log_path)
    if log_path.parent and not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)

    record = _serialize_request_response(request, response)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_interactions(log_path: str | Path) -> List[Dict[str, Any]]:
    """
    Read all interactions from a JSONL log file and return them
    as a list of dictionaries.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

    return records
