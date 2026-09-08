"""Process-local steering queue for active AgenticEDA runs."""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

_LOCK = Lock()
_QUEUES: dict[str, list[dict]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def enqueue_steering(
    session_id: str,
    content: str,
    message_id: str | None = None,
) -> dict:
    """Append a steering message to a session queue."""
    item = {
        "id": message_id or str(uuid4()),
        "session_id": session_id,
        "content": content,
        "created_at": _now(),
        "status": "queued",
        "read_at": None,
    }
    with _LOCK:
        _QUEUES.setdefault(session_id, []).append(item)
        return dict(item)


def drain_steering(session_id: str, limit: int | None = None) -> list[dict]:
    """Mark queued steering messages as read and return them in FIFO order."""
    with _LOCK:
        items = _QUEUES.get(session_id, [])
        queued = [item for item in items if item.get("status") == "queued"]
        if limit is not None:
            queued = queued[:limit]
        read_at = _now()
        drained: list[dict] = []
        for item in queued:
            item["status"] = "read"
            item["read_at"] = read_at
            drained.append(dict(item))
        return drained


def get_steering_items(session_id: str) -> list[dict]:
    """Return all steering items for a session."""
    with _LOCK:
        return [dict(item) for item in _QUEUES.get(session_id, [])]


def clear_steering(session_id: str) -> None:
    """Remove all steering items for a session."""
    with _LOCK:
        _QUEUES.pop(session_id, None)
