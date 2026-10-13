"""Version history for notebook + story snapshots."""
from __future__ import annotations

import json
import shutil
import pathlib
import datetime
import logging

_LOG = logging.getLogger(__name__)
SESSIONS_DIR = pathlib.Path(__file__).resolve().parents[2] / "sessions"


def _history_dir(session_id: str) -> pathlib.Path:
    d = SESSIONS_DIR / session_id / "history"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _versions_index(session_id: str) -> pathlib.Path:
    return _history_dir(session_id) / "versions.json"


def _load_index(session_id: str) -> list[dict]:
    p = _versions_index(session_id)
    if p.exists():
        return json.loads(p.read_text())
    return []


def _save_index(session_id: str, versions: list[dict]):
    _versions_index(session_id).write_text(json.dumps(versions, default=str, indent=2))


def create_snapshot(session_id: str, trigger: str) -> dict:
    """Save current notebook + story as a version snapshot."""
    session_dir = SESSIONS_DIR / session_id
    versions = _load_index(session_id)
    version_id = len(versions) + 1
    snap_dir = _history_dir(session_id) / f"v{version_id}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Copy notebook if exists
    nb = session_dir / "notebook.ipynb"
    if nb.exists():
        shutil.copy2(nb, snap_dir / "notebook.ipynb")

    # Copy story if exists
    story = session_dir / "story.json"
    if story.exists():
        shutil.copy2(story, snap_dir / "story.json")

    entry = {
        "version_id": version_id,
        "trigger": trigger,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "has_notebook": nb.exists(),
        "has_story": story.exists(),
    }
    versions.append(entry)
    _save_index(session_id, versions)
    _LOG.info("Created snapshot v%d for session %s: %s", version_id, session_id, trigger)
    return entry


def list_versions(session_id: str) -> list[dict]:
    """Return all version snapshots."""
    return _load_index(session_id)


def restore_version(session_id: str, version_id: int) -> dict:
    """Auto-snapshot current state, then restore from a previous version."""
    # Auto-snapshot current state first
    create_snapshot(session_id, f"auto-save before restoring v{version_id}")

    session_dir = SESSIONS_DIR / session_id
    snap_dir = _history_dir(session_id) / f"v{version_id}"
    if not snap_dir.exists():
        raise FileNotFoundError(f"Version {version_id} not found")

    # Restore notebook
    snap_nb = snap_dir / "notebook.ipynb"
    if snap_nb.exists():
        shutil.copy2(snap_nb, session_dir / "notebook.ipynb")

    # Restore story
    snap_story = snap_dir / "story.json"
    if snap_story.exists():
        shutil.copy2(snap_story, session_dir / "story.json")

    return {"restored_version_id": version_id, "new_version_id": len(_load_index(session_id))}
