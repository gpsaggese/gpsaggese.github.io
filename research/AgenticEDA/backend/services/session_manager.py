"""Session lifecycle helpers: create, list, get, delete."""
from __future__ import annotations

import json
import pathlib
import shutil
import uuid
from datetime import datetime, timezone

import pandas as pd

SESSIONS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "sessions"


def create_session(
    file_path: pathlib.Path,
    original_filename: str,
    df: pd.DataFrame,
    metadata: dict,
) -> str:
    """Create a new session directory, persist the upload and metadata.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the temporary uploaded file on disk.
    original_filename : str
        Original name of the uploaded file.
    df : pd.DataFrame
        The parsed DataFrame (used only for reference; not persisted as parquet here).
    metadata : dict
        Metadata dict returned by ``load_file`` (source_format, row_count, etc.).

    Returns
    -------
    str
        The newly created session ID.
    """
    session_id = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / session_id
    uploads_dir = session_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Copy the uploaded file into the session uploads directory
    dest = uploads_dir / original_filename
    shutil.copy2(str(file_path), str(dest))

    # Write metadata.json
    meta = {
        "session_id": session_id,
        "original_filename": original_filename,
        "source_format": metadata.get("source_format", "unknown"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": metadata.get("row_count", len(df)),
        "col_count": metadata.get("col_count", len(df.columns)),
    }
    (session_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return session_id


def get_session_dir(session_id: str) -> pathlib.Path:
    """Return the session directory path.

    Raises
    ------
    FileNotFoundError
        If the session directory does not exist.
    """
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session not found: {session_id}")
    return session_dir


def list_sessions() -> list[dict]:
    """Read metadata.json from every session directory.

    Returns
    -------
    list[dict]
        List of metadata dicts, one per session.
    """
    sessions: list[dict] = []
    if not SESSIONS_DIR.is_dir():
        return sessions
    for child in sorted(SESSIONS_DIR.iterdir()):
        meta_path = child / "metadata.json"
        if child.is_dir() and meta_path.exists():
            try:
                sessions.append(json.loads(meta_path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
    return sessions


def delete_session(session_id: str) -> None:
    """Remove a session directory and all its contents.

    Raises
    ------
    FileNotFoundError
        If the session directory does not exist.
    """
    session_dir = get_session_dir(session_id)
    try:
        from backend.services.steering_service import clear_steering

        clear_steering(session_id)
    except Exception:
        pass
    shutil.rmtree(session_dir)
