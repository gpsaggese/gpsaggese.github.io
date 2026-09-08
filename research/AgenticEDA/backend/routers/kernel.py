"""Kernel router: execute code in the session IPython kernel."""
from __future__ import annotations

import json
import pathlib

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.kernel_manager import execute_code, is_kernel_alive, shutdown_kernel
from backend.services.session_manager import get_session_dir

router = APIRouter(tags=["kernel"])


class ExecuteCellBody(BaseModel):
    """Body for POST /kernel/{session_id}/execute."""

    cell_id: str = Field(min_length=1, max_length=200)
    code: str
    timeout: int = Field(default=60, ge=1, le=120)


def _first_uploaded_file(session_dir: pathlib.Path) -> pathlib.Path | None:
    uploads_dir = session_dir / "uploads"
    if not uploads_dir.is_dir():
        return None
    files = sorted(path for path in uploads_dir.iterdir() if path.is_file())
    return files[0] if files else None


def _dataset_preload_code(dataset_path: pathlib.Path) -> str:
    """Return setup code that loads the session dataset as df if df is missing."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    return f"""
import sys
from pathlib import Path

_agenticeda_repo_root = {json.dumps(str(repo_root))}
if _agenticeda_repo_root not in sys.path:
    sys.path.insert(0, _agenticeda_repo_root)

if "df" not in globals():
    from src.ingest.file_loader import load_file
    df, dataset_metadata = load_file(Path({json.dumps(str(dataset_path))}))
"""


def _trim_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n...[truncated after {max_chars} characters]"


def _limit_outputs(outputs: list[dict]) -> list[dict]:
    """Keep notebook responses large enough to be useful but safe for the UI."""
    limited: list[dict] = []
    for output in outputs[:50]:
        item = dict(output)
        if isinstance(item.get("text"), str):
            item["text"] = _trim_text(item["text"], 100_000)
        if isinstance(item.get("traceback"), list):
            item["traceback"] = [
                _trim_text(str(line), 20_000)
                for line in item["traceback"][-40:]
            ]
        data = item.get("data")
        if isinstance(data, dict):
            clean_data = {}
            for key, value in data.items():
                if isinstance(value, str):
                    max_chars = 2_000_000 if key.startswith("image/") else 250_000
                    clean_data[key] = _trim_text(value, max_chars)
                else:
                    clean_data[key] = value
            item["data"] = clean_data
        limited.append(item)
    if len(outputs) > 50:
        limited.append({
            "output_type": "stream",
            "name": "stderr",
            "text": f"...[{len(outputs) - 50} additional outputs truncated]",
        })
    return limited


@router.get("/kernel/{session_id}/status")
async def kernel_status(session_id: str):
    """Return whether a session kernel is currently alive."""
    try:
        get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return {"status": "connected" if is_kernel_alive(session_id) else "disconnected"}


@router.post("/kernel/{session_id}/execute")
async def execute_cell(session_id: str, body: ExecuteCellBody):
    """Execute a single code cell in the session kernel."""
    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    if not body.code.strip():
        return {"outputs": [], "error": None}

    dataset_path = _first_uploaded_file(session_dir)
    if dataset_path is not None:
        _, preload_error = execute_code(
            session_id,
            _dataset_preload_code(dataset_path),
            timeout=min(body.timeout, 30),
        )
        if preload_error:
            return {
                "outputs": [{
                    "output_type": "error",
                    "ename": "DatasetLoadError",
                    "evalue": preload_error,
                    "traceback": [preload_error],
                }],
                "error": preload_error,
            }

    outputs, error = execute_code(
        session_id,
        body.code,
        timeout=body.timeout,
        cell_id=body.cell_id,
    )
    if error and error.startswith("Execution timed out"):
        shutdown_kernel(session_id)

    return {"outputs": _limit_outputs(outputs), "error": error}
