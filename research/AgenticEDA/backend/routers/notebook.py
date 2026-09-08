"""Notebook router: read, update, and confirm notebooks."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from typing import Any

import nbformat
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.session_manager import get_session_dir

router = APIRouter(tags=["notebook"])


class NotebookPatchBody(BaseModel):
    """Body for PATCH /notebook/{session_id}."""

    cells: list[dict]


FRONTEND_ONLY_CELL_FIELDS = {"executing", "error", "thinking", "baselineOutputs"}


def _normalize_source(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value)


def _trim_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n...[truncated after {max_chars} characters]"


def _normalize_mime_bundle(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"text/plain": str(value)}

    normalized: dict[str, Any] = {}
    for key, raw in value.items():
        mime = str(key)
        if isinstance(raw, str):
            max_chars = 2_000_000 if mime.startswith("image/") else 250_000
            normalized[mime] = _trim_text(raw, max_chars)
        elif isinstance(raw, list):
            normalized[mime] = [_trim_text(str(part), 50_000) for part in raw[:200]]
        elif raw is None:
            normalized[mime] = ""
        else:
            normalized[mime] = raw
    return normalized


def _normalize_outputs(outputs: Any, cell_execution_count: int | None) -> list[dict[str, Any]]:
    if not isinstance(outputs, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw_output in outputs[:50]:
        if not isinstance(raw_output, dict):
            normalized.append(nbformat.v4.new_output("stream", name="stdout", text=str(raw_output)))
            continue

        output_type = str(raw_output.get("output_type") or "stream")
        if output_type == "stream":
            name = raw_output.get("name")
            if name not in {"stdout", "stderr"}:
                name = "stdout"
            text = raw_output.get("text", "")
            normalized.append(
                nbformat.v4.new_output(
                    "stream",
                    name=name,
                    text=_trim_text(_normalize_source(text), 100_000),
                )
            )
            continue

        if output_type == "error":
            traceback = raw_output.get("traceback", [])
            if not isinstance(traceback, list):
                traceback = [_normalize_source(traceback)]
            normalized.append(
                nbformat.v4.new_output(
                    "error",
                    ename=_normalize_source(raw_output.get("ename") or "Error"),
                    evalue=_normalize_source(raw_output.get("evalue") or ""),
                    traceback=[_trim_text(_normalize_source(line), 20_000) for line in traceback[-40:]],
                )
            )
            continue

        if output_type in {"display_data", "execute_result"}:
            data = _normalize_mime_bundle(raw_output.get("data", {}))
            metadata = raw_output.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            kwargs: dict[str, Any] = {
                "data": data,
                "metadata": metadata,
            }
            if output_type == "execute_result":
                output_execution_count = raw_output.get("execution_count")
                kwargs["execution_count"] = (
                    output_execution_count
                    if isinstance(output_execution_count, int)
                    else cell_execution_count
                )
            normalized.append(nbformat.v4.new_output(output_type, **kwargs))
            continue

        text = raw_output.get("text") or raw_output.get("data") or raw_output
        normalized.append(
            nbformat.v4.new_output(
                "stream",
                name="stdout",
                text=_trim_text(_normalize_source(text), 100_000),
            )
        )

    if len(outputs) > 50:
        normalized.append(
            nbformat.v4.new_output(
                "stream",
                name="stderr",
                text=f"...[{len(outputs) - 50} additional outputs truncated]",
            )
        )
    return normalized


def _existing_cells_by_id(nb_path) -> tuple[dict[str, dict], dict]:
    if not nb_path.exists():
        return {}, {}
    try:
        existing_nb = nbformat.read(str(nb_path), as_version=4)
    except Exception:
        return {}, {}

    cells_by_id = {}
    for cell in existing_nb.cells:
        cell_id = cell.get("id")
        if cell_id:
            cells_by_id[str(cell_id)] = cell
    metadata = existing_nb.get("metadata", {})
    return cells_by_id, metadata if isinstance(metadata, dict) else {}


def _cell_metadata(cell_data: dict, existing_cell: dict | None) -> dict:
    metadata = cell_data.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    if existing_cell is not None and isinstance(existing_cell.get("metadata"), dict):
        return dict(existing_cell.get("metadata") or {})
    return {}


def _write_notebook_atomic(nb_path, nb) -> None:
    fd, tmp = tempfile.mkstemp(dir=nb_path.parent, suffix=".ipynb.tmp")
    os.close(fd)
    try:
        nbformat.write(nb, tmp)
        os.replace(tmp, nb_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


@router.get("/notebook/{session_id}")
async def get_notebook(session_id: str):
    """Return the notebook.ipynb from the session directory as JSON."""
    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    nb_path = session_dir / "notebook.ipynb"
    if not nb_path.exists():
        raise HTTPException(status_code=404, detail="Notebook not found for this session")

    nb = nbformat.read(str(nb_path), as_version=4)
    return json.loads(nbformat.writes(nb))


@router.patch("/notebook/{session_id}")
async def patch_notebook(session_id: str, body: NotebookPatchBody):
    """Overwrite notebook.ipynb with the provided full cell list."""
    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    nb_path = session_dir / "notebook.ipynb"
    existing_cells, notebook_metadata = _existing_cells_by_id(nb_path)

    nb = nbformat.v4.new_notebook(metadata=notebook_metadata)
    for cell_data in body.cells:
        if not isinstance(cell_data, dict):
            continue

        cell_type = cell_data.get("cell_type", "code")
        source = _normalize_source(cell_data.get("source", ""))
        cell_id = cell_data.get("id")
        if not isinstance(cell_id, str) or not cell_id.strip():
            cell_id = str(uuid.uuid4())
        existing_cell = existing_cells.get(cell_id)
        metadata = _cell_metadata(cell_data, existing_cell)

        if cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source, id=cell_id, metadata=metadata)
        else:
            execution_count = cell_data.get("execution_count")
            if not isinstance(execution_count, int):
                execution_count = None
            outputs = _normalize_outputs(cell_data.get("outputs", []), execution_count)
            cell = nbformat.v4.new_code_cell(
                source,
                id=cell_id,
                metadata=metadata,
                execution_count=execution_count,
                outputs=outputs,
            )

        for field in FRONTEND_ONLY_CELL_FIELDS:
            cell.pop(field, None)
        nb.cells.append(cell)

    try:
        nbformat.validate(nb)
        _write_notebook_atomic(nb_path, nb)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid notebook payload: {exc}") from exc

    return {"status": "updated", "cell_count": len(nb.cells)}


@router.post("/notebook/{session_id}/confirm")
async def confirm_notebook(session_id: str):
    """Placeholder: confirm the current notebook version."""
    try:
        get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    return {"status": "confirmed", "version_id": 1}
