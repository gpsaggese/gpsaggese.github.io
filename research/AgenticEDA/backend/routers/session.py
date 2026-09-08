"""Session router: upload, list, and delete sessions."""
from __future__ import annotations

import tempfile
import pathlib

from fastapi import APIRouter, HTTPException, UploadFile, File

from backend.models.session import (
    DatasetPreview,
    SessionList,
    UploadResponse,
)
from backend.services.session_manager import (
    create_session,
    delete_session,
    list_sessions,
)

router = APIRouter(tags=["sessions"])


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Accept a multipart file upload, parse it, and create a new session."""
    from src.ingest.file_loader import load_file

    # Write the upload to a temporary file so load_file can read it
    suffix = pathlib.Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = pathlib.Path(tmp.name)

    try:
        df, metadata = load_file(tmp_path)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        # Clean up the temp file after loading
        tmp_path.unlink(missing_ok=True)

    # Build the dataset preview
    preview = DatasetPreview(
        columns=df.columns.tolist(),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        first_5_rows=df.head(5).to_dict(orient="records"),
        row_count=metadata.get("row_count", len(df)),
        col_count=metadata.get("col_count", len(df.columns)),
    )

    # We need to re-write the upload for persistence (temp was cleaned up).
    # Write it again into a fresh temp file that create_session will copy.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp2:
        tmp2.write(contents)
        tmp2_path = pathlib.Path(tmp2.name)

    try:
        session_id = create_session(
            file_path=tmp2_path,
            original_filename=file.filename or "upload",
            df=df,
            metadata=metadata,
        )
    finally:
        tmp2_path.unlink(missing_ok=True)

    return UploadResponse(
        session_id=session_id,
        dataset_preview=preview,
        source_format=metadata.get("source_format", "unknown"),
        load_warnings=metadata.get("load_warnings", []),
    )


@router.get("/sessions", response_model=SessionList)
async def get_sessions():
    """List all existing sessions with their metadata."""
    return SessionList(sessions=list_sessions())


@router.delete("/session/{session_id}")
async def remove_session(session_id: str):
    """Delete a session and all associated files."""
    try:
        delete_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return {"status": "deleted", "session_id": session_id}
