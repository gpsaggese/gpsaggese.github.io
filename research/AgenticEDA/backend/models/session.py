"""Pydantic models for session-related request and response schemas."""
from __future__ import annotations

from pydantic import BaseModel


class DatasetPreview(BaseModel):
    """Preview of the uploaded dataset."""

    columns: list[str]
    dtypes: dict[str, str]
    first_5_rows: list[dict]
    row_count: int
    col_count: int


class UploadResponse(BaseModel):
    """Response returned after a successful file upload."""

    session_id: str
    dataset_preview: DatasetPreview
    source_format: str
    load_warnings: list[str]


class SessionInfo(BaseModel):
    """Summary metadata for a single session."""

    session_id: str
    original_filename: str
    source_format: str
    created_at: str
    row_count: int
    col_count: int


class SessionList(BaseModel):
    """List of all known sessions."""

    sessions: list[SessionInfo]
