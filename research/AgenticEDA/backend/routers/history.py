"""FastAPI router for version history endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models.history import VersionList, RestoreResponse
from backend.services.history_service import list_versions, restore_version

router = APIRouter(tags=["history"])


@router.get("/history/{session_id}", response_model=VersionList)
async def get_history(session_id: str):
    """Return all version snapshots for a session."""
    versions = list_versions(session_id)
    return VersionList(versions=versions)


@router.post("/history/{session_id}/restore/{version_id}", response_model=RestoreResponse)
async def restore(session_id: str, version_id: int):
    """Restore a previous version snapshot."""
    try:
        result = restore_version(session_id, version_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return RestoreResponse(**result)
