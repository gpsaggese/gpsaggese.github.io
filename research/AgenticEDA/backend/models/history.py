"""Pydantic models for version history."""
from __future__ import annotations

from pydantic import BaseModel


class VersionSnapshot(BaseModel):
    version_id: int
    trigger: str
    timestamp: str
    has_notebook: bool = False
    has_story: bool = False


class VersionList(BaseModel):
    versions: list[VersionSnapshot]


class RestoreResponse(BaseModel):
    restored_version_id: int
    new_version_id: int
