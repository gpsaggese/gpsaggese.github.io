"""History service — delegates to src.reporting.versioning."""
from __future__ import annotations

from src.reporting.versioning import create_snapshot, list_versions, restore_version

__all__ = ["create_snapshot", "list_versions", "restore_version"]
