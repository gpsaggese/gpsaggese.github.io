"""
CrewAI custom tools for the FRED macro briefing project.
"""

import json
from pathlib import Path
from typing import Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import fred_tools


class FREDResearchInput(BaseModel):
    """Input schema for downloading and profiling FRED data."""

    data_dir: str = Field(
        default="data/fred",
        description="Root directory in which dated FRED snapshots are saved.",
    )
    download_date: Optional[str] = Field(
        default=None,
        description=(
            "Optional ISO download date. If omitted, today's date is used."
        ),
    )


class FREDResearchTool(BaseTool):
    """Download and profile the four required FRED series."""

    name: str = "download_and_profile_fred_data"
    description: str = (
        "Download CPI, unemployment, federal funds rate, and Treasury "
        "spread data from FRED. Save a dated CSV snapshot and return the "
        "latest value, prior-year value, dates, and missing-value count."
    )
    args_schema: type[BaseModel] = FREDResearchInput

    def _run(
        self,
        data_dir: str = "data/fred",
        download_date: Optional[str] = None,
    ) -> str:
        paths = fred_tools.download_all_fred_series(
            data_dir=data_dir,
            download_date=download_date,
        )
        snapshot_dir = next(iter(paths.values())).parent
        profiles = fred_tools.profile_fred_snapshot(snapshot_dir)

        result = {
            "snapshot_dir": str(snapshot_dir),
            "download_date": snapshot_dir.name,
            "series": profiles,
        }
        return json.dumps(result, indent=2)


class FREDAnalysisInput(BaseModel):
    """Input schema for calculating macroeconomic statistics."""

    snapshot_dir: str = Field(
        ...,
        description=(
            "Directory containing CPIAUCSL.csv, UNRATE.csv, "
            "FEDFUNDS.csv, and T10Y2Y.csv."
        ),
    )


class FREDAnalysisTool(BaseTool):
    """Calculate deterministic statistics from a saved FRED snapshot."""

    name: str = "calculate_fred_macro_statistics"
    description: str = (
        "Calculate year-over-year CPI inflation, the one-year change in "
        "unemployment, the current federal funds rate and its one-year "
        "change, and the value and sign of the Treasury yield spread."
    )
    args_schema: type[BaseModel] = FREDAnalysisInput

    def _run(self, snapshot_dir: str) -> str:
        statistics = fred_tools.calculate_macro_statistics(
            Path(snapshot_dir)
        )
        return json.dumps(statistics, indent=2)