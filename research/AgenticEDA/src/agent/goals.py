"""EDA goal checklist — the agent works through these in order."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.agent.state import AgentState


@dataclass
class EDAGoal:
    """A single EDA goal the agent should accomplish."""
    name: str
    phase: str
    description: str
    skip_condition: Callable[[AgentState], bool] | None = None

    def should_skip(self, state: AgentState) -> bool:
        if self.skip_condition:
            return self.skip_condition(state)
        return False


def build_goal_checklist() -> list[EDAGoal]:
    """Build the ordered EDA goal checklist."""
    return [
        # Phase 1: Data Loading & Inspection
        EDAGoal(
            name="load_dataset",
            phase="Data Loading",
            description="Load the dataset and display first rows",
        ),
        EDAGoal(
            name="inspect_dtypes",
            phase="Data Loading",
            description="Show column types and shape",
        ),
        EDAGoal(
            name="inspect_describe",
            phase="Data Loading",
            description="Show statistical summary",
        ),

        # Phase 2: Data Cleaning
        EDAGoal(
            name="parse_datetime",
            phase="Data Cleaning",
            description="Parse datetime columns",
            # Only skip once we have a canonical, quality-checked time axis.
            skip_condition=lambda s: s.time_col == "__agenticeda_time",
        ),
        EDAGoal(
            name="check_missing",
            phase="Data Cleaning",
            description="Audit missing values",
        ),
        EDAGoal(
            name="handle_missing",
            phase="Data Cleaning",
            description="Handle missing values",
            skip_condition=lambda s: not any(s.dtypes),  # no data loaded yet
        ),

        # Phase 3: Univariate Analysis
        EDAGoal(
            name="distributions",
            phase="Univariate Analysis",
            description="Plot distributions of numeric columns",
            skip_condition=lambda s: len(s.numeric_cols) == 0,
        ),

        # Phase 4: Time Series Visualization
        EDAGoal(
            name="time_series_plot",
            phase="Time Series",
            description="Plot time series of key variables",
            skip_condition=lambda s: s.time_col is None,
        ),
        EDAGoal(
            name="seasonality",
            phase="Time Series",
            description="Check for seasonal patterns",
            skip_condition=lambda s: s.time_col is None or len(s.numeric_cols) == 0,
        ),

        # Phase 5: Dynamics
        EDAGoal(
            name="rolling_stats",
            phase="Dynamics",
            description="Compute rolling statistics",
            skip_condition=lambda s: s.time_col is None or len(s.numeric_cols) == 0,
        ),
        EDAGoal(
            name="outlier_detection",
            phase="Dynamics",
            description="Detect outliers",
            skip_condition=lambda s: len(s.numeric_cols) == 0,
        ),

        # Phase 6: Multivariate
        EDAGoal(
            name="correlations",
            phase="Correlations",
            description="Compute correlation matrix",
            skip_condition=lambda s: len(s.numeric_cols) < 2,
        ),

        # Phase 9: Train/Test Split
        EDAGoal(
            name="train_test_split",
            phase="Train/Test Split",
            description="Split data chronologically",
            skip_condition=lambda s: s.time_col is None,
        ),

        # Phase 11: Summary
        EDAGoal(
            name="summary",
            phase="Summary",
            description="Summarize all findings",
        ),
    ]
