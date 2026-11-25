from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    """
    Shared state object passed between all nodes in the LangGraph pipeline.
    Each agent reads from and writes to this state.
    """

    # --- Input ---
    dataset_path: str                  # Path to the loaded CSV file
    dataset_summary: Optional[str]     # Quick schema/shape summary for the Planner

    # --- Planner outputs ---
    problem_type: Optional[str]        # e.g. "binary classification"
    target_column: Optional[str]       # e.g. "rating"
    plan: Optional[str]                # Full strategy written by the Planner

    # --- EDA Analyst outputs ---
    eda_summary: Optional[str]         # Key findings from exploratory analysis
    cleaned_data_path: Optional[str]   # Path to cleaned CSV after EDA
    data_issues: Optional[List[str]]   # e.g. ["nulls in text column", "class imbalance"]

    # --- ML Architect outputs ---
    model_name: Optional[str]          # e.g. "XGBoost"
    metrics: Optional[dict]            # e.g. {"accuracy": 0.91, "recall": 0.73, "f1": 0.81}
    model_path: Optional[str]          # Path to saved model file

    # --- Quality Reviewer outputs ---
    review_passed: Optional[bool]      # True = move forward, False = loop back
    review_feedback: Optional[str]     # Feedback sent back to ML Architect if failed
    review_attempts: Optional[int]     # Tracks how many review loops have happened

    # --- Report Writer outputs ---
    final_report: Optional[str]        # Full markdown report of the entire run

    # --- Global ---
    errors: Optional[List[str]]        # Any errors caught during the run
    current_node: Optional[str]        # Tracks which node is currently executing
