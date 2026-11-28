from langgraph.graph import StateGraph, END
from graph.state import AgentState

from agents.planner import planner_node
from agents.eda_analyst import eda_analyst_node
from agents.ml_architect import ml_architect_node
from agents.quality_reviewer import quality_reviewer_node
from agents.report_writer import report_writer_node

# --- Conditional edge logic ---
def should_retry_or_finish(state: AgentState) -> str:
    """
    After the Quality Reviewer runs, decide:
    - If review failed AND attempts < 3 → loop back to ML Architect
    - If review passed OR attempts >= 3 → move to Report Writer
    """
    attempts = state.get("review_attempts", 0)
    passed = state.get("review_passed", False)

    if passed:
        print(f"[Reviewer] PASSED after {attempts} attempt(s). Moving to Report Writer.")
        return "report_writer"
    elif attempts >= 3:
        print(f"[Reviewer] Max attempts ({attempts}) reached. Moving to Report Writer anyway.")
        return "report_writer"
    else:
        print(f"[Reviewer] FAILED (attempt {attempts}). Looping back to ML Architect.")
        return "ml_architect"


def build_graph() -> StateGraph:
    """Builds and compiles the full multi-agent LangGraph pipeline."""

    graph = StateGraph(AgentState)

    # --- Register all nodes ---
    graph.add_node("planner",          planner_node)
    graph.add_node("eda_analyst",      eda_analyst_node)
    graph.add_node("ml_architect",     ml_architect_node)
    graph.add_node("quality_reviewer", quality_reviewer_node)
    graph.add_node("report_writer",    report_writer_node)

    # --- Define linear edges ---
    graph.set_entry_point("planner")
    graph.add_edge("planner",      "eda_analyst")
    graph.add_edge("eda_analyst",  "ml_architect")
    graph.add_edge("ml_architect", "quality_reviewer")

    # --- Conditional edge: Reviewer → loop or finish ---
    graph.add_conditional_edges(
        "quality_reviewer",
        should_retry_or_finish,
        {
            "ml_architect": "ml_architect",
            "report_writer": "report_writer",
        }
    )

    graph.add_edge("report_writer", END)

    return graph.compile()
