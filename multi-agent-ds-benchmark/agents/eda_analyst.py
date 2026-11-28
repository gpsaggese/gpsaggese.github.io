from graph.state import AgentState

def eda_analyst_node(state: AgentState) -> AgentState:
    print("\n[EDA Analyst] Placeholder — coming soon.")
    state["current_node"] = "eda_analyst"
    state["eda_summary"] = "Placeholder EDA summary."
    state["cleaned_data_path"] = state["dataset_path"]
    state["data_issues"] = []
    return state
