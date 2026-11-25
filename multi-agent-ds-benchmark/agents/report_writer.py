from graph.state import AgentState

def report_writer_node(state: AgentState) -> AgentState:
    print("\n[Report Writer] Placeholder — coming soon.")
    state["current_node"] = "report_writer"
    state["final_report"] = "Placeholder report."
    return state
