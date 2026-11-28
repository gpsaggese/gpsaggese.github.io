from graph.workflow import build_graph

# --- Initial state ---
initial_state = {
    "dataset_path": "data/amazon_beauty_50k.csv",
    "dataset_summary": None,
    "problem_type": None,
    "target_column": None,
    "plan": None,
    "eda_summary": None,
    "cleaned_data_path": None,
    "data_issues": None,
    "model_name": None,
    "metrics": None,
    "model_path": None,
    "review_passed": None,
    "review_feedback": None,
    "review_attempts": 0,
    "final_report": None,
    "errors": [],
    "current_node": None,
}

print("=" * 50)
print("  Multi-Agent DS Benchmark — Starting Run")
print("=" * 50)

graph = build_graph()
final_state = graph.invoke(initial_state)

print("\n" + "=" * 50)
print("  Run Complete!")
print("=" * 50)
print(f"\n  Problem type : {final_state['problem_type']}")
print(f"  Target column: {final_state['target_column']}")
print(f"\n  Planner Plan:\n{final_state['plan']}")
