import pandas as pd
from graph.state import AgentState
from config import get_llm

def planner_node(state: AgentState) -> AgentState:
    """
    Node 1: Planner
    - Reads the dataset
    - Summarizes schema and shape
    - Decides problem type and target column
    - Writes a strategy plan for downstream agents
    """
    print("\n[Planner] Starting analysis...")
    state["current_node"] = "planner"

    # --- Load dataset and build summary ---
    df = pd.read_csv(state["dataset_path"])
    
    summary = f"""
    Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns
    Columns: {list(df.columns)}
    Dtypes:
    {df.dtypes.to_string()}
    
    Null counts:
    {df.isnull().sum().to_string()}
    
    Sample rows:
    {df.head(3).to_string()}
    """
    state["dataset_summary"] = summary

    # --- Ask the LLM to create a plan ---
    llm = get_llm(temperature=0.1)

    prompt = f"""
You are a senior data scientist. You have been given the following dataset summary:

{summary}

Based on this, answer the following:
1. What type of ML problem is this? (e.g. binary classification, regression, NLP sentiment)
2. Which column should be the target variable?
3. What are the top 3 data quality concerns you see?
4. Write a concise step-by-step analysis plan for an EDA agent and an ML agent to follow.

Be specific and direct. No generic advice.
"""

    response = llm.invoke(prompt)
    plan = response.content

    # --- Parse problem type and target column from the response ---
    problem_type = "classification"
    target_column = "rating"

    if "regression" in plan.lower():
        problem_type = "regression"
    if "sentiment" in plan.lower():
        problem_type = "sentiment classification"

    state["plan"] = plan
    state["problem_type"] = problem_type
    state["target_column"] = target_column

    print(f"[Planner] Problem type: {problem_type}")
    print(f"[Planner] Target column: {target_column}")
    print(f"[Planner] Plan generated ({len(plan)} chars)")

    return state
