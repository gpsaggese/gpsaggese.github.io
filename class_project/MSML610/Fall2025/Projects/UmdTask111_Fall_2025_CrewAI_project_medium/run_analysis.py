#!/usr/bin/env python3
"""
Run NBA Data Analysis with CrewAI
"""
import os
from crewai_utils import create_crew, create_flow_crew, create_analyst_only_crew, NBA_DATA_PATH

def main():
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        print("\nPlease set it using:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr set it in this script:")
        print("  os.environ['OPENAI_API_KEY'] = 'your-api-key-here'")
        return
    
    print("=" * 70)
    print("NBA Data Analysis with CrewAI")
    print("=" * 70)
    print()
    
    # Option 1: Basic analysis (Engineer + Analyst)
    print("Option 1: Running basic analysis...")
    print("-" * 70)
    crew = create_crew()
    result = crew.kickoff()
    
    print("\n" + "=" * 70)
    print("BASIC ANALYSIS RESULTS")
    print("=" * 70)
    print(result)
    print()
    
    # Option 2: Custom query with all agents
    print("\n" + "=" * 70)
    print("Option 2: Custom query analysis...")
    print("-" * 70)
    user_query = "Who are the top 5 three-point shooters?"
    print(f"Query: {user_query}")
    
    crew = create_flow_crew(user_query, NBA_DATA_PATH)
    result = crew.kickoff()
    
    print("\n" + "=" * 70)
    print("CUSTOM QUERY RESULTS")
    print("=" * 70)
    if hasattr(result, 'tasks_output') and result.tasks_output:
        print("\nEngineer Output:")
        print(result.tasks_output[0] if len(result.tasks_output) > 0 else "N/A")
        print("\nAnalyst Output:")
        print(result.tasks_output[1] if len(result.tasks_output) > 1 else "N/A")
        print("\nStoryteller Output:")
        print(result.tasks_output[2] if len(result.tasks_output) > 2 else "N/A")
    else:
        print(result)

if __name__ == "__main__":
    main()
