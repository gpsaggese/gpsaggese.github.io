"""
Main entry point for NBA 2024-25 data analysis using CrewAI.
"""
import os
import pandas as pd
from config import NBA_DATA_PATH
from crew_setup import create_crew


def main():
    """Main function to run the NBA data analysis crew."""
    print("=" * 60)
    print("NBA 2024-25 Data Analysis with CrewAI")
    print("Using Ollama Llama3")
    print("=" * 60)
    print()
    
    # Check if data file exists
    if not os.path.exists(NBA_DATA_PATH):
        print(f"Error: {NBA_DATA_PATH} not found!")
        return
    
    print(f"Loading data from {NBA_DATA_PATH}...")
    try:
        # Quick data preview
        df = pd.read_csv(NBA_DATA_PATH)
        print(f"Dataset loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        print()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Starting CrewAI agents...")
    print("Engineer Agent will process and clean the data...")
    print("Analyst Agent will analyze the data for insights...")
    print()
    print("-" * 60)
    print()
    
    # Create and execute the crew
    crew = create_crew()
    result = crew.kickoff()
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print(result)


if __name__ == "__main__":
    main()