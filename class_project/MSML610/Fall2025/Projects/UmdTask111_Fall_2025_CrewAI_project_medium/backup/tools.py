"""
Tools for CrewAI agents to interact with NBA data.
"""
import pandas as pd
from crewai.tools import tool
from typing import Optional


def get_agent_tools(data_path: str):
    """
    Get the list of tools available for agents.
    
    Args:
        data_path: Path to the CSV data file
        
    Returns:
        list: List of tools for agents to use
    """
    
    def _read_nba_data() -> str:
        """Read the NBA data file and return its contents."""
        try:
            # Read only first 1000 lines to avoid overwhelming the context
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:1000]
                return ''.join(lines) + f"\n\n[Note: Showing first 1000 lines of {data_path}. Total file may be larger.]"
        except Exception as e:
            return f"Error reading file {data_path}: {str(e)}"
    
    def _search_nba_data(
        query: Optional[str] = None,
        column: Optional[str] = None,
        value: Optional[str] = None,
        limit: int = 100
    ) -> str:
        """Search and filter NBA data CSV file."""
        try:
            df = pd.read_csv(data_path)
            
            if column and value:
                if column in df.columns:
                    df = df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
                else:
                    return f"Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}"
            
            if query:
                # Search across all string columns
                mask = pd.Series([False] * len(df))
                for col in df.columns:
                    if df[col].dtype == 'object':
                        mask |= df[col].astype(str).str.contains(query, case=False, na=False)
                df = df[mask]
            
            # Limit results
            df = df.head(limit)
            
            if len(df) == 0:
                return "No matching records found."
            
            return f"Found {len(df)} matching records:\n\n{df.to_string()}"
        except Exception as e:
            return f"Error searching CSV {data_path}: {str(e)}"
    
    def _get_nba_data_summary() -> str:
        """Get a comprehensive summary of the NBA data file."""
        try:
            df = pd.read_csv(data_path)
            
            # Calculate basic stats
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            summary = f"""
=== NBA Data Summary for {data_path} ===

DATASET OVERVIEW:
- Total Rows: {len(df):,}
- Total Columns: {len(df.columns)}
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

COLUMN INFORMATION:
{', '.join(df.columns.tolist())}

DATA TYPES:
{df.dtypes.to_string()}

UNIQUE VALUES:
- Unique Players: {df['Player'].nunique() if 'Player' in df.columns else 'N/A'}
- Unique Teams: {df['Tm'].nunique() if 'Tm' in df.columns else 'N/A'}
- Date Range: {df['Data'].min() if 'Data' in df.columns else 'N/A'} to {df['Data'].max() if 'Data' in df.columns else 'N/A'}

MISSING VALUES:
{df.isnull().sum().to_string()}

FIRST 5 ROWS:
{df.head().to_string()}

BASIC STATISTICS (Numeric Columns):
{df[numeric_cols].describe().to_string() if numeric_cols else 'No numeric columns found'}
"""
            return summary
        except Exception as e:
            return f"Error getting CSV summary for {data_path}: {str(e)}"
    
    @tool("read_nba_data")
    def read_nba_data() -> str:
        """Read the NBA data file and return its contents. Use this to examine the raw data structure."""
        return _read_nba_data()
    
    @tool("search_nba_data")
    def search_nba_data(
        query: Optional[str] = None,
        column: Optional[str] = None,
        value: Optional[str] = None,
        limit: int = 100
    ) -> str:
        """
        Search and filter NBA data CSV file. Use this to find specific players, teams, or statistics.
        
        Args:
            query: Optional text query to search for in any column (e.g., player name, team name)
            column: Optional column name to filter by (e.g., 'Player', 'Tm', 'PTS')
            value: Optional value to match in the specified column
            limit: Maximum number of rows to return (default: 100)
        """
        return _search_nba_data(query, column, value, limit)
    
    @tool("get_nba_data_summary")
    def get_nba_data_summary() -> str:
        """
        Get a comprehensive summary of the NBA data file including structure, basic statistics, 
        and data quality information. Use this first to understand the dataset.
        """
        return _get_nba_data_summary()
    
    return [read_nba_data, search_nba_data, get_nba_data_summary]