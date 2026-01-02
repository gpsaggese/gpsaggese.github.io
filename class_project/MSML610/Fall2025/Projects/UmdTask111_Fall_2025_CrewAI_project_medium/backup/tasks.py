"""
Task definitions for NBA data analysis workflow.
"""
from crewai import Task
from config import NBA_DATA_PATH


def create_data_engineering_task(engineer_agent) -> Task:
    """
    Create the data engineering task for processing and cleaning NBA data.
    
    Args:
        engineer_agent: The Engineer Agent to assign this task to
        
    Returns:
        Task: Configured data engineering task
    """
    return Task(
        description=f"""
        Analyze the NBA 2024-25 dataset located at {NBA_DATA_PATH}. 
        
        Your tasks:
        1. Load and examine the dataset structure
        2. Identify any data quality issues (missing values, inconsistencies, outliers)
        3. Clean and preprocess the data
        4. Create a summary of the dataset including:
           - Total number of records
           - Date range of the data
           - Number of unique players
           - Number of unique teams
           - Key statistics columns available
        5. Prepare the data for analysis by ensuring proper data types and formats
        
        Provide a comprehensive data quality report and cleaned dataset summary.
        """,
        agent=engineer_agent,
        expected_output="A detailed data quality report including dataset summary, data issues found, cleaning steps performed, and recommendations for analysis."
    )


def create_data_analysis_task(analyst_agent, data_engineering_task: Task) -> Task:
    """
    Create the data analysis task for extracting insights from NBA data.
    
    Args:
        analyst_agent: The Analyst Agent to assign this task to
        data_engineering_task: The data engineering task for context
        
    Returns:
        Task: Configured data analysis task
    """
    return Task(
        description=f"""
        Using the cleaned NBA 2024-25 dataset, perform comprehensive analysis:
        
        Your tasks:
        1. Analyze player performance metrics:
           - Top performers by points, assists, rebounds
           - Shooting efficiency analysis (FG%, 3P%, FT%)
           - Player efficiency ratings
        2. Team performance analysis:
           - Win/loss records by team
           - Team offensive and defensive statistics
           - Team performance trends
        3. Game insights:
           - High-scoring games
           - Close games vs blowouts
           - Performance by date/period
        4. Identify key patterns and trends:
           - Best performing players
           - Most efficient teams
           - Statistical outliers
        5. Provide actionable insights and recommendations
        
        Create a comprehensive analysis report with key findings and insights.
        """,
        agent=analyst_agent,
        expected_output="A detailed analysis report with key insights, statistical findings, top performers, team analysis, and actionable recommendations based on the NBA 2024-25 data.",
        context=[data_engineering_task]
    )