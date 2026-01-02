"""
Agent definitions for NBA data analysis.
"""
from crewai import Agent
from config import get_llm, NBA_DATA_PATH
from tools import get_agent_tools

# Get LLM and tools
llm = get_llm()
agent_tools = get_agent_tools(NBA_DATA_PATH)


def create_engineer_agent() -> Agent:
    """
    Create the Engineer Agent for data processing and engineering tasks.
    
    Returns:
        Agent: Configured Engineer Agent
    """
    return Agent(
        role="Data Engineer",
        goal="Process, clean, and prepare NBA 2024-25 season data for analysis. Ensure data quality and create structured datasets.",
        backstory="""You are an expert data engineer with years of experience in sports analytics. 
        You specialize in processing large datasets, handling missing values, data validation, 
        and creating clean, analysis-ready datasets. You understand NBA statistics deeply and 
        know how to structure data for optimal analysis.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=agent_tools,
    )


def create_analyst_agent() -> Agent:
    """
    Create the Analyst Agent for data analysis and insights.
    
    Returns:
        Agent: Configured Analyst Agent
    """
    return Agent(
        role="Data Analyst",
        goal="Analyze NBA 2024-25 season data to extract meaningful insights, identify patterns, and provide actionable recommendations.",
        backstory="""You are a seasoned data analyst with a passion for basketball analytics. 
        You excel at finding patterns in data, identifying trends, performing statistical analysis, 
        and translating complex data into clear, actionable insights. You understand player performance 
        metrics, team dynamics, and can provide strategic recommendations based on data.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=agent_tools,
    )