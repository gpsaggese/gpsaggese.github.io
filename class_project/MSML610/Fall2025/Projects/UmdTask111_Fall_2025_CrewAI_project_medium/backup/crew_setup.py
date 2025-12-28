"""
Crew setup for NBA data analysis workflow.
"""
from crewai import Crew, Process
from agents import create_engineer_agent, create_analyst_agent
from tasks import create_data_engineering_task, create_data_analysis_task


def create_crew() -> Crew:
    """
    Create and configure the CrewAI crew with agents and tasks.
    
    Returns:
        Crew: Configured CrewAI crew ready for execution
    """
    # Create agents
    engineer_agent = create_engineer_agent()
    analyst_agent = create_analyst_agent()
    
    # Create tasks
    data_engineering_task = create_data_engineering_task(engineer_agent)
    data_analysis_task = create_data_analysis_task(analyst_agent, data_engineering_task)
    
    # Create and return the crew
    return Crew(
        agents=[engineer_agent, analyst_agent],
        tasks=[data_engineering_task, data_analysis_task],
        process=Process.sequential,
        verbose=True,
    )