"""
Reusable helpers for building CrewAI agents, tasks, and crews.
"""

from typing import Any, Optional

from crewai import Agent, Crew, Process, Task


def create_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: Optional[list[Any]] = None,
    *,
    llm: Optional[str] = None,
    verbose: bool = False,
    allow_delegation: bool = False,
) -> Agent:
    """
    Create and configure a CrewAI agent.

    :param role: Agent's area of responsibility.
    :param goal: Objective the agent should accomplish.
    :param backstory: Context describing the agent's expertise.
    :param tools: Optional tools available to the agent.
    :param llm: Optional model name.
    :param verbose: Whether to display detailed execution logs.
    :param allow_delegation: Whether the agent can delegate work.
    :return: Configured CrewAI agent.
    """
    agent_kwargs: dict[str, Any] = {
        "role": role,
        "goal": goal,
        "backstory": backstory,
        "verbose": verbose,
        "allow_delegation": allow_delegation,
    }
    if tools is not None:
        agent_kwargs["tools"] = tools
    if llm is not None:
        agent_kwargs["llm"] = llm
    return Agent(**agent_kwargs)


def create_task(
    description: str,
    expected_output: str,
    agent: Agent,
    *,
    context: Optional[list[Task]] = None,
) -> Task:
    """
    Create a task assigned to a specific agent.

    :param description: Instructions for completing the task.
    :param expected_output: Description of the required result.
    :param agent: Agent responsible for the task.
    :param context: Earlier tasks whose outputs provide context.
    :return: Configured CrewAI task.
    """
    task_kwargs: dict[str, Any] = {
        "description": description,
        "expected_output": expected_output,
        "agent": agent,
    }
    if context is not None:
        task_kwargs["context"] = context
    return Task(**task_kwargs)


def create_sequential_crew(
    agents: list[Agent],
    tasks: list[Task],
    *,
    verbose: bool = False,
) -> Crew:
    """
    Create a crew that performs tasks sequentially.

    :param agents: Agents participating in the workflow.
    :param tasks: Tasks in their required execution order.
    :param verbose: Whether to display detailed execution logs.
    :return: Configured CrewAI crew.
    """
    if not agents:
        raise ValueError("At least one agent is required.")
    if not tasks:
        raise ValueError("At least one task is required.")

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
    )


def run_crew(
    crew: Crew,
    inputs: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Execute a crew and return its output.

    :param crew: Crew to execute.
    :param inputs: Optional values used to populate task placeholders.
    :return: Crew execution result.
    """
    return crew.kickoff(inputs=inputs or {})