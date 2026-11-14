"""
Create and run a simple sequential CrewAI workflow.

The first agent prepares technical notes about a topic. The second agent uses
those notes to write a short beginner-friendly explanation.
"""

import argparse
import os
from typing import Optional

import crewai_utils as crut


def build_topic_crew(
    *,
    llm: Optional[str] = None,
    verbose: bool = False,
):
    """
    Build a two-agent sequential CrewAI workflow.

    :param llm: Model used by both agents.
    :param verbose: Whether to display detailed CrewAI logs.
    :return: Configured CrewAI crew.
    """
    researcher = crut.create_agent(
        role="Technical Researcher",
        goal="Identify the most important ideas about {topic}",
        backstory=(
            "You investigate technical subjects and organize the main ideas "
            "into accurate, concise notes."
        ),
        llm=llm,
        verbose=verbose,
    )

    writer = crut.create_agent(
        role="Technical Writer",
        goal="Explain {topic} clearly to a beginner",
        backstory=(
            "You turn technical notes into structured explanations that are "
            "easy for students to understand."
        ),
        llm=llm,
        verbose=verbose,
    )

    research_task = crut.create_task(
        description=(
            "Prepare concise notes about {topic}. Include its purpose, core "
            "concepts, and one practical use case."
        ),
        expected_output=(
            "A structured list of technical notes with the purpose, core "
            "concepts, and one practical use case."
        ),
        agent=researcher,
    )

    writing_task = crut.create_task(
        description=(
            "Use the research notes to write a beginner-friendly explanation "
            "of {topic}. Keep the response under 400 words."
        ),
        expected_output=(
            "A short Markdown explanation with a title, overview, key "
            "concepts, and practical example."
        ),
        agent=writer,
        context=[research_task],
    )

    return crut.create_sequential_crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=verbose,
    )


def run_example(
    topic: str,
    *,
    llm: Optional[str] = None,
    verbose: bool = False,
):
    """
    Run the topic-explanation crew.

    :param topic: Subject that the crew should explain.
    :param llm: Model used by both agents.
    :param verbose: Whether to display detailed CrewAI logs.
    :return: Crew execution result.
    """
    crew = build_topic_crew(llm=llm, verbose=verbose)
    return crut.run_crew(crew, inputs={"topic": topic})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-agent CrewAI topic-explanation workflow."
    )
    parser.add_argument(
        "--topic",
        default="multi-agent systems",
        help="Topic that the crew should explain.",
    )
    parser.add_argument(
        "--llm",
        default=os.getenv("CREWAI_MODEL"),
        help="CrewAI model identifier, or CREWAI_MODEL from the environment.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display detailed CrewAI execution logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_example(
        args.topic,
        llm=args.llm,
        verbose=args.verbose,
    )
    print(result)


if __name__ == "__main__":
    main()