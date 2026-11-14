"""
Run the three-agent FRED macroeconomic briefing crew.
"""

import argparse
import os
from datetime import date
from typing import Optional
from pathlib import Path

import crewai_utils as crut
from crewai_fred_tools import FREDAnalysisTool, FREDResearchTool


def build_fred_crew(
    *,
    llm: Optional[str] = None,
    verbose: bool = False,
):
    """
    Build the Researcher, Analyst, and Writer crew.

    :param llm: Model identifier used by all three agents.
    :param verbose: Whether to display detailed CrewAI logs.
    :return: Configured sequential CrewAI crew.
    """
    research_tool = FREDResearchTool()
    analysis_tool = FREDAnalysisTool()

    researcher = crut.create_agent(
        role="FRED Data Researcher",
        goal=(
            "Download the required FRED series and provide an accurate "
            "profile of the saved data snapshot."
        ),
        backstory=(
            "You are a careful economic data researcher. You always use the "
            "provided FRED tool and preserve observation dates, values, and "
            "the snapshot directory exactly as returned."
        ),
        tools=[research_tool],
        llm=llm,
        verbose=verbose,
    )

    analyst = crut.create_agent(
        role="Macroeconomic Data Analyst",
        goal=(
            "Calculate inflation, labor-market, policy-rate, and yield-curve "
            "statistics from the saved FRED snapshot."
        ),
        backstory=(
            "You are a quantitative analyst who never calculates important "
            "numbers mentally. You use the provided analysis tool and report "
            "its results without changing them."
        ),
        tools=[analysis_tool],
        llm=llm,
        verbose=verbose,
    )

    writer = crut.create_agent(
        role="Macroeconomic Briefing Writer",
        goal=(
            "Write a concise briefing whose numeric claims match the "
            "Researcher and Analyst outputs."
        ),
        backstory=(
            "You write clear economic briefings for non-specialists. You do "
            "not invent, estimate, or alter numbers, and you include the "
            "observation date for every reported statistic."
        ),
        llm=llm,
        verbose=verbose,
    )

    research_task = crut.create_task(
        description=(
            "Use download_and_profile_fred_data with data_dir='{data_dir}' "
            "and download_date='{download_date}'. Return the complete tool "
            "output, including snapshot_dir and the profile for all four "
            "series. Do not omit or modify any values."
        ),
        expected_output=(
            "A structured data profile containing snapshot_dir, download "
            "date, and the latest and prior-year observations for CPIAUCSL, "
            "UNRATE, FEDFUNDS, and T10Y2Y."
        ),
        agent=researcher,
    )

    analysis_task = crut.create_task(
        description=(
            "Read the Researcher output and extract snapshot_dir. Call "
            "calculate_fred_macro_statistics using that exact directory. "
            "Report every statistic returned by the tool without performing "
            "your own arithmetic or changing the precision."
        ),
        expected_output=(
            "The exact inflation rate, unemployment rate and one-year "
            "change, federal funds rate and one-year change, Treasury spread, "
            "yield-curve sign, and their observation dates."
        ),
        agent=analyst,
        context=[research_task],
    )

    writing_task = crut.create_task(
        description=(
            "Using only the Researcher and Analyst outputs, write a concise "
            "macroeconomic briefing covering inflation, labor conditions, "
            "interest rates, and the yield curve. Include observation dates. "
            "Do not introduce any numeric claim that is absent from the "
            "provided outputs. Keep the briefing under 350 words."
        ),
        expected_output=(
            "A Markdown briefing with sections for Inflation, Labor Market, "
            "Interest Rates, and Yield Curve, followed by a short outlook."
        ),
        agent=writer,
        context=[research_task, analysis_task],
    )

    return crut.create_sequential_crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        verbose=verbose,
    )


def run_fred_crew(
    data_dir: str,
    download_date: str,
    *,
    llm: Optional[str] = None,
    verbose: bool = False,
):
    """
    Run the FRED macro briefing crew.

    :param data_dir: Root directory for dated FRED snapshots.
    :param download_date: ISO date assigned to the snapshot.
    :param llm: Model identifier used by the agents.
    :param verbose: Whether to display detailed CrewAI logs.
    :return: Crew execution result.
    """
    crew = build_fred_crew(llm=llm, verbose=verbose)
    return crut.run_crew(
        crew,
        inputs={
            "data_dir": data_dir,
            "download_date": download_date,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three-agent FRED macro briefing crew."
    )
    parser.add_argument(
        "--data-dir",
        default="data/fred",
        help="Root directory for saved FRED snapshots.",
    )
    parser.add_argument(
        "--download-date",
        default=date.today().isoformat(),
        help="ISO date assigned to the downloaded snapshot.",
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
    result = run_fred_crew(
        args.data_dir,
        args.download_date,
        llm=args.llm,
        verbose=args.verbose,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / f"fred_macro_briefing_{args.download_date}.md"
    )
    output_path.write_text(str(result), encoding="utf-8")

    print(result)
    print(f"\nBriefing saved to: {output_path}")


if __name__ == "__main__":
    main()