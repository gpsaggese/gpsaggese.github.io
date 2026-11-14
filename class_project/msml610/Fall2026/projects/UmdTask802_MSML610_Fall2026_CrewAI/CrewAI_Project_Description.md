# Description

CrewAI is a lean Python framework (built from scratch) for orchestrating "crews" of
role-based agents and event-driven "flows". It solves the problem of splitting a
complex task across specialist agents, with low-level control when needed. It is
worth a 60-minute tutorial because a working multi-agent pipeline takes a few dozen
lines, and the same primitives scale from a three-agent brief to parallel workflows.

## Technologies Used

CrewAI

- Role-based agents (researcher, analyst, writer, etc.) with custom tools
- Crews for teamwork; Flows for fine-grained orchestration
- Sequential/parallel tasks with automatic dependency handling
- High-performance execution; prompt and tool customization

# Tutorial

- Implement the tutorial "Learn CrewAI in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Read the `README.md` of the Fall2025 CrewAI project, and reuse what is good
    - `class_project/msml610/Fall2025/projects/UmdTask111_Fall_2025_CrewAI_project_medium/`
  - Read the `README.md` of `tutorials/LangChain/`, `tutorials/LangGraph/`, and
    `tutorials/Autogen/` for the related multi-agent tools
- Start from the existing `tutorials/CrewAI/` and make it better
  - It has no `README.md`: add one
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare with `tutorials/LangChain/`, `tutorials/LangGraph/`, and
  `tutorials/Autogen/` from the multi-agent orchestration point of view, and do the
  same clean up for them if you want
- Deliverables:
  - `crewai_utils.py`
  - `crewai.API.ipynb`
  - `crewai.example.ipynb`

# Project

## Project 1 (Fall2026): FRED Macro Briefing Crew

- **Project Objective**: A 3-agent crew (Researcher, Analyst, Writer) turns four FRED
  macro series into a short briefing on inflation, labor, and interest rates whose
  numbers match the data, and beats a single-LLM baseline on the fraction of correct
  numbers
- **Dataset Suggestions**: Four [FRED](https://fred.stlouisfed.org/) series, downloaded
  as CSV from `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`
  - [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL): Consumer Price Index
  - [UNRATE](https://fred.stlouisfed.org/series/UNRATE): unemployment rate
  - [FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS): federal funds rate
  - [T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y): 10-year minus 2-year
    Treasury spread
  - Save the CSV files with the download date, so that the check uses the same
    snapshot as the crew
- **Tasks**:
  - **Load the Data**: Give the Researcher a custom tool that downloads the four
    series and returns a profile (latest date, latest value, value one year ago,
    missing values)
  - **Define the Crew**: Define the Researcher, Analyst, and Writer as `Agent`
    objects with role, goal, and backstory, and one `Task` each with an expected
    output
  - **Run the Crew**: Run the `Crew` with `Process.sequential` through `kickoff()`,
    with the Analyst calling a custom tool for year-over-year inflation, the change
    in unemployment, and the sign of the yield curve spread
  - **Evaluate the Brief**: Recompute the statistics with pandas from the saved CSV
    files and report the fraction of numeric claims in the brief that match within
    rounding, next to a single-LLM baseline that gets the raw profile in one prompt
    and has no tools
  - **Export the Brief**: Export the write-up as a Markdown file with a summary table
    and a line chart of the four series
- **Bonus Ideas (Optional)**: Add a Skeptic agent that checks every number against
  the tool output, and measure how much it raises the fraction of correct numbers

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Load the Data
  - Result: `tutorials/CrewAI/` container running, and the FRED profile table
    produced by the custom tool
- Milestone 2: API notebook
  - Project tasks: Define the Crew, Run the Crew
  - Result: `crewai.API.ipynb` covering role-based agents with custom tools, and
    sequential and parallel tasks
- Milestone 3: Example notebook
  - Project tasks: Define the Crew, Run the Crew, Evaluate the Brief, Export the
    Brief
  - Result: `crewai.example.ipynb` running end to end and exporting the brief

## Project 2: SEC 10-K Analysis Flow

- **Project Objective**: Agents read the latest 10-K filings of five large companies
  and write a comparison of growth, profitability, and main risks, maximizing the
  fraction of numbers that match the SEC XBRL data against a single-LLM baseline
- **Dataset Suggestions**: [SEC EDGAR APIs](https://www.sec.gov/edgar/sec-api-documentation)
  - XBRL facts per company from
    `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`, e.g., Apple is
    `CIK0000320193`
  - Filing list per company from
    `https://data.sec.gov/submissions/CIK##########.json`, used to find the latest
    10-K document and its Item 1A risk factors
  - SEC requires a descriptive `User-Agent` header with a contact email, see the
    [SEC webmaster FAQ](https://www.sec.gov/os/webmaster-faq)
- **Tasks**:
  - **Fetch the Filings**: Write two custom tools: one returns revenue, net income,
    and total assets of the last two fiscal years from `companyfacts` (selected with
    the `fy`, `fp`, and `form` fields), and one downloads the latest 10-K and
    returns the Item 1A text
  - **Define the Agents**: Define a Filing Analyst, a Risk Analyst, and a Writer as
    `Agent` objects with their tools, and one `Task` each with an expected output
  - **Orchestrate with a Flow**: Run one crew per company in parallel through a
    `Flow` (`@start` and `@listen`) with `async_execution=True` tasks, and merge the
    five briefs into one comparison table
  - **Evaluate Against XBRL**: Extract the numeric claims of each brief, and report
    the fraction that match the XBRL values within rounding, next to a single-LLM
    baseline that has no tools and gets the same company and fiscal year
  - **Report the Comparison**: Plot revenue growth and net margin for the five
    companies, and list the top 3 risks of each company with the sentence of Item 1A
    that supports them
- **Bonus Ideas (Optional)**: Add a Compliance Reviewer agent that flags every claim
  without a source figure; extend the flow to the latest 10-Q to compare the
  trend of the quarter

## Project 3: FOMC Hawkish-Dovish Debate Flow

- **Project Objective**: A debate crew labels Federal Reserve sentences as hawkish,
  dovish, or neutral, and a flow escalates only the uncertain sentences to the crew,
  to maximize macro F1 against human labels at a low number of LLM calls
- **Dataset Suggestions**:
  [FOMC Communication](https://huggingface.co/datasets/gtfintechlab/fomc_communication),
  labeled sentences from FOMC minutes, press conferences, and speeches
  - The CSV files have `sentence`, `year`, and `label` (0 dovish, 1 hawkish, 2
    neutral), and the test split has about 500 sentences
  - Use [FRED - FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) to compare the
    stance with the policy rate
- **Tasks**:
  - **Load the Sentences**: Read `test.csv`, map the labels to names, and sample
    about 150 sentences stratified by label to fix the budget of LLM calls
  - **Define the Debate Crew**: Define a Hawk Analyst, a Dove Analyst, and a Judge as
    `Agent` objects, give them a custom tool that counts hawkish and dovish terms in
    the sentence, and return the label and a short rationale with `output_pydantic`
  - **Orchestrate with a Flow**: Build a `Flow` where a single classifier agent
    labels each sentence with a confidence, and a `@router` step sends the sentences
    below a confidence threshold to the debate crew
  - **Evaluate the Systems**: Compute macro F1, per-class recall, and the confusion
    matrix, with the number of LLM calls per sentence, for a keyword baseline, the
    single classifier, the debate crew on all sentences, and the flow
  - **Interpret the Stance**: Plot the yearly mean stance (hawkish +1, dovish -1) of
    the best system against the change of `FEDFUNDS` in the next year, report the
    Spearman correlation, and state that a few sentences per year give a weak test
- **Bonus Ideas (Optional)**: Run the best system on statements published after the
  training cutoff of the LLM from the
  [FOMC calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
  to check for data contamination; try `Process.hierarchical` with a manager agent
