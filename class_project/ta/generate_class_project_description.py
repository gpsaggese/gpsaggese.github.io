#!/usr/bin/env python
"""
Generate project descriptions from a CSV file and save them to Markdown files.

This script reads a CSV file with tool information, generates project
descriptions using OpenAI, and saves them as individual Markdown files.
Set the OPENAI_API_KEY using export before running the script.

> class_project/ta/generate_class_project_description.py \
    --input class_project/DATA605/Spring2026/projects.csv \
    --out_dir class_project/DATA605/Spring2026/projects_descriptions \
    --max_projects 2

# Dry-run: print which projects are missing without generating.
> class_project/ta/generate_class_project_description.py \
    --input class_project/DATA605/Spring2026/projects.csv \
    --out_dir class_project/DATA605/Spring2026/projects_descriptions \
    --dry-run

# Disable incremental mode to regenerate all projects.
> class_project/ta/generate_class_project_description.py \
    --input class_project/DATA605/Spring2026/projects.csv \
    --out_dir class_project/DATA605/Spring2026/projects_descriptions \
    --no-incremental
"""

import argparse
import logging
import pathlib
import time
from typing import Any, Optional

import pandas as pd
import tqdm

import helpers_root.helpers.hdbg as hdbg
import helpers_root.helpers.hio as hio
import helpers_root.helpers.hllm as hllm
import helpers_root.helpers.hparser as hparser
import helpers_root.helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Path to the prompt template file.
_PROMPT_FILE_PATH = pathlib.Path(__file__).parent / "project_prompt.md"

# Expected columns in the input CSV.
_EXPECTED_COLUMNS = [
    "Id",
    "Tool",
    "Description",
    "GitHub",
    "Stars",
    "Website",
    "Category",
]

# #############################################################################
# Helper functions
# #############################################################################


def _read_prompt() -> str:
    """
    Read the prompt template from the prompt.txt file.

    :return: the prompt template string
    """
    prompt_path = str(_PROMPT_FILE_PATH)
    hdbg.dassert(
        pathlib.Path(prompt_path).exists(),
        "Prompt file does not exist: %s",
        prompt_path,
    )
    prompt = hio.from_file(prompt_path)
    return prompt


def _read_csv(input_path: str) -> pd.DataFrame:
    """
    Read and validate a CSV file with tool information.

    :param input_path: path to the CSV file
    :return: the dataframe containing tool data
    """
    hdbg.dassert(
        pathlib.Path(input_path).exists(),
        "Input CSV file does not exist: %s",
        input_path,
    )
    df = pd.read_csv(input_path)
    _LOG.debug("Read CSV with shape=%s", df.shape)
    # Validate expected columns are present.
    for col in _EXPECTED_COLUMNS:
        hdbg.dassert_in(col, df.columns, "Missing column '%s' in CSV", col)
    return df


def _generate_project_description(project_name: str) -> Any:
    """
    Generate a project description for the given tool name.

    :param project_name: the name of the project tool
    :return: the generated project description
    """
    # Read prompt from class_project/ta/prompt.txt.
    prompt_template = _read_prompt()
    prompt = f"{prompt_template}\n\nTool: {project_name}"
    project_desc = hllm.get_completion(
        prompt,
        model="gpt-4o-mini",
        cache_mode="NORMAL",
        temperature=0.5,
        max_tokens=1200,
        print_cost=True,
    )
    return project_desc


def create_markdown_file(
    df: pd.DataFrame,
    out_dir: str,
    max_projects: Optional[int],
    *,
    incremental: bool = True,
    dry_run: bool = False,
    sleep_sec: float = 0.5,
) -> pd.DataFrame:
    """
    Create a Markdown file per tool with its generated project description.

    :param df: the dataframe containing the tool information
    :param out_dir: the path to the output Markdown folder
    :param max_projects: limit to the rows processed (None = all)
    :param incremental: skip projects whose output file already exists
    :param dry_run: only print which projects would be generated, do not
        actually generate them
    :param sleep_sec: amount of time to sleep between API requests
    :return: dataframe with tool names and generated GitHub URLs
    """
    file_githublinks_df = pd.DataFrame(columns=["Tool", "URL"])
    rows = df.head(max_projects) if max_projects is not None else df
    if not dry_run:
        hio.create_dir(out_dir, incremental=True)
    for _, row in tqdm.tqdm(rows.iterrows(), total=len(rows)):
        project_name = row["Tool"]
        file_name = f"{project_name.replace(' ', '_')}_Project_Description.md"
        markdown_path = str(pathlib.Path(out_dir) / file_name)
        # In incremental mode skip projects whose output already exists.
        if incremental and pathlib.Path(markdown_path).exists():
            _LOG.warning("Skipping (already exists): %s", file_name)
            continue
        if dry_run:
            _LOG.info("Would generate: %s", file_name)
            continue
        description = _generate_project_description(project_name)
        content = f"{description}\n\n"
        hio.to_file(markdown_path, content)
        _LOG.info("Generated Markdown File: %s", file_name)
        # Run linter on the generated file to ensure proper formatting.
        lint_script = (
            "helpers_root/dev_scripts_helpers/documentation/lint_txt.py"
        )
        cmd = f"{lint_script} -i {markdown_path}"
        hsystem.system(cmd, suppress_output=False)
        # Base GitHub URL for generated project files.
        base_dir = "https://github.com/gpsaggese/umd_classes/tree/master"
        github_url = f"{base_dir}/{out_dir}/{file_name}"
        file_githublinks_df.loc[len(file_githublinks_df)] = [
            project_name,
            github_url,
        ]
        # Wait before triggering the next API request.
        time.sleep(sleep_sec)
    return file_githublinks_df


# #############################################################################
# Script
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    """
    Create the argument parser for the script.

    :return: the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the CSV file with tool information",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output folder for generated Markdown files",
    )
    parser.add_argument(
        "--max_projects",
        type=int,
        default=None,
        help="Limit rows processed (None = all)",
    )
    parser.add_argument(
        "--no_incremental",
        action="store_true",
        default=False,
        help="Disable incremental mode and regenerate all projects",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print which projects would be generated without running",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Generate project descriptions from a CSV file and save them as Markdown.

    :param parser: the argument parser
    """
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Expand user/relative paths to absolute ones early to avoid surprises.
    input_path = str(pathlib.Path(args.input).expanduser().resolve())
    out_dir = str(pathlib.Path(args.out_dir).expanduser().resolve())
    incremental = not args.no_incremental
    dry_run = args.dry_run
    _LOG.info("incremental=%s dry_run=%s", incremental, dry_run)
    _LOG.info("Reading CSV from %s", input_path)
    df = _read_csv(input_path)
    _LOG.info("Processing %d tools", len(df))
    file_githublinks_df = create_markdown_file(
        df,
        out_dir,
        args.max_projects,
        incremental=incremental,
        dry_run=dry_run,
    )
    _LOG.info("Done: %s", out_dir)
    _LOG.debug("GitHub links:\n%s", file_githublinks_df)


if __name__ == "__main__":
    _main(_parse())
