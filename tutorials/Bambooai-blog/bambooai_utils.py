# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: bambooaivenv (3.12.3)
#     language: python
#     name: python3
# ---


import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.io as pio
from bambooai import BambooAI

pio.renderers.default = "jupyterlab"
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
np.set_printoptions(suppress=True, precision=6)


# %%
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT_DIR = Path.cwd()
_HELPERS_ROOT = _ROOT_DIR / "helpers_root"
if str(_HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(_HELPERS_ROOT))

import helpers.hdbg as hdbg
import helpers.hparser as hparser

load_dotenv()
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)


# %%
_DEFAULT_CSV = Path("testdata.csv")


def _setup_env() -> None:
    """
    Ensure dotenv data is loaded and log the workspace root.
    """
    load_dotenv()
    _LOG.debug("Loaded environment from '%s'.", _ROOT_DIR)


def _parse() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the notebook workflow.
    """
    parser = argparse.ArgumentParser(
        description="Run BambooAI with pandas data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv-path",
        default=_DEFAULT_CSV,
        help="Path to the CSV file to process.",
    )
    parser.add_argument(
        "--execution-mode",
        default="",
        help="Runtime execution mode.",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _resolve_execution_mode(mode: str) -> str:
    """
    Validate that we always run with an execution mode.
    """
    hdbg.dassert_ne(mode, "", "Execution mode cannot be empty.")
    return mode


def _load_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    Load the CSV dataset and validate it contains data.
    """
    hdbg.dassert_path_exists(str(csv_path), "CSV file does not exist:", csv_path)
    df = pd.read_csv(csv_path)
    hdbg.dassert_ne(df.shape[0], 0, "Dataframe must contain at least one row.")
    _LOG.debug("Loaded dataframe from '%s' with shape %s.", csv_path, df.shape)
    return df


def _build_bamboo_agent(
    df: pd.DataFrame, *, planning: bool = True, vector_db: bool = False, search_tool: bool = False
) -> BambooAI:
    """
    Construct and configure the BambooAI agent instance.
    """
    bamboo_ai = BambooAI(
        df=df, planning=planning, vector_db=vector_db, search_tool=search_tool
    )
    _LOG.debug(
        "BambooAI agent initialized with planning=%s, vector_db=%s, search_tool=%s.",
        planning,
        vector_db,
        search_tool,
    )
    return bamboo_ai


def _run_agent(bamboo_ai: BambooAI) -> None:
    """
    Execute the BambooAI conversation loop.
    """
    _LOG.info("Starting BambooAI conversation.")
    bamboo_ai.pd_agent_converse()
    _LOG.info("Finished BambooAI conversation.")


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Parse arguments, initialize logging, and run the BambooAI workflow.
    """
    args = parser.parse_args([])
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    _setup_env()
    execution_mode = _resolve_execution_mode(
        args.execution_mode or os.getenv("EXECUTION_MODE", "")
    )
    _LOG.info("Execution mode is '%s'.", execution_mode)
    csv_path = Path(args.csv_path)
    bamboo_df = _load_dataframe(csv_path)
    bamboo_agent = _build_bamboo_agent(bamboo_df)
    _run_agent(bamboo_agent)


# %%
if __name__ == "__main__":
    _main(_parse())
