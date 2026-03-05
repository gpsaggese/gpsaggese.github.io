"""
Utility functions for CrewAI-based workflows.

Import as:

import tutorials.crewai.crewai_utils as tcrwuti
"""

import io
import logging
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crewai import LLM
from crewai.tools import tool

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)


# #############################################################################
# LLM utilities
# #############################################################################


def get_local_llm(
    model: str = "ollama/gemma3:latest",
    base_url: str = "http://host.docker.internal:11434",
    temperature: float = 0.2,
) -> LLM:
    """
    Create a local Ollama LLM instance for use with CrewAI agents.

    :param model: the Ollama model identifier (e.g. "ollama/gemma3:latest")
    :param base_url: the Ollama server base URL
    :param temperature: the sampling temperature (0 = deterministic)
    :return: configured LLM instance
    """
    return LLM(model=model, base_url=base_url, temperature=temperature)


# #############################################################################
# Data generation
# #############################################################################


def generate_sales_csv(path: str = "data/sales.csv") -> str:
    """
    Generate a synthetic sales CSV for demonstration purposes.

    Columns: region, month, units_sold, price.

    :param path: destination path for the CSV file
    :return: path to the generated CSV
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "region": (["Northeast", "Midwest", "South", "West"] * 5)[:20],
            "month": list(range(1, 21)),
            "units_sold": rng.integers(10, 500, size=20),
            "price": rng.uniform(5.0, 30.0, size=20).round(2),
        }
    )
    df.to_csv(path, index=False)
    _LOG.info("Generated sales CSV at %s", path)
    return path


# #############################################################################
# EDA tools
# #############################################################################


@tool
def read_head(path: str, n: int = 5) -> str:
    """Preview the top rows of a CSV. Returns a table string."""
    hdbg.dassert(os.path.exists(path), "File not found: %s", path)
    df = pd.read_csv(path).head(int(n))
    buf = io.StringIO()
    df.to_string(buf, index=False)
    return buf.getvalue()


@tool
def plot_histogram(path: str, column: str, bins: int = 20) -> str:
    """Save a histogram PNG for a numeric column. Returns saved path."""
    hdbg.dassert(os.path.exists(path), "File not found: %s", path)
    df = pd.read_csv(path)
    hdbg.dassert_in(column, df.columns, "Column not in CSV")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    hdbg.dassert(not values.empty, "Column '%s' has no numeric data", column)
    plt.figure()
    plt.hist(values, bins=int(bins))
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"Histogram of {column}")
    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    out_path = f"artifacts/hist_{column}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return f"SAVED:{out_path}"


@tool
def groupby_agg(path: str, by: str, metric: str) -> str:
    """Mean(metric) grouped by 'by'. Returns a table string."""
    hdbg.dassert(os.path.exists(path), "File not found: %s", path)
    df = pd.read_csv(path)
    hdbg.dassert_in(by, df.columns, "Group-by column not in CSV")
    hdbg.dassert_in(metric, df.columns, "Metric column not in CSV")
    grouped = (
        df.groupby(by, dropna=False)[metric].mean().reset_index()
    )
    grouped.rename(columns={metric: f"mean_{metric}"}, inplace=True)
    buf = io.StringIO()
    grouped.to_string(buf, index=False)
    return buf.getvalue()


# Convenience list of all EDA tools for agent instantiation.
EDA_TOOLS = [read_head, plot_histogram, groupby_agg]
