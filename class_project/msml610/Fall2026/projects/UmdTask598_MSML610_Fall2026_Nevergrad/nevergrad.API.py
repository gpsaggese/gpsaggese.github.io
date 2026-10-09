# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description
#
# This notebook examines ...
#
# The name of this notebook should follow the format: `<tool>.API.ipynb` (e.g., `pycaret.API.ipynb` for exploring the pycaret API).
#
# References:
# - Add description of what the notebook does.
# - Point to references, e.g. (tool.API.md)
# - Add citations.
# - Keep the notebook flow clear.
# - Comments should be imperative and have a period at the end.
# - Your code should be well commented.
#
# Guide: https://github.com/causify-ai/helpers/blob/master/docs/coding/all.jupyter_notebook.how_to_guide.md

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# System libraries.
import logging

# Third-party libraries.
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# %%
# # To install additional packages, use:
# import helpers.hmodule as hmodule
# hmodule.install_module_if_not_present(
#     ["pycaret"],
#     use_activate=True,
#     use_sudo=False,
#     venv_path="/opt/venv",
# )

# %%
# Use this for most notebooks.
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebook

_LOG = logging.getLogger(__name__)

# Initialize notebook configuration and logging.
hdbg.init_logger(verbosity=logging.INFO)
hnotebook.config_notebook()

# Convert `display` into `print()`.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# ## Make the notebook flow clear
# Each notebook needs to follow a clear and logical flow, e.g:
# - Load data
# - Compute stats
# - Clean data
# - Compute stats
# - Do analysis
# - Show results


# %%
class Template:
    """
    Brief imperative description of what the class does in one line, if needed.
    """

    def __init__(self) -> None:
        """
        Initialize the Template class.
        """
        pass

    def method1(self, arg1: int) -> None:
        """
        Brief imperative description of what the method does in one line.

        You can elaborate more in the method docstring in this section, for e.g. explaining
        the formula/algorithm. Every method/function should have a docstring, typehints and include the
        parameters and return as follows:

        :param arg1: description of arg1
        :return: description of return
        """
        # Code blocks go here.
        # Make sure to include comments to explain what the code is doing.
        # No empty lines between code blocks.
        pass


def template_function(arg1: int) -> None:
    """
    Brief imperative description of what the function does in one line.

    You can elaborate more in the function docstring in this section, for e.g. explaining
    the formula/algorithm. Every function should have a docstring, typehints and include the
    parameters and return as follows:

    :param arg1: description of arg1
    :return: description of return
    """
    # Code blocks go here.
    # Make sure to include comments to explain what the code is doing.
    # No empty lines between code blocks.
    pass


# %% [markdown]
# ## The flow should be highlighted using headings in markdown
# ```
# # Level 1
# ## Level 2
# ### Level 3
# ```
