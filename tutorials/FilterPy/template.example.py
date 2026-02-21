# ---
# jupyter:
#   jupytext:
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
# # Template Example Notebook
#
# This is a template notebook. The first heading should be the title of what notebook is about. For example, if it is a project on neo4j tutorial the heading should be `Project Title`.
#
# - Add description of what the notebook does.
# - Point to references, e.g. (neo4j.example.md)
# - Add citations.
# - Keep the notebook flow clear.
# - Comments should be imperative and have a period at the end.
# - Your code should be well commented.
#
# The name of this notebook should in the following format:
# - if the notebook is exploring `pycaret API`, then it is `pycaret.example.ipynb`
#
# Follow the reference to write notebooks in a clear manner: https://github.com/causify-ai/helpers/blob/master/docs/coding/all.jupyter_notebook.how_to_guide.md

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# %%
import logging
# Import libraries in this section.
# Avoid imports like import *, from ... import ..., from ... import *, etc.

import helpers.hdbg as hdbg
import helpers.hprint as hprint

# %%
hdbg.init_logger(verbosity=logging.INFO)

_LOG = logging.getLogger(__name__)

hprint.config_notebook()


# %% [markdown]
# ## Make the notebook flow clear
# Each notebook needs to follow a clear and logical flow, e.g:
# - Load data
# - Compute stats
# - Clean data
# - Compute stats
# - Do analysis
# - Show results
#
#
#
#
#
#
#
#
#
#
#
#


# #############################################################################
# Template
# #############################################################################


# %%
class Template:
    """
    Brief imperative description of what the class does in one line, if needed.
    """

    def __init__(self):
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
        # Code bloks go here.
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
    # Code bloks go here.
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

# %%
