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
# # Prompt Tokens Billed per Character
#
# - This notebook fits the prompt tokens that the seed sellers billed in the
#   2026-11-12 run of `scripts/experiment_prompt_tokens.py` (#720), so the
#   gateway can estimate a prompt's tokens from its characters
# - The pedagogical arc:
#   - Load the raw calls and see which sellers answered
#   - Fit characters per token and the framing per request and per message
#   - Plot billed tokens against characters

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
import prompt_tokens_utils as utils

import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

import research.Noesis.scripts.experiment_prompt_tokens as rnsexprto

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %%
# The analysis signature: the raw calls to fit.
config = {
    "raw_path": rnsexprto.RAW_PATH,
}
print("config=", config)

# %% [markdown]
# # Part 1: Fit the Billed Tokens

# %% [markdown]
# ## Cell 1.1: Load the Raw Calls
#
# **Goal**
# - See how many calls each seller answered and what the run cost
#
# **Description**
# - Inputs
#   - `config["raw_path"]`: one JSON line per call, written by the script
# - Output
#   - One row per seller: calls, successes, failed statuses, and spend

# %%
# Load the raw calls and count them per seller.
measurements = rnsexprto.load_measurements(config["raw_path"])
display(utils.cell1_1_summarize_calls(measurements))
# Together and Novita fail every call: OpenRouter no longer serves the model
# from them.

# %% [markdown]
# **Implementation** `cell1_1_summarize_calls(measurements)`
# - Groups the calls by seller and counts the statuses other than `ok`

# %%
hintros.print_obj_info(utils.cell1_1_summarize_calls)

# %% [markdown]
# ## Cell 1.2: Fit Characters per Token and Framing
#
# **Goal**
# - Measure how many characters each seller bills as one prompt token, and the
#   tokens it adds per request and per extra message
#
# **Description**
# - Inputs
#   - The successful calls of Cell 1.1
# - Output
#   - One row per seller and text type
#   - `chars_per_token`: the inverse slope of tokens against characters
#   - `request_overhead`: the intercept, the framing of a one-message request
#   - `per_message_overhead`: the extra tokens of each message after the first

# %%
# Fit each seller and text type on its one-message and five-message calls.
fits = utils.fit(measurements)
display(utils.cell1_2_fit_table(fits))
# Numbers bill 1.75 characters per token, code 2.44, and prose about 5.1; Groq
# adds about 35 to 40 framing tokens, the others 9 to 16.

# %% [markdown]
# **Implementation** `fit(measurements)`, `cell1_2_fit_table(fits)`
# - Fits a line on the one-message calls with `statistics.linear_regression()`
# - Averages, per extra message, what a split call bills beyond the same text
#   as one message

# %%
hintros.print_obj_info(utils.fit)

# %% [markdown]
# # Part 2: Plot the Billed Tokens

# %% [markdown]
# ## Cell 2.1: Billed Tokens against Characters
#
# **Goal**
# - See that billed tokens grow linearly with characters, with a slope set by
#   the text type
#
# **Description**
# - Inputs
#   - The one-message calls of Cell 1.1
# - Panels
#   - One panel per text type, one line per seller

# %%
# Plot the billed tokens of the one-message calls.
utils.cell2_1_plot_tokens_vs_chars(measurements)
# The lines are straight, and numbers are steepest: they cost the most tokens
# per character.

# %% [markdown]
# **Implementation** `cell2_1_plot_tokens_vs_chars(measurements, figsize=None)`
# - Draws one `ax.plot()` line per seller in each text-type panel

# %%
hintros.print_obj_info(utils.cell2_1_plot_tokens_vs_chars)
