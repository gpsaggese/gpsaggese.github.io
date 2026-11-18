# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Noesis model-drift figures
#
# This notebook regenerates the paper figures from the checked-in Phase 0
# summaries. It makes no model-provider calls and writes the PNG and SVG files
# to `research/Noesis/docs/findings/model_drift_method_comparison/figures`.

# %%
import os
from pathlib import Path

from IPython.display import Image, display

repo_root = next(
    path
    for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "research" / "Noesis").is_dir()
)
os.chdir(repo_root)

from research.Noesis.scripts import plot_model_drift_phase0_results as figures

# %% [markdown]
# ## Generate all figures

# %%
figure_paths = figures.generate_phase0_figures()
figure_paths

# %% [markdown]
# ## Review the generated PNGs

# %%
for figure_path in figure_paths:
    display(Image(filename=figure_path))
