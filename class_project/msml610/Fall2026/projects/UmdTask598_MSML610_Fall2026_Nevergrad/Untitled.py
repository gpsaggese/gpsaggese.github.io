# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import nevergrad_utils as nvutils

etf = nvutils.load_returns("etf")
fr = nvutils.load_returns("french")
etf.shape, etf.index[0], fr.shape, fr.index[0], fr.index[-1]
