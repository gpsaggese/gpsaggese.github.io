"""
nevergrad_utils.py

Utility functions shared by the Nevergrad tutorial notebooks
(nevergrad.API.ipynb and nevergrad.example.ipynb).

- Notebooks should call these functions instead of writing raw logic inline,
  to keep the notebooks clean, modular, and easier to debug.
- Reusable logic (data loading, walk-forward splitting, portfolio statistics,
  cost-aware losses, Nevergrad optimization, out-of-sample evaluation, and
  plotting) will be implemented here as the project progresses.

Import as:

import nevergrad_utils as nvutils
"""

import logging

_LOG = logging.getLogger(__name__)
