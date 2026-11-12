"""
template_utils.py

This file contains utility functions that support the tutorial notebooks.

- Notebooks should call these functions instead of writing raw logic inline.
- This helps keep the notebooks clean, modular, and easier to debug.
- Students should implement functions here for data preprocessing,
  model setup, evaluation, or any reusable logic.

Import as:

import book_cs_refreshers.tutorials._utils as bcretuut
"""

import logging


import helpers.hnotebook as hnotebo


_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Cell 1: Interactive Distribution Explorer with Plot Type Selection
# #############################################################################


# def cell1_interactive_distribution_explorer(
#     *,
#     figsize: Optional[Tuple[float, float]] = None,
# ) -> None:
#     """
#     Create interactive widget to explore Beta and Normal distributions.
#
#     Demonstrates:
#     - Slider widgets for continuous parameters via build_widget_control()
#     - Dropdown for selecting distribution type
#     - Real-time plot updates using observe() callbacks
#     - Multiple synchronized plots (1x3 layout)
#
#     :param figsize: Optional figure size (width, height). Defaults to
#         plt.rcParams["figure.figsize"]
#     """


# #############################################################################
# Cell 2: Interactive Sampling Visualization
# #############################################################################


# def cell2_interactive_sample_generator(
#     *,
#     figsize: Optional[Tuple[float, float]] = None,
# ) -> None:
#     """
#     Create interactive widget to generate and visualize random samples.
#
#     Demonstrates:
#     - Multiple linked slider widgets via build_widget_control()
#     - Logarithmic scale slider for sample count via build_log_widget_control()
#     - Histogram visualization with theoretical overlay
#     - Sample statistics display
#
#     :param figsize: Optional figure size (width, height). Defaults to
#         plt.rcParams["figure.figsize"]
#     """
