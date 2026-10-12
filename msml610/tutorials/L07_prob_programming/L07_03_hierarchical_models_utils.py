"""
Utility functions for the hierarchical models tutorial (L07_03).

Import as:

import msml610.tutorials.L07_prob_programming.L07_03_hierarchical_models_utils as mtlppl0hmu
"""

import logging
from typing import Any

import arviz as az

import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Cell 2.6: Comparing hierarchical vs non-hierarchical estimates
# #############################################################################


def plot_group_comparison_forest(idata_h: Any, idata_nh: Any) -> None:
    """
    Compare hierarchical vs non-hierarchical group-mean estimates.

    Plots both models' 94% credible intervals for `mu` on one forest plot
    (there are 20 groups and each model has 20 estimates), plus the
    hierarchical model's global mean and the non-hierarchical model's
    overall mean as reference lines. The hierarchical (blue) means are
    pulled toward the global mean relative to the non-hierarchical
    (orange) ones.

    :param idata_h: hierarchical model's inference data
    :param idata_nh: non-hierarchical model's inference data
    """
    axes = az.plot_forest(
        [idata_h, idata_nh],
        model_names=["h", "n_h"],
        var_names="mu",
        combined=True,
        colors="cycle",
    )
    y_lims = axes[0].get_ylim()
    axes[0].vlines(idata_h.posterior["mu_mu"].mean(), *y_lims, color="navy")
    axes[0].vlines(idata_nh.posterior["mu"].mean(), *y_lims, color="orange")
