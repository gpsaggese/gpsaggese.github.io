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
# # Noesis model-drift analysis
#
# This notebook reproduces the Phase 0 method-comparison analysis from the
# frozen experiment artifacts in `research/Noesis/docs/findings`. It makes no
# provider calls and does not rerun the memory-intensive permutation jobs.
#
# The notebook is a reviewable front end to the same implementation used by
# the command-line analysis script. The generated JSON, CSV, and findings
# report remain under `docs/findings/model_drift_method_comparison`.

# %%
import os
from pathlib import Path

from IPython.display import Markdown, display

repo_root = next(
    path
    for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "research" / "Noesis").is_dir()
)
os.chdir(repo_root)

from research.Noesis.scripts import analyze_model_drift_method_comparison as analysis

# %% [markdown]
# ## Rebuild the frozen-evidence comparison

# %%
summary = analysis.run_method_comparison()
display(summary)

# %% [markdown]
# ## Live full-swap result

# %%
capability = summary["live_full_swap"]["capability"]
mmd = summary["live_full_swap"]["mmd"]
display(
    Markdown(
        "\n".join(
            (
                "| Detector | Frozen result |",
                "| :-- | :-- |",
                "| Capability | 70B: %s/%s; 8B: %s/%s; paired McNemar p = %.6f |"
                % (
                    capability["reference_correct"],
                    capability["reference_gradable"],
                    capability["alternative_correct"],
                    capability["alternative_gradable"],
                    capability["mcnemar_p_value"],
                ),
                "| Hamming-MMD | 60 responses per route; empirical p < %.3f "
                "(%s permutations) |"
                % (mmd["p_value"], mmd["permutations"]),
            )
        )
    )
)
# %% [markdown]
# ## Directional substitution sensitivity

# %%
table_lines = (
    "| Scenario | Capability detection | Hamming-MMD detection |",
    "| :-- | --: | --: |",
) + tuple(
    "| %s | %.1f%% | %.1f%% |"
    % (
        row["scenario"].replace("_", " "),
        100 * row["capability_detection_rate"],
        100 * row["mmd_detection_rate"],
    )
    for row in summary["scenario_comparison"]
)
display(Markdown("\n".join(table_lines)))

# %% [markdown]
# These rates are directional evidence, not a head-to-head ranking. The two
# detectors used different prompt banks, response collections, and trial
# designs. A strict power comparison would require both detectors to operate
# on the same future collection.

# %% [markdown]
# ## Healthy controls

# %%
controls = summary["healthy_controls"]
display(
    Markdown(
        "\n".join(
            (
                "- Capability alerts: **%s/%s** synthetic healthy trials."
                % (controls["capability_alerts"], controls["capability_trials"]),
                "- Hamming-MMD alerts: **%s raw, %s Holm-adjusted, across %s** "
                "same-model comparisons."
                % (
                    controls["mmd_raw_alerts"],
                    controls["mmd_holm_alerts"],
                    controls["mmd_comparisons"],
                ),
            )
        )
    )
)
