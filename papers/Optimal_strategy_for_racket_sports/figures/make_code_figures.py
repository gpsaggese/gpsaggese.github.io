"""
Generate the paper figures that come from the code of the project.

- `winner_heatmap_tennis.png` (Section IV-H): the `P(winner)` map of tennis at
  the defaults of Cell 2.1 of notebook 01 (Gdoc Experiment 2.1), computed and
  drawn with the functions of its utils
- `rl_env_vs_m2_pickleball.png` (Section VIII-E): the brute force and the
  Milestone 2 maps of the middle pickleball serve (Gdoc Experiment 6.1),
  extracted from the saved outputs of the full run of notebook 06, which takes
  about 1 hour

Run from the root of the repo, in the container of the project:
> PYTHONPATH=.:helpers_root python3 \
    papers/Optimal_strategy_for_racket_sports/figures/make_code_figures.py
"""

import base64
import json
import logging
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt

import helpers.hdbg as hdbg

import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_01_shot_probability_utils as rosfrsnrs0spu
from research.Optimal_strategy_for_racket_sports import racket_params

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


# Directory of the paper figures, from the root of the repo.
_FIGURES_DIR = "papers/Optimal_strategy_for_racket_sports/figures"

# Notebook 06, whose saved outputs come from its full run.
_NOTEBOOK_06 = (
    "research/Optimal_strategy_for_racket_sports/notebooks/"
    "racket_sports_06_rl_return.ipynb"
)


# #############################################################################
# Figures
# #############################################################################


def _make_winner_heatmap_tennis() -> str:
    """
    Draw the `P(winner)` map of tennis at the defaults of Cell 2.1.

    The settings string is `sport=Tennis side=deuce x1=-2.1 y1=3.2 std_x2=0.5
    std_y2=1.0 ball_speed=25.0 t_r=0.2 v_p=1.5 seed=1 n_x=10 n_y=15
    n_samples=5000`.

    :return: the path of the figure
    """
    _LOG.debug("Drawing the tennis map")
    court = racket_params.TENNIS.court
    serve_xy = (-2.1, 3.2)
    n_x, n_y = rosfrsnrs0spu.HEATMAP_GRID_SIZE["Tennis"]
    heatmap = rosfrsnrs0spu.compute_winner_heatmap(
        serve_xy,
        (0.5, 1.0),
        rosfrsnrs0spu.WINNER_BALL_SPEED_MPS["Tennis"],
        racket_params.DEFAULT_PLAYER.reaction_time_s,
        racket_params.DEFAULT_PLAYER.move_speed_mps,
        court,
        n_x,
        n_y,
        rosfrsnrs0spu.HEATMAP_N_SAMPLES,
        1,
    )
    best = rosfrsnrs0spu.find_best_target(heatmap)
    n_region = int(rosfrsnrs0spu.find_best_region(heatmap).sum())
    _LOG.info(
        "Tennis map: best target (%.2f, %.2f), P(winner) %.4f, region %s of %s"
        " cells",
        best["x_m"],
        best["y_m"],
        best["p_winner"],
        n_region,
        len(heatmap),
    )
    fig, ax = plt.subplots(figsize=(3.4, 5.6))
    rosfrsnrs0spu.plot_winner_heatmap(ax, heatmap, court, serve_xy)
    # At column width the legend of the notebook covers the court, so it goes
    # below the court, with the names of the players used in the paper.
    names = {"player 1": "server (player 1)", "player 2": "returner (player 2)"}
    handles, labels = ax.get_legend_handles_labels()
    labels = [names.get(label, label) for label in labels]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=6.5,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    path = os.path.join(_FIGURES_DIR, "winner_heatmap_tennis.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _get_cells(
    notebook: Dict[str, Any], source_snippet: str
) -> List[Dict[str, Any]]:
    """
    Get the code cells of a notebook whose source contains a snippet.

    :param notebook: the notebook, as loaded from its JSON file
    :param source_snippet: text that the source of the cell contains, e.g.,
        `"get_experiment_config("`
    :return: the matching cells
    """
    cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and source_snippet in "".join(cell["source"])
    ]
    return cells


def _extract_rl_env_vs_m2_pickleball() -> str:
    """
    Extract the brute force and the Milestone 2 maps of the middle pickleball
    serve from the saved outputs of notebook 06.

    :return: the path of the figure
    """
    _LOG.debug("Extracting the pickleball maps from notebook 06")
    with open(_NOTEBOOK_06, encoding="utf-8") as file:
        notebook = json.load(file)
    # The saved outputs must come from the full run, although the source of
    # the cell is saved with `RUN_FULL_SIMS = False`.
    config_cells = _get_cells(notebook, "get_experiment_config(")
    hdbg.dassert_eq(len(config_cells), 1, "Expected 1 configuration cell")
    config_text = "".join(
        "".join(output.get("text", ""))
        for output in config_cells[0].get("outputs", [])
    )
    hdbg.dassert_in(
        "RUN_FULL_SIMS= True",
        config_text,
        "Notebook 06 must be saved from its full run",
    )
    # The cell that plots both maps of the middle serve, one image per sport in
    # the order of `sports`: tennis, then pickleball.
    plot_cells = _get_cells(
        notebook, "# Plot both maps of the middle serve, for each sport."
    )
    hdbg.dassert_eq(len(plot_cells), 1, "Expected 1 plotting cell")
    images = [
        output["data"]["image/png"]
        for output in plot_cells[0]["outputs"]
        if "image/png" in output.get("data", {})
    ]
    hdbg.dassert_eq(len(images), 2, "Expected 1 image per sport")
    path = os.path.join(_FIGURES_DIR, "rl_env_vs_m2_pickleball.png")
    with open(path, "wb") as file:
        file.write(base64.b64decode(images[1]))
    return path


# #############################################################################
# Main
# #############################################################################


def main() -> None:
    """
    Write the figures of the paper that come from the code of the project.
    """
    logging.basicConfig(level=logging.INFO)
    for path in (
        _make_winner_heatmap_tennis(),
        _extract_rl_env_vs_m2_pickleball(),
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
