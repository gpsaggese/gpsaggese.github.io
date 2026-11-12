"""
Generate the paper figures that come from the code of the project.

- `winner_heatmap_tennis.png` (Section IV-H): the `P(winner)` map of tennis at
  the defaults of Cell 2.1 of notebook 01 (Gdoc Experiment 2.1), computed and
  drawn with the functions of its utils
- `rl_env_vs_m2_pickleball.png` (Section VIII-E): the brute force map through
  the env and the Monte Carlo map of Section IV-H for the middle pickleball
  serve (Gdoc Experiment 6.1), computed with the functions of the notebook 06
  utils, with the seeds and the samples of its full run, which reproduce the
  maps saved in the notebook; it takes seconds, since it trains no agent

Run from the root of the repo, in the container of the project, with
`gymnasium` installed:
> PYTHONPATH=.:helpers_root python3 \
    papers/Optimal_strategy_for_racket_sports/figures/make_code_figures.py
"""

import logging
import os
from typing import List

import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import helpers.hdbg as hdbg

import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_01_shot_probability_utils as rosfrsnrs0spu
import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_06_rl_return_utils as rosfrsnrs0rru
from research.Optimal_strategy_for_racket_sports import racket_params

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


# Directory of the paper figures, from the root of the repo.
_FIGURES_DIR = "papers/Optimal_strategy_for_racket_sports/figures"

# Names of the players in the paper, for the labels of `plot_winner_heatmap()`.
_PLAYER_NAMES = {"player 1": "server", "player 2": "returner"}


# #############################################################################
# Helpers
# #############################################################################


def _move_legend_below(ax: Axes, fontsize: float) -> None:
    """
    Move the legend of a map below the court, with the names of the paper.

    :param ax: axes drawn by `plot_winner_heatmap()`
    :param fontsize: font size of the legend
    """
    handles, labels = ax.get_legend_handles_labels()
    labels = [_PLAYER_NAMES.get(label, label) for label in labels]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=fontsize,
    )


def _add_shared_legend(fig: Figure, ax: Axes) -> None:
    """
    Replace the legends of the maps with one legend below the figure.

    The labels drop the numbers of each map (the size of the region, the
    `P(winner)` of the best target), which the caption gives, and the returner,
    who hits from the other half of the court.

    :param fig: figure of the maps
    :param ax: one of the maps, drawn by `plot_winner_heatmap()`
    """
    labels_by_prefix = {
        "best region": "best region: within 2 SE of the highest",
        "best target": "best target: middle of the region",
        "player 1": "server",
    }
    handles, labels = [], []
    for handle, label in zip(*ax.get_legend_handles_labels()):
        for prefix, new_label in labels_by_prefix.items():
            if label.startswith(prefix):
                handles.append(handle)
                labels.append(new_label)
    hdbg.dassert_eq(
        len(handles), 3, "Expected the region, the target, and the server"
    )
    for map_ax in fig.axes:
        legend = map_ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.45, -0.04),
        ncol=3,
        fontsize=8,
    )


def _remove_colorbar(ax: Axes) -> mcollections.QuadMesh:
    """
    Remove the colorbar that `plot_winner_heatmap()` adds next to a map.

    :param ax: axes drawn by `plot_winner_heatmap()`
    :return: the mesh of the map, to draw a shared colorbar
    """
    meshes: List[mcollections.QuadMesh] = [
        collection
        for collection in ax.collections
        if isinstance(collection, mcollections.QuadMesh)
    ]
    hdbg.dassert_eq(len(meshes), 1, "Expected 1 map on the axes")
    mesh = meshes[0]
    hdbg.dassert_is_not(mesh.colorbar, None, "Expected a colorbar")
    mesh.colorbar.remove()
    return mesh


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
    _move_legend_below(ax, 6.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    path = os.path.join(_FIGURES_DIR, "winner_heatmap_tennis.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_rl_maps_pickleball() -> str:
    """
    Draw the brute force and the Monte Carlo maps of the middle pickleball
    serve, side by side, with one colorbar.

    The settings are those of the full run of notebook 06: 1000 points per
    target from the reset seed 1000 for the brute force, and 5000 samples with
    seed 2 for the Monte Carlo map.

    :return: the path of the figure
    """
    _LOG.debug("Drawing the pickleball maps of the env")
    sport = racket_params.PICKLEBALL
    court = sport.court
    env = rosfrsnrs0rru.build_env(sport)
    config = rosfrsnrs0rru.get_experiment_config(True)
    n_side = config.n_serves_per_side
    serves = rosfrsnrs0rru.get_eval_serves(env, n_side, n_side)
    serve_xy = serves[n_side**2 // 2]
    heatmaps = {
        "Brute force (environment)": rosfrsnrs0rru.compute_env_heatmap(
            env, serve_xy, config.n_samples, config.env_seed
        ),
        "Monte Carlo map (Sec. IV-H)": rosfrsnrs0rru.compute_m2_heatmap(
            env, serve_xy, config.m2_n_samples, config.m2_seed
        ),
    }
    comparison = rosfrsnrs0rru.compare_heatmaps(*heatmaps.values())
    _LOG.info("Pickleball middle serve %s: %s", serve_xy, comparison)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8))
    for ax, (title, heatmap) in zip(axes, heatmaps.items()):
        rosfrsnrs0spu.plot_winner_heatmap(ax, heatmap, court, serve_xy)
        mesh = _remove_colorbar(ax)
        ax.set_ylim(-court.length_m / 2 - 1, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("y (m)")
    # `plot_winner_heatmap()` labels both maps, and the label of the right map
    # falls between the two.
    axes[1].set_ylabel("")
    _add_shared_legend(fig, axes[0])
    fig.colorbar(mesh, ax=list(axes), shrink=0.8, label="P(winner)")
    path = os.path.join(_FIGURES_DIR, "rl_env_vs_m2_pickleball.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# #############################################################################
# Main
# #############################################################################


def main() -> None:
    """
    Write the figures of the paper that come from the code of the project.
    """
    logging.basicConfig(level=logging.INFO)
    for path in (_make_winner_heatmap_tennis(), _make_rl_maps_pickleball()):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
