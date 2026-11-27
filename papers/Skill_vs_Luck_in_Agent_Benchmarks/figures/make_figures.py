"""
Copy the figures referenced from `paper.md` out of the executed notebooks of
the project, so the paper shows the same plots as the notebooks.

Run from the paper dir, after running the notebooks (see the `README.md` of
the project):
> python3 figures/make_figures.py
"""

import base64
import json
import os

# The project dir, relative to the paper dir.
_PROJECT_DIR = os.path.join(
    "..",
    "..",
    "class_project",
    "msml610",
    "Fall2026",
    "projects",
    "UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks",
)
# For each figure: the notebook, the first line of the code cell that plots
# it, and which image of that cell.
_FIGURES = {
    "leaderboard_cis": (
        "skill_vs_luck.example",
        "# Plot the top 25 agents, shading the interval of the leader.",
        0,
    ),
    "min_tasks_curve": (
        "skill_vs_luck.example",
        "# Plot the tasks needed against the gap size.",
        0,
    ),
    "rank_intervals": (
        "skill_vs_luck.example",
        "# Plot the rank intervals of the top 25 agents.",
        0,
    ),
    "records": (
        "skill_vs_luck.records",
        "# Plot the records over time.",
        0,
    ),
    "variance_parts": (
        "skill_vs_luck.runs",
        "# Plot the task part and the run part of every agent.",
        0,
    ),
    "transfer_scores": (
        "skill_vs_luck.transfer",
        "# Plot the 2 scores of each model and the residuals.",
        0,
    ),
    "best_of_k_runs": (
        "skill_vs_luck.selection",
        "# Plot the measured and the predicted inflation.",
        0,
    ),
    "correction_errors": (
        "skill_vs_luck.selection",
        "# Plot the mean error of each correction.",
        0,
    ),
}


def _get_images(notebook: str, first_line: str) -> list:
    """
    Get the PNG images of the code cell that starts with `first_line`.
    """
    path = os.path.join(_PROJECT_DIR, f"{notebook}.ipynb")
    with open(path) as file:
        cells = json.load(file)["cells"]
    matches = [
        cell
        for cell in cells
        if cell["cell_type"] == "code"
        and "".join(cell["source"]).startswith(first_line)
    ]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} cells start with {first_line!r}")
    images = [
        output["data"]["image/png"]
        for output in matches[0].get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    return images


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for name, (notebook, first_line, index) in _FIGURES.items():
        images = _get_images(notebook, first_line)
        path = os.path.join(out_dir, f"{name}.png")
        with open(path, "wb") as file:
            file.write(base64.b64decode(images[index]))
        print(f"Saved '{path}'")


if __name__ == "__main__":
    main()
