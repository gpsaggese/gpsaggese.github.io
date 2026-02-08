"""
Utility functions for Lesson 5.1 - Learning Theory VC Dimension notebook.

Import as:

import msml610.tutorials.utils_Lesson05_1_Learning_Theory_VC_Dimension as utils
"""

import logging
from typing import Dict, List, Set, Tuple

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)


# #############################################################################
# Cell 1: Dichotomy Explorer - 2D Perceptron with 3 Points
# #############################################################################


def _classify_points_by_line(
    points: np.ndarray, angle: float, offset: float
) -> np.ndarray:
    """
    Classify points based on which side of a line they fall on.

    The line is defined by angle (in degrees) and offset from origin.

    :param points: Array of shape (N, 2) containing point coordinates
    :param angle: Angle of the line normal in degrees (0-360)
    :param offset: Distance of line from origin
    :return: Array of classifications (+1 or -1) for each point
    """
    # Convert angle to radians.
    theta = np.radians(angle)
    # Normal vector to the line.
    normal = np.array([np.cos(theta), np.sin(theta)])
    # Classify points: +1 if on one side, -1 if on the other.
    classifications = np.sign(np.dot(points, normal) - offset)
    # Handle points exactly on the line.
    classifications[classifications == 0] = 1
    return classifications.astype(int)


def _get_line_endpoints(
    angle: float, offset: float, xlim: Tuple[float, float], ylim: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get endpoints of a line segment for plotting.

    :param angle: Angle of line normal in degrees
    :param offset: Distance of line from origin
    :param xlim: X-axis limits as (min, max)
    :param ylim: Y-axis limits as (min, max)
    :return: Tuple of (x_coords, y_coords) for line endpoints
    """
    theta = np.radians(angle)
    # Normal and tangent vectors.
    normal = np.array([np.cos(theta), np.sin(theta)])
    tangent = np.array([-np.sin(theta), np.cos(theta)])
    # Point on the line closest to origin.
    point_on_line = offset * normal
    # Extend line in both directions.
    t_vals = np.linspace(-10, 10, 100)
    line_points = point_on_line[:, np.newaxis] + tangent[:, np.newaxis] * t_vals
    # Filter points within plot limits.
    mask = (
        (line_points[0] >= xlim[0])
        & (line_points[0] <= xlim[1])
        & (line_points[1] >= ylim[0])
        & (line_points[1] <= ylim[1])
    )
    valid_points = line_points[:, mask]
    if valid_points.shape[1] > 0:
        return valid_points[0], valid_points[1]
    return np.array([]), np.array([])


def _draw_dichotomy_3points(
    angle: float,
    offset: float,
    point_positions: Dict[str, Tuple[float, float]],
    discovered_dichotomies: Set[Tuple[int, int, int]],
) -> None:
    """
    Draw 2D plot with 3 points and separating line.

    :param angle: Angle of separating line in degrees (0-360)
    :param offset: Offset of the line from origin
    :param point_positions: Dictionary mapping point names to (x, y) coordinates
    :param discovered_dichotomies: Set of discovered dichotomies as tuples
    """
    # Create figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Extract points.
    points = np.array(
        [point_positions["A"], point_positions["B"], point_positions["C"]]
    )
    labels = ["A", "B", "C"]
    # Classify points.
    classifications = _classify_points_by_line(points, angle, offset)
    # Update discovered dichotomies.
    dichotomy = tuple(classifications)
    discovered_dichotomies.add(dichotomy)
    # Plot points on left subplot.
    for i, (point, label, classification) in enumerate(
        zip(points, labels, classifications)
    ):
        color = "blue" if classification == 1 else "red"
        ax1.scatter(point[0], point[1], c=color, s=200, edgecolors="black", zorder=3)
        ax1.text(
            point[0],
            point[1] + 0.15,
            label,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    # Plot separating line.
    xlim = (-1.5, 1.5)
    ylim = (-1.5, 1.5)
    x_line, y_line = _get_line_endpoints(angle, offset, xlim, ylim)
    if len(x_line) > 0:
        ax1.plot(x_line, y_line, "g-", linewidth=2, label="Separating line")
    # Format left subplot.
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel("x1", fontsize=12)
    ax1.set_ylabel("x2", fontsize=12)
    ax1.set_title("2D Perceptron - 3 Points", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
    ax1.legend()
    ax1.set_aspect("equal")
    # Create text content for right subplot.
    classification_text = ", ".join(
        [f"{label}: {'+1' if c == 1 else '-1'}" for label, c in zip(labels, classifications)]
    )
    text_content = f"Current Classification:\n{classification_text}\n\n"
    text_content += f"Angle: {angle:.1f} degrees\n"
    text_content += f"Offset: {offset:.2f}\n\n"
    text_content += f"Unique Dichotomies Found: {len(discovered_dichotomies)} / 8\n\n"
    text_content += "All Discovered Dichotomies:\n"
    for i, dichot in enumerate(sorted(discovered_dichotomies), 1):
        dichot_str = ", ".join([f"{label}: {'+1' if c == 1 else '-1'}" for label, c in zip(labels, dichot)])
        text_content += f"{i}. {dichot_str}\n"
    # Add text box to right subplot.
    ax2.axis("off")
    mtumsuti.add_fitted_text_box(ax2, text_content)
    plt.tight_layout()
    plt.show()


def cell1_dichotomy_explorer_3points() -> None:
    """
    Create interactive dichotomy explorer for 3 points.

    Interactive visualization showing how a 2D perceptron can classify 3 points
    in different ways by adjusting the separating line. Helps discover that
    3 points can be classified in 2^3 = 8 different ways.
    """
    # Initialize parameters.
    angle_init = 0.0
    offset_init = 0.0
    # Fixed point positions (in a triangle).
    point_positions = {
        "A": (0.0, 0.8),
        "B": (-0.7, -0.5),
        "C": (0.7, -0.5),
    }
    # Track discovered dichotomies.
    discovered_dichotomies = set()
    # Create widgets for angle and offset.
    angle_slider, angle_box = mtumsuti.build_widget_control(
        name="angle",
        description="angle of line (degrees)",
        min_val=0.0,
        max_val=360.0,
        step=5.0,
        initial_value=angle_init,
        is_float=True,
    )
    offset_slider, offset_box = mtumsuti.build_widget_control(
        name="offset",
        description="line offset",
        min_val=-1.5,
        max_val=1.5,
        step=0.1,
        initial_value=offset_init,
        is_float=True,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        lambda angle, offset: _draw_dichotomy_3points(
            angle, offset, point_positions, discovered_dichotomies
        ),
        {"angle": angle_slider, "offset": offset_slider},
    )
    # Display widgets.
    display(ipywidgets.VBox([angle_box, offset_box, output]))
