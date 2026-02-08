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


def _get_point_configuration(config_name: str) -> Dict[str, Tuple[float, float]]:
    """
    Get predefined point configurations.

    :param config_name: Name of configuration
    :return: Dictionary mapping point names to (x, y) coordinates
    """
    configurations = {
        "collinear1": {
            "A": (-0.8, 0.0),
            "B": (0.0, 0.0),
            "C": (0.8, 0.0),
        },
        "collinear2": {
            "A": (-0.6, -0.6),
            "B": (0.0, 0.0),
            "C": (0.6, 0.6),
        },
        "triangle1": {
            "A": (0.0, 0.8),
            "B": (-0.7, -0.5),
            "C": (0.7, -0.5),
        },
        "triangle2": {
            "A": (-0.8, -0.6),
            "B": (0.8, -0.6),
            "C": (0.8, 0.6),
        },
        "triangle3": {
            "A": (0.0, -0.8),
            "B": (-0.6, 0.5),
            "C": (0.6, 0.5),
        },
    }
    return configurations.get(config_name, configurations["triangle1"])


def _get_target_classification(assignment_idx: int) -> Tuple[int, int, int]:
    """
    Get target classification for a given assignment index.

    :param assignment_idx: Index from 0 to 7 representing one of 2^3 assignments
    :return: Tuple of (A_label, B_label, C_label) each being +1 or -1
    """
    # Generate all 8 possible assignments.
    assignments = []
    for i in range(8):
        # Convert index to binary representation.
        a = 1 if (i & 4) else -1
        b = 1 if (i & 2) else -1
        c = 1 if (i & 1) else -1
        assignments.append((a, b, c))
    return assignments[assignment_idx]


def _find_solution(
    points: np.ndarray, target: Tuple[int, int, int]
) -> Tuple[float, float, bool]:
    """
    Find angle and offset that achieves target classification.

    :param points: Array of shape (3, 2) containing point coordinates
    :param target: Target classification as (A_label, B_label, C_label)
    :return: Tuple of (angle, offset, found) where found indicates success
    """
    # Try different angles and offsets to find a solution.
    for angle in np.linspace(0, 360, 360):
        for offset in np.linspace(-2, 2, 100):
            classifications = _classify_points_by_line(points, angle, offset)
            if tuple(classifications) == target:
                return angle, offset, True
    return 0.0, 0.0, False


def _draw_dichotomy_3points(
    angle: float,
    offset: float,
    point_config: str,
    point_positions: Dict[str, Tuple[float, float]],
) -> None:
    """
    Draw 2D plot with 3 points and separating line.

    :param angle: Angle of separating line in degrees (0-360)
    :param offset: Offset of the line from origin
    :param point_config: Point configuration name ("collinear" or "triangle")
    :param point_positions: Dictionary mapping point names to (x, y) coordinates
    """
    # Create figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Get points based on configuration.
    points = np.array(
        [point_positions["A"], point_positions["B"], point_positions["C"]]
    )
    labels = ["A", "B", "C"]
    # Classify points with current line.
    current_classification = _classify_points_by_line(points, angle, offset)
    # Plot points on left subplot.
    for i, (point, label, classification) in enumerate(
        zip(points, labels, current_classification)
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
    text_content = "Comment:\n"
    text_content += "Explore how a 2D perceptron (separating line) can classify\n"
    text_content += "3 points in different ways. Adjust the angle and offset\n"
    text_content += "to discover all possible dichotomies.\n"
    text_content += "\n"
    text_content += "-" * 50 + "\n\n"
    text_content += f"Point Configuration:  {point_config}\n\n"
    text_content += f"Angle:                {angle:.1f} degrees\n"
    text_content += f"Offset:               {offset:.2f}\n\n"
    text_content += "Current Classification:\n"
    for label, classification in zip(labels, current_classification):
        sign = "+1" if classification == 1 else "-1"
        text_content += f"  {label}:                  {sign}\n"
    # Add text box to right subplot.
    ax2.axis("off")
    mtumsuti.add_fitted_text_box(ax2, text_content)
    plt.tight_layout()
    plt.show()


def cell1_dichotomy_explorer_3points() -> None:
    """
    Create interactive dichotomy explorer for 3 points.

    Interactive visualization showing how a 2D perceptron can classify 3 points
    in different ways by adjusting the separating line.
    """
    # Initialize parameters.
    angle_init = 0.0
    offset_init = 0.0
    point_config_init = "triangle1"
    # Store current point configuration.
    current_point_positions = {"value": _get_point_configuration(point_config_init)}
    # Create dropdown for point configuration.
    config_dropdown = ipywidgets.Dropdown(
        options=["collinear1", "collinear2", "triangle1", "triangle2", "triangle3"],
        value=point_config_init,
        description="Point Config:",
        style={"description_width": "120px"},
    )
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

    def on_config_change(change):
        """Update point positions when configuration changes."""
        current_point_positions["value"] = _get_point_configuration(change["new"])

    config_dropdown.observe(on_config_change, names="value")
    # Create interactive output.
    output = ipywidgets.interactive_output(
        lambda angle, offset, config: _draw_dichotomy_3points(
            angle,
            offset,
            config,
            current_point_positions["value"],
        ),
        {
            "angle": angle_slider,
            "offset": offset_slider,
            "config": config_dropdown,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                config_dropdown,
                angle_box,
                offset_box,
                output,
            ]
        )
    )
