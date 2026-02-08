"""
Utility functions for Lesson 5.1 - Learning Theory VC Dimension notebook.

Import as:

import msml610.tutorials.utils_Lesson05_1_Learning_Theory_VC_Dimension as mtul1ltvd
"""

import logging
from typing import Dict, Set, Tuple

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
    angle: float,
    offset: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
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
        ax1.scatter(
            point[0], point[1], c=color, s=200, edgecolors="black", zorder=3
        )
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
    text_content += (
        "Explore how a 2D perceptron (separating line) can classify\n"
    )
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
    current_point_positions = {
        "value": _get_point_configuration(point_config_init)
    }
    # Create dropdown for point configuration.
    config_dropdown = ipywidgets.Dropdown(
        options=[
            "collinear1",
            "collinear2",
            "triangle1",
            "triangle2",
            "triangle3",
        ],
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
        current_point_positions["value"] = _get_point_configuration(
            change["new"]
        )

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


# #############################################################################
# Cell 2: Dichotomy Explorer - 2D Perceptron with 3 Points (Target Assignment)
# #############################################################################


def _draw_dichotomy_3points_with_target(
    angle: float,
    offset: float,
    point_config: str,
    target_idx: int,
    point_positions: Dict[str, Tuple[float, float]],
) -> None:
    """
    Draw 2D plot with 3 points, separating line, and target classification.

    :param angle: Angle of separating line in degrees (0-360)
    :param offset: Offset of the line from origin
    :param point_config: Point configuration name
    :param target_idx: Index of target classification (0-7)
    :param point_positions: Dictionary mapping point names to (x, y) coordinates
    """
    # Create figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Get points based on configuration.
    points = np.array(
        [point_positions["A"], point_positions["B"], point_positions["C"]]
    )
    labels = ["A", "B", "C"]
    # Get target classification.
    target_classification = _get_target_classification(target_idx)
    # Classify points with current line.
    current_classification = _classify_points_by_line(points, angle, offset)
    # Check if current matches target.
    match = tuple(current_classification) == target_classification
    # Plot points on left subplot.
    for i, (point, label, target_class) in enumerate(
        zip(points, labels, target_classification)
    ):
        # Color based on target classification.
        color = "blue" if target_class == 1 else "red"
        ax1.scatter(
            point[0], point[1], c=color, s=200, edgecolors="black", zorder=3
        )
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
        line_color = "green" if match else "orange"
        line_style = "-" if match else "--"
        ax1.plot(
            x_line,
            y_line,
            color=line_color,
            linestyle=line_style,
            linewidth=2,
            label="Separating line",
        )
    # Format left subplot.
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel("x1", fontsize=12)
    ax1.set_ylabel("x2", fontsize=12)
    title = "2D Perceptron - 3 Points (Target Assignment)"
    if match:
        title += " - MATCH!"
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
    ax1.legend()
    ax1.set_aspect("equal")
    # Create text content for right subplot.
    text_content = "Comment:\n"
    text_content += "Discover that 3 points can be classified in 2^3 = 8\n"
    text_content += "different ways. Adjust the angle and offset to match\n"
    text_content += "the target classification shown by the point colors.\n"
    text_content += "Use 'Find Solution' to see a working configuration.\n"
    text_content += "\n"
    text_content += "-" * 50 + "\n\n"
    text_content += f"Point Configuration:   {point_config}\n\n"
    text_content += "Target Classification:\n"
    for label, target_class in zip(labels, target_classification):
        sign = "+1" if target_class == 1 else "-1"
        text_content += f"  {label}:                   {sign}\n"
    text_content += "\n"
    text_content += "Current Classification:\n"
    for label, current_class in zip(labels, current_classification):
        sign = "+1" if current_class == 1 else "-1"
        text_content += f"  {label}:                   {sign}\n"
    text_content += "\n"
    if match:
        text_content += "STATUS: MATCH! You found the correct classification!\n"
    else:
        text_content += "STATUS: Keep adjusting to match the target.\n"
    # Add text box to right subplot.
    ax2.axis("off")
    mtumsuti.add_fitted_text_box(ax2, text_content)
    plt.tight_layout()
    plt.show()


def cell2_dichotomy_explorer_3points_target() -> None:
    """
    Create interactive dichotomy explorer with target assignment.

    Shows target classification and allows user to find the separating line
    that achieves it.
    """
    # Initialize parameters.
    angle_init = 0.0
    offset_init = 0.0
    point_config_init = "triangle1"
    target_idx_init = 0
    # Store current point configuration.
    current_point_positions = {
        "value": _get_point_configuration(point_config_init)
    }
    # Create dropdown for point configuration.
    config_dropdown = ipywidgets.Dropdown(
        options=[
            "collinear1",
            "collinear2",
            "triangle1",
            "triangle2",
            "triangle3",
        ],
        value=point_config_init,
        description="Point Config:",
        style={"description_width": "120px"},
    )
    # Create dropdown for target assignment.
    target_options = []
    for i in range(8):
        assignment = _get_target_classification(i)
        a_sign = "+" if assignment[0] == 1 else "-"
        b_sign = "+" if assignment[1] == 1 else "-"
        c_sign = "+" if assignment[2] == 1 else "-"
        label = f"Assignment {i}: A={a_sign}1, B={b_sign}1, C={c_sign}1"
        target_options.append((label, i))
    target_dropdown = ipywidgets.Dropdown(
        options=target_options,
        value=target_idx_init,
        description="Target:",
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
    # Create button to find solution.
    find_button = ipywidgets.Button(
        description="Find Solution",
        button_style="info",
        tooltip="Find angle and offset that match target",
    )

    def on_config_change(change):
        """Update point positions when configuration changes."""
        current_point_positions["value"] = _get_point_configuration(
            change["new"]
        )

    def on_find_click(b):
        """Find solution for current target."""
        points = np.array(
            [
                current_point_positions["value"]["A"],
                current_point_positions["value"]["B"],
                current_point_positions["value"]["C"],
            ]
        )
        target = _get_target_classification(target_dropdown.value)
        angle, offset, found = _find_solution(points, target)
        if found:
            angle_slider.value = angle
            offset_slider.value = offset
        else:
            _LOG.warning("No solution found for target classification")

    config_dropdown.observe(on_config_change, names="value")
    find_button.on_click(on_find_click)
    # Create interactive output.
    output = ipywidgets.interactive_output(
        lambda angle,
        offset,
        config,
        target: _draw_dichotomy_3points_with_target(
            angle,
            offset,
            config,
            target,
            current_point_positions["value"],
        ),
        {
            "angle": angle_slider,
            "offset": offset_slider,
            "config": config_dropdown,
            "target": target_dropdown,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                config_dropdown,
                target_dropdown,
                angle_box,
                offset_box,
                find_button,
                output,
            ]
        )
    )


# #############################################################################
# Cell 3: Dichotomy Explorer - 2D Perceptron with 4 Points
# #############################################################################


def _get_point_configuration_4points(
    config_name: str,
) -> Dict[str, Tuple[float, float]]:
    """
    Get predefined 4-point configurations.

    :param config_name: Name of configuration
    :return: Dictionary mapping point names to (x, y) coordinates
    """
    configurations = {
        "square": {
            "A": (-0.6, -0.6),
            "B": (0.6, -0.6),
            "C": (0.6, 0.6),
            "D": (-0.6, 0.6),
        },
        "circle": {
            "A": (0.0, 0.8),
            "B": (0.8, 0.0),
            "C": (0.0, -0.8),
            "D": (-0.8, 0.0),
        },
        "line": {
            "A": (-0.9, 0.0),
            "B": (-0.3, 0.0),
            "C": (0.3, 0.0),
            "D": (0.9, 0.0),
        },
        "diamond": {
            "A": (0.0, 0.8),
            "B": (0.6, 0.0),
            "C": (0.0, -0.8),
            "D": (-0.6, 0.0),
        },
    }
    return configurations.get(config_name, configurations["square"])


def _classify_points_by_line_4points(
    points: np.ndarray, angle: float, offset: float
) -> np.ndarray:
    """
    Classify 4 points based on which side of a line they fall on.

    :param points: Array of shape (4, 2) containing point coordinates
    :param angle: Angle of the line normal in degrees (0-360)
    :param offset: Distance of line from origin
    :return: Array of classifications (+1 or -1) for each point
    """
    return _classify_points_by_line(points, angle, offset)


def _get_impossible_dichotomies_4points(
    points: np.ndarray,
) -> Set[Tuple[int, ...]]:
    """
    Find which dichotomies cannot be achieved for 4 points.

    :param points: Array of shape (4, 2) containing point coordinates
    :return: Set of impossible classification tuples
    """
    # Try all 16 possible classifications.
    all_dichotomies = set()
    achievable_dichotomies = set()
    for i in range(16):
        # Convert index to binary representation.
        target = tuple(1 if (i & (1 << (3 - j))) else -1 for j in range(4))
        all_dichotomies.add(target)
        # Try to find a line that achieves this classification.
        for angle in np.linspace(0, 360, 180):
            for offset in np.linspace(-2, 2, 100):
                classifications = _classify_points_by_line_4points(
                    points, angle, offset
                )
                if tuple(classifications) == target:
                    achievable_dichotomies.add(target)
                    break
            if target in achievable_dichotomies:
                break
    return all_dichotomies - achievable_dichotomies


def _draw_dichotomy_4points(
    angle: float,
    offset: float,
    point_config: str,
    point_positions: Dict[str, Tuple[float, float]],
    unique_dichotomies: Set[Tuple[int, ...]],
) -> None:
    """
    Draw 2D plot with 4 points and separating line.

    :param angle: Angle of separating line in degrees (0-360)
    :param offset: Offset of the line from origin
    :param point_config: Point configuration name
    :param point_positions: Dictionary mapping point names to (x, y) coordinates
    :param unique_dichotomies: Set of unique dichotomies found so far
    """
    # Create figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Get points based on configuration.
    points = np.array(
        [
            point_positions["A"],
            point_positions["B"],
            point_positions["C"],
            point_positions["D"],
        ]
    )
    labels = ["A", "B", "C", "D"]
    # Classify points with current line.
    current_classification = _classify_points_by_line_4points(
        points, angle, offset
    )
    # Plot points on left subplot.
    for i, (point, label, classification) in enumerate(
        zip(points, labels, current_classification)
    ):
        color = "blue" if classification == 1 else "red"
        ax1.scatter(
            point[0], point[1], c=color, s=200, edgecolors="black", zorder=3
        )
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
    ax1.set_title("2D Perceptron - 4 Points", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
    ax1.legend()
    ax1.set_aspect("equal")
    # Get impossible dichotomies for this configuration.
    impossible = _get_impossible_dichotomies_4points(points)
    # Check if current is XOR pattern for square.
    is_xor = False
    if point_config == "square":
        # XOR patterns: (A, C same) and (B, D same) but (A != B).
        xor_patterns = [
            (1, -1, 1, -1),
            (-1, 1, -1, 1),
        ]
        if tuple(current_classification) in xor_patterns:
            is_xor = True
    # Create text content for right subplot.
    text_content = "Comment:\n"
    text_content += "With 4 points, not all 2^4 = 16 classifications are\n"
    text_content += "possible with a linear separator. Try different\n"
    text_content += "angles and offsets to discover the limit.\n"
    text_content += "\n"
    text_content += "-" * 50 + "\n\n"
    text_content += f"Point Configuration:   {point_config}\n\n"
    text_content += "Current Classification:\n"
    for label, classification in zip(labels, current_classification):
        sign = "+1" if classification == 1 else "-1"
        text_content += f"  {label}:                   {sign}\n"
    text_content += "\n"
    text_content += f"Unique dichotomies found: {len(unique_dichotomies)}\n"
    text_content += "Maximum achievable:       14\n"
    text_content += "Total possible (2^4):     16\n"
    text_content += f"Impossible:               {len(impossible)}\n"
    text_content += "\n"
    if is_xor:
        text_content += "XOR PATTERN DETECTED!\n"
        text_content += "This pattern is NOT achievable with a\n"
        text_content += "linear separator (2D perceptron).\n"
    else:
        text_content += "This introduces the concept of BREAK POINT:\n"
        text_content += "For 2D perceptron, break point k = 4\n"
        text_content += "because not all 2^4 dichotomies are achievable.\n"
    # Add text box to right subplot.
    ax2.axis("off")
    mtumsuti.add_fitted_text_box(ax2, text_content)
    plt.tight_layout()
    plt.show()


def cell3_dichotomy_explorer_4points() -> None:
    """
    Create interactive dichotomy explorer for 4 points.

    Shows that with 4 points, not all 16 classifications are possible,
    introducing the concept of break point.
    """
    # Initialize parameters.
    angle_init = 0.0
    offset_init = 0.0
    point_config_init = "square"
    # Store current point configuration.
    current_point_positions = {
        "value": _get_point_configuration_4points(point_config_init)
    }
    # Store unique dichotomies found.
    unique_dichotomies: Dict[str, Set[Tuple[int, ...]]] = {"value": set()}
    # Create dropdown for point configuration.
    config_dropdown = ipywidgets.Dropdown(
        options=["square", "circle", "line", "diamond"],
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
    # Create button to reset dichotomy counter.
    reset_button = ipywidgets.Button(
        description="Reset Counter",
        button_style="warning",
        tooltip="Reset unique dichotomies counter",
    )

    def on_config_change(change):
        """Update point positions and reset counter when configuration changes."""
        current_point_positions["value"] = _get_point_configuration_4points(
            change["new"]
        )
        unique_dichotomies["value"] = set()

    def on_reset_click(b):
        """Reset the unique dichotomies counter."""
        unique_dichotomies["value"] = set()

    def update_plot(angle, offset, config):
        """Update plot and track unique dichotomies."""
        points = np.array(
            [
                current_point_positions["value"]["A"],
                current_point_positions["value"]["B"],
                current_point_positions["value"]["C"],
                current_point_positions["value"]["D"],
            ]
        )
        current_classification = _classify_points_by_line_4points(
            points, angle, offset
        )
        unique_dichotomies["value"].add(tuple(current_classification))
        _draw_dichotomy_4points(
            angle,
            offset,
            config,
            current_point_positions["value"],
            unique_dichotomies["value"],
        )

    config_dropdown.observe(on_config_change, names="value")
    reset_button.on_click(on_reset_click)
    # Create interactive output.
    output = ipywidgets.interactive_output(
        update_plot,
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
                reset_button,
                output,
            ]
        )
    )
