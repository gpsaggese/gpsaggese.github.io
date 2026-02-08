"""
Utility functions for Lesson 5.1 - Learning Theory VC Dimension notebook.

Import as:

import msml610.tutorials.utils_Lesson05_1_Learning_Theory_VC_Dimension as mtul1ltvd
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
        # Both fill and border show current classification (no target in this cell).
        color = "blue" if classification == 1 else "red"
        ax1.scatter(
            point[0],
            point[1],
            c=color,
            s=200,
            edgecolors=color,
            linewidths=3,
            zorder=3,
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
    for i, (point, label, target_class, current_class) in enumerate(
        zip(points, labels, target_classification, current_classification)
    ):
        # Fill color based on target classification.
        fill_color = "blue" if target_class == 1 else "red"
        # Edge color based on current classification (assigned by hyperplane).
        edge_color = "blue" if current_class == 1 else "red"
        ax1.scatter(
            point[0],
            point[1],
            c=fill_color,
            s=200,
            edgecolors=edge_color,
            linewidths=3,
            zorder=3,
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
    text_content += "the target classification.\n\n"
    text_content += "Visualization:\n"
    text_content += "  • Fill color = Target assignment\n"
    text_content += "  • Border color = Hyperplane assignment\n"
    text_content += "  • When colors match, classification is correct!\n\n"
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
) -> None:
    """
    Draw 2D plot with 4 points and separating line.

    :param angle: Angle of separating line in degrees (0-360)
    :param offset: Offset of the line from origin
    :param point_config: Point configuration name
    :param point_positions: Dictionary mapping point names to (x, y) coordinates
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
        # Both fill and border show current classification.
        color = "blue" if classification == 1 else "red"
        ax1.scatter(
            point[0],
            point[1],
            c=color,
            s=200,
            edgecolors=color,
            linewidths=3,
            zorder=3,
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

    def on_config_change(change):
        """Update point positions when configuration changes."""
        current_point_positions["value"] = _get_point_configuration_4points(
            change["new"]
        )

    config_dropdown.observe(on_config_change, names="value")
    # Create interactive output.
    output = ipywidgets.interactive_output(
        lambda angle, offset, config: _draw_dichotomy_4points(
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
# Cell 4: Dichotomy Explorer - Positive Rays
# #############################################################################


def _draw_positive_rays(n: int, threshold: float) -> None:
    """
    Draw 1D number line with N points and threshold for positive rays.

    :param n: Number of points
    :param threshold: Threshold position
    """
    # Create figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Create points evenly spaced on number line.
    points = np.linspace(-1.0, 1.0, n)
    # Classify points: +1 if >= threshold, -1 otherwise.
    current_classification = np.where(points >= threshold, 1, -1)
    # Plot points on left subplot.
    for i, (point, classification) in enumerate(
        zip(points, current_classification)
    ):
        # Both colors based on current classification.
        fill_color = "blue" if classification == 1 else "red"
        edge_color = fill_color
        ax1.scatter(
            point,
            0,
            c=fill_color,
            s=200,
            edgecolors=edge_color,
            linewidths=3,
            zorder=3,
        )
        ax1.text(
            point,
            0.15,
            f"P{i+1}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Plot threshold line.
    ax1.axvline(x=threshold, color="green", linewidth=2, label="Threshold")
    # Format left subplot.
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel("Position", fontsize=12)
    ax1.set_yticks([])
    title = "Positive Rays"
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.legend()
    # Create text content for right subplot.
    text_content = "Comment:\n"
    text_content += "Positive rays: Points to the right of threshold 'a'\n"
    text_content += "are classified as +1, points to the left as -1.\n\n"
    text_content += "This is the simplest hypothesis set where the growth\n"
    text_content += "function is linear: m_H(N) = N + 1.\n\n"
    text_content += "Key observation:\n"
    text_content += "- There are N+1 possible dichotomies for N points\n"
    text_content += "- Threshold can be placed before first point, after\n"
    text_content += "  last point, or between any two consecutive points\n"
    text_content += "\n"
    text_content += "-" * 50 + "\n\n"
    text_content += f"N (number of points):  {n}\n"
    text_content += f"Threshold position:    {threshold:.2f}\n"
    text_content += f"m_H(N):                {n + 1}\n"
    text_content += f"2^N:                   {2**n}\n\n"
    text_content += "Current Classification:\n"
    for i, classification in enumerate(current_classification):
        sign = "+1" if classification == 1 else "-1"
        text_content += f"  P{i+1}:                {sign}\n"
    # Add text box to right subplot.
    ax2.axis("off")
    mtumsuti.add_fitted_text_box(ax2, text_content)
    plt.tight_layout()
    plt.show()


def cell4_dichotomy_explorer_positive_rays() -> None:
    """
    Create interactive dichotomy explorer for positive rays.

    Demonstrates linear growth function m_H(N) = N + 1.
    """
    # Initialize parameters.
    n_init = 5
    threshold_init = 0.0
    # Create widgets for N and threshold.
    n_slider, n_box = mtumsuti.build_widget_control(
        name="N",
        description="number of points",
        min_val=1,
        max_val=10,
        step=1,
        initial_value=n_init,
        is_float=False,
    )
    threshold_slider, threshold_box = mtumsuti.build_widget_control(
        name="threshold",
        description="threshold position",
        min_val=-1.5,
        max_val=1.5,
        step=0.1,
        initial_value=threshold_init,
        is_float=True,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _draw_positive_rays,
        {
            "n": n_slider,
            "threshold": threshold_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                n_box,
                threshold_box,
                output,
            ]
        )
    )


# #############################################################################
# Cell 5: Dichotomy Explorer - Positive Intervals
# #############################################################################


def _draw_positive_intervals(
    n: int, left_threshold: float, right_threshold: float
) -> None:
    """
    Draw 1D number line with N points and interval for positive intervals.

    :param n: Number of points
    :param left_threshold: Left boundary of interval
    :param right_threshold: Right boundary of interval
    """
    # Create figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Create points evenly spaced on number line.
    points = np.linspace(-1.0, 1.0, n)
    # Classify points: +1 if in [left, right], -1 otherwise.
    current_classification = np.where(
        (points >= left_threshold) & (points <= right_threshold), 1, -1
    )
    # Plot points on left subplot.
    for i, (point, classification) in enumerate(
        zip(points, current_classification)
    ):
        # Both colors based on current classification.
        fill_color = "blue" if classification == 1 else "red"
        edge_color = fill_color
        ax1.scatter(
            point,
            0,
            c=fill_color,
            s=200,
            edgecolors=edge_color,
            linewidths=3,
            zorder=3,
        )
        ax1.text(
            point,
            0.15,
            f"P{i+1}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Plot interval boundaries.
    ax1.axvline(
        x=left_threshold, color="green", linewidth=2, linestyle="--", label="Left boundary"
    )
    ax1.axvline(
        x=right_threshold, color="purple", linewidth=2, linestyle="--", label="Right boundary"
    )
    # Shade the interval region.
    ax1.axvspan(left_threshold, right_threshold, alpha=0.2, color="blue")
    # Format left subplot.
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel("Position", fontsize=12)
    ax1.set_yticks([])
    title = "Positive Intervals"
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.legend()
    # Create text content for right subplot.
    text_content = "Comment:\n"
    text_content += "Positive intervals: Points inside [a, b] are +1,\n"
    text_content += "points outside are -1.\n\n"
    text_content += "Growth function is quadratic: m_H(N) ~ N^2/2 + N + 1.\n\n"
    text_content += "Key observation:\n"
    text_content += "- Can select any contiguous interval of points\n"
    text_content += "- Number of intervals grows quadratically with N\n"
    text_content += "- Still polynomial growth -> learning is feasible\n"
    text_content += "\n"
    text_content += "-" * 50 + "\n\n"
    text_content += f"N (number of points):  {n}\n"
    text_content += f"Left boundary:         {left_threshold:.2f}\n"
    text_content += f"Right boundary:        {right_threshold:.2f}\n"
    # Estimate m_H(N) for positive intervals.
    m_h_n = n * (n + 1) // 2 + n + 1
    text_content += f"m_H(N) (approx):       {m_h_n}\n"
    text_content += f"2^N:                   {2**n}\n\n"
    text_content += "Current Classification:\n"
    for i, classification in enumerate(current_classification):
        sign = "+1" if classification == 1 else "-1"
        text_content += f"  P{i+1}:                {sign}\n"
    # Add text box to right subplot.
    ax2.axis("off")
    mtumsuti.add_fitted_text_box(ax2, text_content)
    plt.tight_layout()
    plt.show()


def cell5_dichotomy_explorer_positive_intervals() -> None:
    """
    Create interactive dichotomy explorer for positive intervals.

    Demonstrates quadratic growth function m_H(N) ~ N^2/2 + N + 1.
    """
    # Initialize parameters.
    n_init = 5
    left_threshold_init = -0.5
    right_threshold_init = 0.5
    # Create widgets for N and thresholds.
    n_slider, n_box = mtumsuti.build_widget_control(
        name="N",
        description="number of points",
        min_val=1,
        max_val=8,
        step=1,
        initial_value=n_init,
        is_float=False,
    )
    left_slider, left_box = mtumsuti.build_widget_control(
        name="left",
        description="left boundary",
        min_val=-1.5,
        max_val=1.5,
        step=0.1,
        initial_value=left_threshold_init,
        is_float=True,
    )
    right_slider, right_box = mtumsuti.build_widget_control(
        name="right",
        description="right boundary",
        min_val=-1.5,
        max_val=1.5,
        step=0.1,
        initial_value=right_threshold_init,
        is_float=True,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _draw_positive_intervals,
        {
            "n": n_slider,
            "left_threshold": left_slider,
            "right_threshold": right_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                n_box,
                left_box,
                right_box,
                output,
            ]
        )
    )


# #############################################################################
# Cell 6: Dichotomy Explorer - Convex Sets
# #############################################################################


def _get_convex_hull_points(points: np.ndarray, selected_indices: List[int]) -> np.ndarray:
    """
    Get convex hull of selected points.

    :param points: All points as array of shape (N, 2)
    :param selected_indices: Indices of selected points
    :return: Array of hull vertices in order
    """
    if len(selected_indices) < 3:
        # Not enough points for a hull, return the points themselves.
        return points[selected_indices]
    from scipy.spatial import ConvexHull
    selected_points = points[selected_indices]
    hull = ConvexHull(selected_points)
    return selected_points[hull.vertices]


def _point_in_convex_hull(point: np.ndarray, hull_points: np.ndarray) -> bool:
    """
    Check if a point is inside a convex hull.

    :param point: Point to check as array of shape (2,)
    :param hull_points: Hull vertices as array of shape (M, 2)
    :return: True if point is inside hull
    """
    if len(hull_points) < 3:
        # Check if point matches any hull point for degenerate cases.
        return any(np.allclose(point, hp) for hp in hull_points)
    # Use cross product method to check if point is inside convex polygon.
    n = len(hull_points)
    for i in range(n):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % n]
        # Vector from p1 to p2.
        edge = p2 - p1
        # Vector from p1 to point.
        to_point = point - p1
        # Cross product (2D).
        cross = edge[0] * to_point[1] - edge[1] * to_point[0]
        if cross < -1e-10:  # Point is on right side of edge.
            return False
    return True


def _get_target_classification_convex(n: int, seed: int) -> np.ndarray:
    """
    Get random target classification for convex sets based on seed.

    :param n: Number of points
    :param seed: Random seed for generating assignment
    :return: Array of classifications (+1 or -1)
    """
    # Generate random target classification using seed.
    np.random.seed(seed)
    target = np.random.choice([-1, 1], size=n)
    return target


def _find_convex_hull_for_target(
    points: np.ndarray, target: np.ndarray
) -> List[int]:
    """
    Find which points to select to achieve target classification.

    :param points: Array of shape (N, 2) containing point coordinates
    :param target: Target classification array
    :return: List of indices of points to select
    """
    # For convex sets, we simply select all points labeled +1.
    # Their convex hull will include all +1 points and potentially some -1 points,
    # but for points on a circle, selecting the +1 points gives the correct hull.
    selected_indices = [i for i, label in enumerate(target) if label == 1]
    return selected_indices


def _draw_convex_sets(
    n: int, target_idx: int, selected_indices_state: Dict
) -> None:
    """
    Draw 2D plot with N points arranged in circle and convex hull.

    :param n: Number of points
    :param target_idx: Target dichotomy index
    :param selected_indices_state: Dictionary containing current selected indices
    """
    # Create figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Create points arranged in a circle.
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    # Get target classification.
    target_classification = _get_target_classification_convex(n, target_idx)
    # Get current selected indices.
    selected_indices = selected_indices_state["value"]
    # Classify points based on whether they're in convex hull.
    if len(selected_indices) >= 3:
        hull_points = _get_convex_hull_points(points, selected_indices)
        current_classification = np.array(
            [
                1 if _point_in_convex_hull(point, hull_points) else -1
                for point in points
            ]
        )
    elif len(selected_indices) > 0:
        # Degenerate case: selected points are +1.
        current_classification = np.full(n, -1)
        current_classification[selected_indices] = 1
        hull_points = points[selected_indices]
    else:
        # No points selected.
        current_classification = np.full(n, -1)
        hull_points = np.array([])
    # Check if current matches target.
    match = np.array_equal(current_classification, target_classification)
    # Plot points on left subplot.
    for i, (point, target_class, current_class) in enumerate(
        zip(points, target_classification, current_classification)
    ):
        # Fill color based on target classification.
        fill_color = "blue" if target_class == 1 else "red"
        # Edge color based on current classification.
        edge_color = "blue" if current_class == 1 else "red"
        ax1.scatter(
            point[0],
            point[1],
            c=fill_color,
            s=200,
            edgecolors=edge_color,
            linewidths=3,
            zorder=3,
        )
        ax1.text(
            point[0] * 1.15,
            point[1] * 1.15,
            f"P{i+1}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    # Plot convex hull if it exists.
    if len(hull_points) >= 3:
        # Close the hull by repeating first point.
        hull_closed = np.vstack([hull_points, hull_points[0]])
        hull_color = "green" if match else "orange"
        ax1.fill(
            hull_closed[:, 0],
            hull_closed[:, 1],
            color=hull_color,
            alpha=0.2,
            label="Convex hull",
        )
        ax1.plot(
            hull_closed[:, 0], hull_closed[:, 1], color=hull_color, linewidth=2
        )
    elif len(hull_points) > 0:
        # Draw selected points for degenerate case.
        marker_color = "green" if match else "orange"
        ax1.scatter(
            hull_points[:, 0],
            hull_points[:, 1],
            marker="x",
            s=100,
            c=marker_color,
            linewidths=2,
            label="Selected",
        )
    # Format left subplot.
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlabel("x1", fontsize=12)
    ax1.set_ylabel("x2", fontsize=12)
    title = "Convex Sets - Target Dichotomy"
    if match:
        title += " - MATCH!"
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
    # Only show legend if there are labeled elements.
    if len(hull_points) > 0:
        ax1.legend()
    ax1.set_aspect("equal")
    # Create text content for right subplot.
    text_content = "Comment:\n"
    text_content += "Convex sets: Any subset of points can be selected,\n"
    text_content += "and all points inside their convex hull are +1.\n\n"
    text_content += "Growth function is EXPONENTIAL: m_H(N) = 2^N.\n\n"
    text_content += "Visualization:\n"
    text_content += "  • Fill color = Target assignment\n"
    text_content += "  • Border color = Convex hull assignment\n"
    text_content += "  • When colors match, classification is correct!\n\n"
    text_content += "Key observation:\n"
    text_content += "- Every dichotomy is achievable!\n"
    text_content += "- For any labeling, select all +1 points and take\n"
    text_content += "  their convex hull\n"
    text_content += "- Use 'Find Solution' to see the correct hull\n"
    text_content += "\n"
    text_content += "-" * 50 + "\n\n"
    text_content += f"N (number of points):  {n}\n"
    text_content += f"Selected points:       {len(selected_indices)}\n"
    text_content += f"m_H(N):                {2**n}\n"
    text_content += f"2^N:                   {2**n}\n\n"
    text_content += "Target Classification:\n"
    for i, classification in enumerate(target_classification):
        sign = "+1" if classification == 1 else "-1"
        text_content += f"  P{i+1}:                {sign}\n"
    text_content += "\n"
    text_content += "Current Classification:\n"
    for i, classification in enumerate(current_classification):
        sign = "+1" if classification == 1 else "-1"
        text_content += f"  P{i+1}:                {sign}\n"
    text_content += "\n"
    if match:
        text_content += "STATUS: MATCH! Classification achieved!\n"
    else:
        text_content += "STATUS: Use 'Find Solution' to achieve target.\n"
    text_content += "\n"
    text_content += "Selected point indices: " + str(selected_indices) + "\n"
    # Add text box to right subplot.
    ax2.axis("off")
    mtumsuti.add_fitted_text_box(ax2, text_content)
    plt.tight_layout()
    plt.show()


def cell6_dichotomy_explorer_convex_sets() -> None:
    """
    Create interactive dichotomy explorer for convex sets.

    Demonstrates exponential growth function m_H(N) = 2^N.
    """
    # Initialize parameters.
    n_init = 5
    seed_init = 0
    # Store current selected indices.
    selected_indices_state = {"value": []}
    # Create widgets for N.
    n_slider, n_box = mtumsuti.build_widget_control(
        name="N",
        description="number of points",
        min_val=3,
        max_val=8,
        step=1,
        initial_value=n_init,
        is_float=False,
    )
    # Create seed slider for random target generation.
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=seed_init,
        is_float=False,
    )
    # Create "Find Solution" button.
    find_button = ipywidgets.Button(
        description="Find Solution",
        button_style="info",
        tooltip="Find convex hull that achieves target",
    )

    def on_n_change(change):
        """Reset selected indices when N changes."""
        selected_indices_state["value"] = []

    def on_find_click(b):
        """Find solution for current target."""
        n = n_slider.value
        seed = seed_slider.value
        # Create points.
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        # Get target.
        target = _get_target_classification_convex(n, seed)
        # Find solution.
        selected_indices = _find_convex_hull_for_target(points, target)
        selected_indices_state["value"] = selected_indices

    n_slider.observe(on_n_change, names="value")
    find_button.on_click(on_find_click)
    # Create interactive output.
    output = ipywidgets.interactive_output(
        lambda n, seed: _draw_convex_sets(n, seed, selected_indices_state),
        {
            "n": n_slider,
            "seed": seed_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                n_box,
                seed_box,
                find_button,
                output,
            ]
        )
    )
