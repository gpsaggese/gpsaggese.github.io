"""
Utility functions for the entailment, implication, and inference lesson.

Stays on the lecture's own smallest examples (rain and wet ground, and
`x = 0` implies `x*y = 0`) to make the model-theoretic definitions concrete:
- Models, possible worlds, and satisfaction.
- Entailment as model inclusion, decided by model checking.
- The same definition applied to a non-Boolean world.
- Implication vs entailment vs inference, on one running example.

Import as:

import msml610.tutorials.L03_knowledge_representation.L03_01_entailment_implication_inference_utils as mtlkrl0eiiu
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import ipywidgets
import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sympy
from IPython.display import clear_output, display

import helpers.hnotebook as hnotebo
import helpers.htutorial as htutori

_LOG = logging.getLogger(__name__)

# #############################################################################
# Colors and display constants
# #############################################################################

# Colors used to shade a model set inside a model table.
_COLOR_PRIMARY = "#3182bd"
_COLOR_SECONDARY = "#e6550d"
# Named models for the 2-variable rain/wet-ground world, in the lecture's own
# order: `m1 = (T, T)`, `m2 = (T, F)`, `m3 = (F, T)`, `m4 = (F, F)`.
_RAIN_WETGROUND_WORLDS = ((True, True), (True, False), (False, True), (False, False))


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Propositional model checking, by direct substitution
# #############################################################################


def format_sentence(sentence: sympy.Basic) -> str:
    """
    Render a `sympy` sentence with an infix `=>` for implication.

    `sympy`'s default printer spells out `Implies(A, B)`, which is harder to
    read than the lecture's own `A => B` notation.

    :param sentence: sentence to render
    :return: string form, e.g., `Rain => WetGround`
    """
    if isinstance(sentence, sympy.Implies):
        antecedent, consequent = sentence.args
        result = "%s => %s" % (
            format_sentence(antecedent),
            format_sentence(consequent),
        )
    else:
        result = sympy.sstr(sentence)
    return result


def enumerate_assignments(n_vars: int) -> np.ndarray:
    """
    Enumerate every truth assignment over `n_vars` propositional variables.

    Row `i` holds the binary expansion of `i`, so the rows are exactly the
    $2^n$ candidate worlds a model checker has to examine.

    :param n_vars: number of variables
    :return: boolean array of shape `(2 ** n_vars, n_vars)`
    """
    index = np.arange(1 << n_vars, dtype=np.int64)
    bits = ((index[:, None] >> np.arange(n_vars)) & 1).astype(bool)
    return bits


def satisfying_mask(
    sentences: Sequence[sympy.Basic],
    var_order: Sequence[sympy.Symbol],
    bits: np.ndarray,
) -> np.ndarray:
    """
    Mark the assignments in `bits` that satisfy every sentence.

    This is the model-checking algorithm applied literally: for each
    candidate world, substitute its truth values into every sentence and
    check that all of them come out true. A `KB` with no sentences is
    satisfied vacuously by every world.

    :param sentences: sentences forming the knowledge base
    :param var_order: variables, in the column order of `bits`
    :param bits: assignments from `enumerate_assignments()`
    :return: boolean mask of shape `(2 ** n_vars,)`, one entry per assignment
    """
    n_models = bits.shape[0]
    mask = np.zeros(n_models, dtype=bool)
    for row in range(n_models):
        subs = [(var, bool(bits[row, i])) for i, var in enumerate(var_order)]
        mask[row] = all(bool(sentence.subs(subs)) for sentence in sentences)
    return mask


def entailment_verdict(kb_mask: np.ndarray, alpha_mask: np.ndarray) -> str:
    """
    Decide entailment by comparing the two model sets.

    $KB \\models \\alpha$ holds exactly when $M(KB) \\subseteq M(\\alpha)$.

    :param kb_mask: mask marking $M(KB)$
    :param alpha_mask: mask marking $M(\\alpha)$
    :return: one of `entailed`, `contradicted`, `unknown`, `inconsistent`
    """
    if not kb_mask.any():
        # No model satisfies the `KB`, so every query is vacuously entailed.
        verdict = "inconsistent"
    elif bool(np.all(alpha_mask[kb_mask])):
        verdict = "entailed"
    elif not bool(np.any(alpha_mask[kb_mask])):
        verdict = "contradicted"
    else:
        verdict = "unknown"
    return verdict


# #############################################################################
# Drawing helpers
# #############################################################################


def comment_panel(ax: matplotlib.axes.Axes, text: str) -> None:
    """
    Render a comment panel showing the current variable state.

    :param ax: axes to draw on
    :param text: comment text
    """
    ax.axis("off")
    ax.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    htutori.add_fitted_text_box(ax, text, max_fontsize=12, min_fontsize=7)


def make_param_info(descriptions: Dict[str, str]) -> ipywidgets.HTML:
    """
    Build a styled HTML info box describing the controls of a cell.

    :param descriptions: map from parameter name to its description text
    :return: styled HTML widget ready for display
    """
    body = "\n".join(
        "<b>%s</b>: %s<br>" % (name, desc) for name, desc in descriptions.items()
    )
    html = (
        "<div style='background:#f5f5f5; padding:10px 14px; "
        "border-radius:4px; border-left:3px solid #4682b4; "
        "font-size:13px; line-height:1.8'>"
        "%s"
        "</div>"
    ) % body
    return ipywidgets.HTML(html)


def draw_model_table(
    ax: matplotlib.axes.Axes,
    bits: np.ndarray,
    var_names: Sequence[str],
    primary_mask: np.ndarray,
    *,
    primary_name: str = "KB",
    secondary_mask: Optional[np.ndarray] = None,
    secondary_name: str = "alpha",
    row_names: Optional[Sequence[str]] = None,
    title: str = "Model table",
) -> None:
    """
    Draw the truth table over the model variables, shading the model sets.

    Rows in the primary set (e.g., $M(KB)$) get a blue wash, and rows in the
    optional secondary set (e.g., $M(\\alpha)$) get a dashed orange outline,
    so inclusion or a counterexample row is visible at a glance.

    :param ax: axes to draw on
    :param bits: assignments from `enumerate_assignments()`
    :param var_names: column headers, one per variable
    :param primary_mask: mask marking the primary model set
    :param primary_name: header used for the primary indicator column
    :param secondary_mask: optional mask marking a second model set
    :param secondary_name: header used for the secondary indicator column
    :param row_names: optional row labels, e.g., `m1`, `m2`, ...
    :param title: panel title
    """
    n_models, n_vars = bits.shape
    columns = [bits[:, i].astype(int) for i in range(n_vars)]
    headers = list(var_names)
    columns.append(np.where(primary_mask, 2, 3))
    headers.append("M(%s)" % primary_name)
    if secondary_mask is not None:
        columns.append(np.where(secondary_mask, 4, 5))
        headers.append("M(%s)" % secondary_name)
    image = np.stack(columns, axis=1)
    cmap = mcolors.ListedColormap(
        [
            "#f7f7f7",  # 0: False
            "#c6dbef",  # 1: True
            _COLOR_PRIMARY,  # 2: in the primary set
            "#ffffff",  # 3: outside the primary set
            _COLOR_SECONDARY,  # 4: in the secondary set
            "#ffffff",  # 5: outside the secondary set
        ]
    )
    ax.imshow(image, cmap=cmap, vmin=0, vmax=5, aspect="auto")
    # Wash the rows in the primary set so the shaded subset reads as a set.
    for row_index in np.flatnonzero(primary_mask):
        ax.add_patch(
            mpatches.Rectangle(
                (-0.5, row_index - 0.5),
                image.shape[1],
                1.0,
                facecolor=_COLOR_PRIMARY,
                alpha=0.18,
                edgecolor="none",
            )
        )
    if secondary_mask is not None:
        for row_index in np.flatnonzero(secondary_mask):
            ax.add_patch(
                mpatches.Rectangle(
                    (-0.5, row_index - 0.5),
                    image.shape[1],
                    1.0,
                    fill=False,
                    edgecolor=_COLOR_SECONDARY,
                    linestyle="--",
                    linewidth=1.2,
                )
            )
    for row_index in range(n_models):
        for col_index in range(n_vars):
            ax.text(
                col_index,
                row_index,
                "T" if bits[row_index, col_index] else "F",
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set_xticks(range(len(headers)))
    ax.set_xticklabels(headers, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(
        row_names if row_names is not None else ["" for _ in range(n_models)],
        fontsize=8,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(
        "one row per model\n\n"
        "blue wash: row in the primary set, dashed outline: row in the "
        "secondary set",
        fontsize=9,
    )
    ax.grid(False)


def draw_text_panel(
    ax: matplotlib.axes.Axes,
    lines: Sequence[Tuple[str, str, str]],
    *,
    title: str,
) -> None:
    """
    Draw a panel of left-aligned monospace lines, one color per line.

    :param ax: axes to draw on
    :param lines: triples of `(text, color, fontweight)`
    :param title: panel title
    """
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")
    step = 1.0 / (len(lines) + 1)
    for index, (text, color, weight) in enumerate(lines):
        ax.text(
            0.02,
            0.97 - index * step,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=10,
            color=color,
            fontweight=weight,
        )


# #############################################################################
# Cell 1.1: Possible worlds, models, and satisfaction
# #############################################################################


def rain_wetground_models() -> Tuple[List[sympy.Symbol], np.ndarray]:
    """
    Return the variable order and the 4 named models of the rain world.

    :return: tuple of
        - `[Rain, WetGround]`
        - boolean array of shape `(4, 2)`, rows `m1`, `m2`, `m3`, `m4`
    """
    rain, wet_ground = sympy.symbols("Rain WetGround")
    var_order = [rain, wet_ground]
    bits = np.array(_RAIN_WETGROUND_WORLDS, dtype=bool)
    return var_order, bits


def cell1_1_models_and_satisfaction(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Shade $M(\\alpha)$ over the 4 models of the rain and wet-ground world.

    Interactive controls (ipywidgets):
    - `alpha`: the sentence whose model set is shaded

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (16, 6)
    var_order, bits = rain_wetground_models()
    rain, wet_ground = var_order
    row_names = ["m1", "m2", "m3", "m4"]
    alpha_options = {
        "Rain": rain,
        "WetGround": wet_ground,
        "Rain and WetGround": sympy.And(rain, wet_ground),
        "Rain => WetGround": sympy.Implies(rain, wet_ground),
        "not Rain": sympy.Not(rain),
    }
    alpha_dropdown = ipywidgets.Dropdown(
        options=list(alpha_options.keys()),
        value="Rain",
        description="alpha:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            label = alpha_dropdown.value
            alpha = alpha_options[label]
            alpha_mask = satisfying_mask([alpha], var_order, bits)
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            # Panel 1: the 4 named models, with M(alpha) shaded.
            draw_model_table(
                ax1,
                bits,
                ["Rain", "WetGround"],
                alpha_mask,
                primary_name="alpha",
                row_names=row_names,
                title="M(alpha) over the 4 models",
            )
            # Panel 2: comments naming each model and whether alpha holds.
            per_model = "\n  ".join(
                "%s = (Rain=%s, WetGround=%s): alpha is %s"
                % (
                    row_names[i],
                    "T" if bits[i, 0] else "F",
                    "T" if bits[i, 1] else "F",
                    "true" if alpha_mask[i] else "false",
                )
                for i in range(4)
            )
            text = (
                "Parameters:\n"
                "  alpha: %s\n"
                "  alpha sentence: %s\n\n"
                "|M(alpha)|: %d out of 4\n\n"
                "Per-model satisfaction:\n"
                "  %s"
                % (
                    label,
                    format_sentence(alpha),
                    int(alpha_mask.sum()),
                    per_model,
                )
            )
            comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "alpha": "the sentence whose model set <code>M(alpha)</code> is "
            "shaded; a model 'satisfies' alpha when alpha is true in that "
            "one fixed row",
        }
    )
    alpha_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [alpha_dropdown], layout=ipywidgets.Layout(padding="0px 8px 0px 0px")
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.1: Entailment as model inclusion, by model checking
# #############################################################################


def cell2_1_entailment_model_checking(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Decide `KB |= alpha` over the rain world by checking model inclusion.

    Interactive controls (ipywidgets):
    - `Rain`, `Rain => WetGround`: checkboxes that add or remove each
      sentence from the `KB`
    - `alpha`: the query sentence

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (16, 6)
    var_order, bits = rain_wetground_models()
    rain, wet_ground = var_order
    row_names = ["m1", "m2", "m3", "m4"]
    alpha_options = {
        "WetGround": wet_ground,
        "not WetGround": sympy.Not(wet_ground),
        "Rain or WetGround": sympy.Or(rain, wet_ground),
    }
    rain_checkbox = ipywidgets.Checkbox(value=True, description="Rain")
    rule_checkbox = ipywidgets.Checkbox(
        value=True, description="Rain => WetGround"
    )
    alpha_dropdown = ipywidgets.Dropdown(
        options=list(alpha_options.keys()),
        value="WetGround",
        description="alpha:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            kb_sentences = []
            if rain_checkbox.value:
                kb_sentences.append(rain)
            if rule_checkbox.value:
                kb_sentences.append(sympy.Implies(rain, wet_ground))
            label = alpha_dropdown.value
            alpha = alpha_options[label]
            kb_mask = satisfying_mask(kb_sentences, var_order, bits)
            alpha_mask = satisfying_mask([alpha], var_order, bits)
            verdict = entailment_verdict(kb_mask, alpha_mask)
            counterexample_mask = kb_mask & ~alpha_mask
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            draw_model_table(
                ax1,
                bits,
                ["Rain", "WetGround"],
                kb_mask,
                primary_name="KB",
                secondary_mask=alpha_mask,
                secondary_name=label,
                row_names=row_names,
                title="M(KB) against M(alpha)",
            )
            kb_text = (
                "\n  ".join(format_sentence(s) for s in kb_sentences)
                if kb_sentences
                else "(empty: no constraints)"
            )
            counter_text = (
                ", ".join(
                    row_names[i] for i in np.flatnonzero(counterexample_mask)
                )
                if counterexample_mask.any()
                else "none"
            )
            text = (
                "KB sentences:\n"
                "  %s\n\n"
                "alpha: %s\n\n"
                "Model sets:\n"
                "  |M(KB)|: %d\n"
                "  |M(alpha)|: %d\n"
                "  counterexample rows: %s\n\n"
                "Verdict: KB %s alpha"
                % (
                    kb_text,
                    format_sentence(alpha),
                    int(kb_mask.sum()),
                    int(alpha_mask.sum()),
                    counter_text,
                    verdict,
                )
            )
            comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            'KB: "Rain", "Rain => WetGround"': "toggle each sentence in or "
            "out of the <code>KB</code>; dropping the rule leaves "
            "<code>KB</code> unable to entail <code>WetGround</code>",
            "alpha": "the query sentence checked against <code>M(KB)</code>",
        }
    )
    rain_checkbox.observe(update_plot, names="value")
    rule_checkbox.observe(update_plot, names="value")
    alpha_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [rain_checkbox, rule_checkbox, alpha_dropdown],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.2: The same definition on a non-Boolean world
# #############################################################################


def draw_xy_world(
    ax: matplotlib.axes.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    alpha_mask: np.ndarray,
    beta_mask: np.ndarray,
    *,
    alpha_name: str,
    beta_name: str,
    title: str,
) -> None:
    """
    Scatter every `(x, y)` pair, shading $M(\\alpha)$ and outlining $M(\\beta)$.

    :param ax: axes to draw on
    :param x_values: flattened `x` coordinate of every model
    :param y_values: flattened `y` coordinate of every model
    :param alpha_mask: mask marking $M(\\alpha)$
    :param beta_mask: mask marking $M(\\beta)$
    :param alpha_name: label for $\\alpha$
    :param beta_name: label for $\\beta$
    :param title: panel title
    """
    counterexample_mask = alpha_mask & ~beta_mask
    both_mask = alpha_mask & beta_mask
    only_beta_mask = ~alpha_mask & beta_mask
    neither_mask = ~alpha_mask & ~beta_mask
    ax.scatter(
        x_values[neither_mask],
        y_values[neither_mask],
        c="#d9d9d9",
        s=40,
        label="neither",
    )
    ax.scatter(
        x_values[only_beta_mask],
        y_values[only_beta_mask],
        facecolors="none",
        edgecolors=_COLOR_SECONDARY,
        s=60,
        linewidths=1.5,
        label="M(%s) only" % beta_name,
    )
    ax.scatter(
        x_values[both_mask],
        y_values[both_mask],
        c=_COLOR_PRIMARY,
        edgecolors=_COLOR_SECONDARY,
        linewidths=1.5,
        s=70,
        label="M(%s) and M(%s)" % (alpha_name, beta_name),
    )
    ax.scatter(
        x_values[counterexample_mask],
        y_values[counterexample_mask],
        c="#de2d26",
        marker="x",
        s=90,
        linewidths=2.5,
        label="counterexample",
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")


def cell2_2_nonboolean_world(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Check `alpha |= beta` over integer pairs `(x, y)` instead of truth values.

    Interactive controls (ipywidgets):
    - `alpha`: the query sentence, checked against the fixed `beta: x*y = 0`
    - `range`: how far `x` and `y` range from 0

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (16, 6)
    alpha_options: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "x = 0": lambda x, _y: x == 0,
        "y = 0": lambda _x, y: y == 0,
        "x = 1": lambda x, _y: x == 1,
    }
    alpha_dropdown = ipywidgets.Dropdown(
        options=list(alpha_options.keys()),
        value="x = 0",
        description="alpha:",
        style={"description_width": "initial"},
    )
    range_slider = ipywidgets.IntSlider(
        value=3, min=2, max=5, step=1, description="range:"
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            label = alpha_dropdown.value
            r = range_slider.value
            values = np.arange(-r, r + 1)
            x_grid, y_grid = np.meshgrid(values, values)
            x_values = x_grid.ravel()
            y_values = y_grid.ravel()
            alpha_mask = alpha_options[label](x_values, y_values)
            beta_mask = x_values * y_values == 0
            counterexamples = alpha_mask & ~beta_mask
            verdict = "entailed" if not counterexamples.any() else "not entailed"
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            draw_xy_world(
                ax1,
                x_values,
                y_values,
                alpha_mask,
                beta_mask,
                alpha_name=label,
                beta_name="x*y = 0",
                title="alpha: %s, beta: x*y = 0" % label,
            )
            counter_text = (
                ", ".join(
                    "(%d, %d)" % (x_values[i], y_values[i])
                    for i in np.flatnonzero(counterexamples)[:6]
                )
                if counterexamples.any()
                else "none"
            )
            text = (
                "Parameters:\n"
                "  alpha: %s\n"
                "  x, y range: [-%d, %d]\n\n"
                "Model counts:\n"
                "  |M(alpha)|: %d\n"
                "  |M(beta)|: %d\n"
                "  counterexamples: %s\n\n"
                "Verdict: alpha %s beta"
                % (
                    label,
                    r,
                    r,
                    int(alpha_mask.sum()),
                    int(beta_mask.sum()),
                    counter_text,
                    verdict,
                )
            )
            comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "alpha": "the query sentence; a model here is a pair "
            "<code>(x, y)</code>, not a truth assignment",
            "range": "how far <code>x</code> and <code>y</code> range from "
            "0; the verdict does not change as the range grows",
        }
    )
    alpha_dropdown.observe(update_plot, names="value")
    range_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [alpha_dropdown, range_slider],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 3.1: Implication, entailment, and inference: three views
# #############################################################################


def _implication_view(
    var_order: Sequence[sympy.Symbol], bits: np.ndarray, row_names: Sequence[str]
) -> List[Tuple[str, str, str]]:
    """
    Build the text panel reading `Rain => WetGround` as one sentence.

    :param var_order: `[Rain, WetGround]`
    :param bits: the 4 named models
    :param row_names: `m1`, ..., `m4`
    :return: triples of `(text, color, fontweight)`
    """
    rain, wet_ground = var_order
    implication = sympy.Implies(rain, wet_ground)
    lines = [
        ("The sentence itself, one object inside the logic:", "black", "bold"),
        ("  %s" % format_sentence(implication), "black", "normal"),
        ("", "black", "normal"),
        ("True unless Rain is true and WetGround is false:", "black", "bold"),
    ]
    for i in range(4):
        rain_val, wet_val = bits[i]
        holds = not (rain_val and not wet_val)
        color = "#b2182b" if not holds else "#006d2c"
        lines.append(
            (
                "  %s = (Rain=%s, WetGround=%s): implication is %s"
                % (
                    row_names[i],
                    "T" if rain_val else "F",
                    "T" if wet_val else "F",
                    "true" if holds else "false",
                ),
                color,
                "bold" if not holds else "normal",
            )
        )
    lines.append(("", "black", "normal"))
    lines.append(
        (
            "Note: true even when Rain is false (m3, m4).",
            "#b2182b",
            "bold",
        )
    )
    return lines


def _inference_view(forward: bool) -> List[Tuple[str, str, str]]:
    """
    Build the step-by-step proof trace, forward or backward.

    :param forward: `True` for forward chaining, `False` for backward
    :return: triples of `(text, color, fontweight)`
    """
    if forward:
        lines = [
            ("Forward chaining: start from facts, apply rules", "black", "bold"),
            ("", "black", "normal"),
            ("1. Fact:  Rain", "black", "normal"),
            ("2. Rule:  Rain => WetGround", "black", "normal"),
            ("3. Modus ponens on 1, 2:", "#2166ac", "bold"),
            ("   WetGround", "#2166ac", "normal"),
            ("", "black", "normal"),
            ("Conclusion: WetGround", "#b2182b", "bold"),
        ]
    else:
        lines = [
            ("Backward chaining: start from the goal, work back", "black", "bold"),
            ("", "black", "normal"),
            ("Goal:      WetGround", "black", "normal"),
            ("Rule that matches the goal:", "black", "normal"),
            ("   Rain => WetGround", "black", "normal"),
            ("New subgoal:", "#2166ac", "bold"),
            ("   Rain", "#2166ac", "normal"),
            ("Rain is a known fact: subgoal proved", "#2166ac", "normal"),
            ("", "black", "normal"),
            ("Conclusion: WetGround is proved", "#b2182b", "bold"),
        ]
    return lines


def cell3_1_three_views(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Separate implication, entailment, and inference on the rain example.

    Interactive controls (ipywidgets):
    - `view`: which of the four panels to show

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (15, 6)
    var_order, bits = rain_wetground_models()
    rain, wet_ground = var_order
    row_names = ["m1", "m2", "m3", "m4"]
    definitions = {
        "implication": "a connective inside one sentence (syntax)",
        "entailment": "truth that follows in every model (semantics)",
        "inference (forward)": "a data-driven proof procedure (computation)",
        "inference (backward)": "a goal-driven proof procedure (computation)",
    }
    view_dropdown = ipywidgets.Dropdown(
        options=list(definitions.keys()),
        value="implication",
        description="view:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            view = view_dropdown.value
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            if view == "implication":
                draw_text_panel(
                    ax1,
                    _implication_view(var_order, bits, row_names),
                    title="Implication (syntactic)",
                )
            elif view == "entailment":
                kb_sentences = [rain, sympy.Implies(rain, wet_ground)]
                kb_mask = satisfying_mask(kb_sentences, var_order, bits)
                alpha_mask = satisfying_mask([wet_ground], var_order, bits)
                draw_model_table(
                    ax1,
                    bits,
                    ["Rain", "WetGround"],
                    kb_mask,
                    primary_name="KB",
                    secondary_mask=alpha_mask,
                    secondary_name="WetGround",
                    row_names=row_names,
                    title="Entailment (semantic)",
                )
            elif view == "inference (forward)":
                draw_text_panel(
                    ax1,
                    _inference_view(forward=True),
                    title="Inference (computational, forward)",
                )
            else:
                draw_text_panel(
                    ax1,
                    _inference_view(forward=False),
                    title="Inference (computational, backward)",
                )
            text = (
                "Parameters:\n"
                "  view: %s\n\n"
                "Definition:\n"
                "  %s\n\n"
                "Where it lives:\n"
                "  implication: inside one sentence\n"
                "  entailment: across all models\n"
                "  inference: inside an algorithm\n\n"
                "Same example throughout:\n"
                "  KB = {Rain, Rain => WetGround}, alpha = WetGround"
                % (view, definitions[view])
            )
            comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "view": "<code>implication</code> is syntax inside one "
            "sentence, <code>entailment</code> is truth across all models, "
            "<code>inference</code> is a procedure that tries to track it, "
            "run forward from facts or backward from the goal",
        }
    )
    view_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [view_dropdown], layout=ipywidgets.Layout(padding="0px 8px 0px 0px")
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))
