"""
Utilities for causal discovery notebook.

Import as:

import msml610.tutorials.L10_causal_discovery.L10_2_causal_discovery_utils as mtlcdl2cdu
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

# TODO(ai_gp): Use import
from ipywidgets import Output, HBox, VBox, ToggleButtons, Dropdown, Checkbox
from IPython.display import display, clear_output
import networkx as nx
import scipy.stats

import helpers.hnotebook as hnotebo
import helpers.htutorial as htutori

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(
        notebook_log, utils_log=_LOG, set_all_loggers_to_print=True
    )


# #############################################################################
# Cell 1: Correlation vs. Causation
# #############################################################################


def cell1_correlation_vs_causation():
    """
    Interactive widget showing correlation vs causation problem.
    """

    def plot_causal_structures(seed, correlation_strength, _intervention_mode):
        """
        Plot two DAGs with identical correlation but different causation.
        """
        _, axes = plt.subplots(1, 3, figsize=(18, 5))
        # Generate correlated data.
        np.random.seed(seed)
        n_samples = 200
        noise = np.random.normal(0, 1 - correlation_strength, n_samples)
        X = np.random.normal(0, 1, n_samples)
        # Chain: X -> Y.
        Y_chain = correlation_strength * X + noise
        # Reverse: Z -> Y -> X (same correlation structure).
        Y_reverse = correlation_strength * X + noise
        # Plot Chain: X -> Y.
        axes[0].scatter(X, Y_chain, alpha=0.6, s=30, color="steelblue")
        axes[0].set_xlabel("X", fontsize=12)
        axes[0].set_ylabel("Y", fontsize=12)
        axes[0].set_title("Chain: X -> Y", fontsize=13, fontweight="bold")
        corr = np.corrcoef(X, Y_chain)[0, 1]
        axes[0].text(
            0.05,
            0.95,
            f"Correlation: r = {corr:.2f}\n(Intervening on X changes Y)",
            transform=axes[0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=10,
        )
        # Plot Reverse: Z -> Y -> X (same correlation).
        axes[1].scatter(X, Y_reverse, alpha=0.6, s=30, color="coral")
        axes[1].set_xlabel("X", fontsize=12)
        axes[1].set_ylabel("Y", fontsize=12)
        axes[1].set_title("Reverse: Z -> Y -> X", fontsize=13, fontweight="bold")
        corr_rev = np.corrcoef(X, Y_reverse)[0, 1]
        axes[1].text(
            0.05,
            0.95,
            f"Correlation: r = {corr_rev:.2f}\n(Intervening on X has NO effect)",
            transform=axes[1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
            fontsize=10,
        )
        for ax in axes[:2]:
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.grid(True, alpha=0.3)
        # Comments panel.
        axes[2].axis("off")
        axes[2].set_title("Comments", fontsize=13, fontweight="bold")
        detail = (
            f"seed = {seed}\n"
            f"correlation (r) = {correlation_strength:.2f}\n\n"
            "KEY INSIGHT: same observational correlation,\n"
            "opposite causal implications.\n\n"
            "Without intervention data or additional\n"
            "assumptions, we cannot distinguish these\n"
            "structures from correlation alone."
        )
        htutori.add_fitted_text_box(
            axes[2], detail, max_fontsize=12, min_fontsize=9
        )
        plt.tight_layout()
        plt.show()

    # Create interactive widget.
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    correlation_slider, correlation_box = htutori.build_widget_control(
        name="r",
        description="Correlation (r)",
        min_val=0.3,
        max_val=0.99,
        step=0.05,
        initial_value=0.8,
        is_float=True,
    )
    mode_toggle = ToggleButtons(
        options=["Observational", "Interventional"],
        description="Mode:",
    )
    output = Output()

    def update(change):
        with output:
            clear_output(wait=True)
            plot_causal_structures(
                seed_slider.value, correlation_slider.value, mode_toggle.value
            )

    seed_slider.observe(update, names="value")
    correlation_slider.observe(update, names="value")
    mode_toggle.observe(update, names="value")
    display(
        VBox(
            [
                seed_box,
                HBox([correlation_box, mode_toggle]),
                output,
            ]
        )
    )
    update(None)


# #############################################################################
# Cell 2: Markov Equivalence
# #############################################################################
def cell2_markov_equivalence():
    """
    Show three indistinguishable DAG structures.
    """

    def plot_markov_equivalence(seed, sample_size):
        """
        Plot three Markov equivalent structures.
        """
        fig = plt.figure(figsize=(15, 8))
        # Create 3 subplots for DAGs + CI structure.
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        # Generate data from one structure (chain).
        np.random.seed(seed)
        Z = np.random.normal(0, 1, sample_size)
        Y = 0.8 * Z + np.random.normal(0, 0.5, sample_size)
        X = 0.8 * Y + np.random.normal(0, 0.5, sample_size)
        # Compute correlations.
        corr_XZ = np.corrcoef(X, Z)[0, 1]
        # Compute partial correlation X-Z given Y using residuals.
        reg_X_Y = scipy.stats.linregress(Y, X)
        residuals_X = X - (reg_X_Y.slope * Y + reg_X_Y.intercept)  # type: ignore
        reg_Z_Y = scipy.stats.linregress(Y, Z)
        residuals_Z = Z - (reg_Z_Y.slope * Y + reg_Z_Y.intercept)  # type: ignore
        corr_XZ_given_Y = np.corrcoef(residuals_X, residuals_Z)[0, 1]
        # Plot three DAG structures.
        structures = [
            ("Chain: X -> Y -> Z", [(0, 1), (1, 2)]),
            ("Reverse: Z -> Y -> X", [(2, 1), (1, 0)]),
            ("Common Cause: X <- Y -> Z", [(1, 0), (1, 2)]),
        ]
        for idx, (title, edges) in enumerate(structures):
            ax = fig.add_subplot(gs[0, idx])
            # Draw simple DAG.
            G = nx.DiGraph()
            G.add_nodes_from([0, 1, 2])
            G.add_edges_from(edges)
            pos = {0: (0, 0), 1: (1, 1), 2: (2, 0)}
            node_labels = {0: "X", 1: "Y", 2: "Z"}
            nx.draw_networkx_nodes(
                G, pos, node_color="lightblue", node_size=500, ax=ax
            )
            nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=20, width=2)
            nx.draw_networkx_labels(G, pos, node_labels, font_size=12, ax=ax)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")
        # Plot CI structure.
        ax_ci = fig.add_subplot(gs[1, :])
        ci_text = (
            "All three structures imply the SAME conditional independence:\n"
            "X _|_ Z | Y (X is independent of Z given Y)\n\n"
            "Why? Because Y blocks all paths between X and Z in all three\n"
            "structures."
        )
        ax_ci.text(
            0.5,
            0.5,
            ci_text,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
            transform=ax_ci.transAxes,
        )
        ax_ci.axis("off")
        # Plot correlation and conditional correlation.
        ax_corr = fig.add_subplot(gs[2, 0])
        ax_corr.bar(
            ["All 3 structures"],
            [corr_XZ],
            color="steelblue",
            alpha=0.7,
            width=0.3,
        )
        ax_corr.set_ylabel("Correlation X-Z", fontsize=10)
        ax_corr.set_title("Marginal Correlation", fontsize=11, fontweight="bold")
        ax_corr.set_ylim(-1, 1)
        ax_corr.grid(True, alpha=0.3, axis="y")
        ax_cond = fig.add_subplot(gs[2, 1])
        ax_cond.bar(
            ["All 3 structures"],
            [corr_XZ_given_Y],
            color="coral",
            alpha=0.7,
            width=0.3,
        )
        ax_cond.set_ylabel("Conditional Correlation X-Z|Y", fontsize=10)
        ax_cond.set_title(
            "Conditional Correlation", fontsize=11, fontweight="bold"
        )
        ax_cond.set_ylim(-1, 1)
        ax_cond.grid(True, alpha=0.3, axis="y")
        ax_n = fig.add_subplot(gs[2, 2])
        ax_n.axis("off")
        ax_n.set_title("Comments", fontsize=11, fontweight="bold")
        detail = (
            f"seed = {seed}\n"
            f"sample size N = {sample_size}\n\n"
            f"corr(X, Z) = {corr_XZ:.3f}\n"
            f"corr(X, Z | Y) = {corr_XZ_given_Y:.3f}\n\n"
            "All 3 structures have identical\n"
            "covariance matrices: no amount of\n"
            "data can tell them apart."
        )
        htutori.add_fitted_text_box(
            ax_n, detail, max_fontsize=10, min_fontsize=7
        )
        plt.suptitle(
            "Markov Equivalence: Three Indistinguishable Structures",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.show()

    # Interactive widget.
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    sample_slider, sample_box = htutori.build_widget_control(
        name="N",
        description="Sample Size (N)",
        min_val=50,
        max_val=5000,
        step=50,
        initial_value=100,
        is_float=False,
    )
    output = Output()

    def update(change):
        with output:
            clear_output(wait=True)
            plot_markov_equivalence(seed_slider.value, sample_slider.value)

    seed_slider.observe(update, names="value")
    sample_slider.observe(update, names="value")
    display(VBox([seed_box, sample_box, output]))
    update(None)


# #############################################################################
# Cell 3: Causal Effects via Intervention
# #############################################################################
def cell3_causal_effects():
    """
    Show why edge direction determines causal effect.
    """

    def plot_interventions(seed, intervention_strength, sample_size):
        """
        Plot counterfactual outcomes for three structures.
        """
        fig, axes = plt.subplots(1, 4, figsize=(19, 4))
        # Simulate three structures with intervention.
        np.random.seed(seed)
        # Chain: X -> Y -> Z.
        # Draw and discard to keep the random stream unchanged.
        np.random.normal(0, 1, sample_size)
        X_intervened = np.ones(sample_size) * intervention_strength
        Y_chain = 0.8 * X_intervened + np.random.normal(0, 0.3, sample_size)
        Z_chain = 0.8 * Y_chain + np.random.normal(0, 0.3, sample_size)
        effect_chain = np.mean(Z_chain) - 0  # baseline at 0.
        # Reverse: Z -> Y -> X (no effect on Z).
        effect_reverse = 0.0  # Intervening on X doesn't affect Z.
        # Common Cause: Y confounds both X and Z.
        # Draw and discard to keep the random stream unchanged.
        np.random.normal(0, 1, sample_size)
        np.random.normal(0, 0.3, sample_size)
        np.random.normal(0, 0.3, sample_size)
        effect_common = 0.0  # No direct effect on Z.
        effects = [effect_chain, effect_reverse, effect_common]
        titles = [
            "Chain: X -> Y -> Z\n(LARGE effect)",
            "Reverse: Z -> Y -> X\n(NO effect)",
            "Common Cause: Y confounds\n(NO direct effect)",
        ]
        colors = ["green", "red", "orange"]
        for idx, (ax, title, effect, color) in enumerate(
            zip(axes[:3], titles, effects, colors)
        ):
            ax.bar(
                ["Effect on Z"],
                [effect],
                color=color,
                alpha=0.7,
                width=0.3,
            )
            ax.set_ylabel("Causal Effect (delta Z)", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylim(-2, 3)
            ax.grid(True, alpha=0.3, axis="y")
            ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
            ax.text(
                0.5,
                effect + 0.1,
                f"{effect:.2f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )
        # Comments panel.
        axes[3].axis("off")
        axes[3].set_title("Comments", fontsize=12, fontweight="bold")
        detail = (
            f"seed = {seed}\n"
            f"intervention strength = {intervention_strength:.2f}\n"
            f"sample size N = {sample_size}\n\n"
            "KEY INSIGHT: edge direction determines\n"
            "whether an intervention works.\n"
            "Choosing the wrong DAG leads to\n"
            "ineffective or harmful interventions."
        )
        htutori.add_fitted_text_box(
            axes[3], detail, max_fontsize=11, min_fontsize=8
        )
        plt.suptitle(
            "Why Edge Direction Matters: Same Correlation, Different Effects",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    # Interactive widgets.
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    intervention_slider, intervention_box = htutori.build_widget_control(
        name="strength",
        description="Intervention Strength",
        min_val=0.0,
        max_val=3.0,
        step=0.1,
        initial_value=1.5,
        is_float=True,
    )
    sample_slider, sample_box = htutori.build_widget_control(
        name="N",
        description="Sample Size (N)",
        min_val=100,
        max_val=5000,
        step=100,
        initial_value=100,
        is_float=False,
    )
    output = Output()

    def update(change):
        with output:
            clear_output(wait=True)
            plot_interventions(
                seed_slider.value, intervention_slider.value, sample_slider.value
            )

    seed_slider.observe(update, names="value")
    intervention_slider.observe(update, names="value")
    sample_slider.observe(update, names="value")
    display(
        VBox(
            [
                seed_box,
                HBox([intervention_box, sample_box]),
                output,
            ]
        )
    )
    update(None)


# #############################################################################
# Cell 4: PC Algorithm
# #############################################################################
def cell4_pc_algorithm():
    """
    Show PC algorithm step-by-step.
    """

    def plot_pc_steps(alpha_threshold, test_type, speed):
        """
        Animate PC algorithm on small DAG.
        """
        fig, (ax_left, ax_right, ax_comments) = plt.subplots(
            1, 3, figsize=(19, 6)
        )
        # Simplified visualization of PC steps.
        steps = [
            ("Step 1: Test X1 _|_ X2", True, "Test 1/6"),
            ("Step 2: Test X1 _|_ X3", False, "Test 2/6"),
            ("Step 3: Orient V-structures", None, "Orientation"),
            ("Step 4: Final CPDAG", None, "Complete"),
        ]
        for step_idx, (step_desc, result, step_label) in enumerate(steps):
            ax_left.clear()
            # Draw evolving graph structure.
            G = nx.Graph()
            G.add_nodes_from([1, 2, 3])
            if step_idx == 0:
                G.add_edges_from([(1, 2), (1, 3), (2, 3)])
            elif step_idx == 1:
                G.add_edges_from([(1, 3), (2, 3)])
            else:
                G.add_edges_from([(1, 3), (2, 3)])
            pos = {1: (0, 0), 2: (2, 0), 3: (1, 1.5)}
            nx.draw_networkx_nodes(
                G, pos, node_color="lightblue", node_size=600, ax=ax_left
            )
            nx.draw_networkx_edges(G, pos, ax=ax_left, width=2)
            nx.draw_networkx_labels(G, pos, font_size=12, ax=ax_left)
            ax_left.set_title(
                "PC Algorithm Progression", fontsize=12, fontweight="bold"
            )
            ax_left.axis("off")
            # Right panel: Test details.
            ax_right.clear()
            test_text = (
                f"{step_desc}\n"
                f"Alpha Threshold: {alpha_threshold:.3f}\n"
                f"Test Type: {test_type}\n"
                f"p-value: {np.random.uniform(0, 1):.4f}\n"
                f"Separating Set: {step_label}"
            )
            ax_right.text(
                0.5,
                0.5,
                test_text,
                ha="center",
                va="center",
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
                transform=ax_right.transAxes,
            )
            ax_right.set_title("Current Test", fontsize=12, fontweight="bold")
            ax_right.axis("off")
        # Comments panel.
        ax_comments.axis("off")
        ax_comments.set_title("Comments", fontsize=12, fontweight="bold")
        detail = (
            f"alpha = {alpha_threshold:.3f}\n"
            f"test type = {test_type}\n"
            f"speed = {speed:.1f}x\n\n"
            "PC recovers the skeleton via CI\n"
            "tests, then orients v-structures.\n"
            "It is sound in the large-sample\n"
            "limit, but CI tests are\n"
            "underpowered in finite samples."
        )
        htutori.add_fitted_text_box(
            ax_comments, detail, max_fontsize=11, min_fontsize=8
        )
        plt.suptitle(
            "The PC Algorithm: Learning from Conditional Independence Tests",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    # Interactive widgets.
    alpha_slider, alpha_box = htutori.build_widget_control(
        name="alpha",
        description="Alpha (CI threshold)",
        min_val=0.001,
        max_val=0.2,
        step=0.01,
        initial_value=0.05,
        is_float=True,
    )
    test_dropdown = Dropdown(
        options=["Partial correlation", "Gaussian G-squared", "Conditional MI"],
        value="Partial correlation",
        description="Test Type:",
    )
    speed_slider, speed_box = htutori.build_widget_control(
        name="speed",
        description="Speed",
        min_val=0.5,
        max_val=2.0,
        step=0.1,
        initial_value=1.0,
        is_float=True,
    )
    output = Output()

    def update(change):
        with output:
            clear_output(wait=True)
            plot_pc_steps(
                alpha_slider.value, test_dropdown.value, speed_slider.value
            )

    alpha_slider.observe(update, names="value")
    test_dropdown.observe(update, names="value")
    speed_slider.observe(update, names="value")
    display(
        VBox(
            [
                HBox([alpha_box, test_dropdown, speed_box]),
                output,
            ]
        )
    )
    update(None)


# #############################################################################
# Cell 5: GES Score-Based Search
# #############################################################################
def cell5_ges_algorithm():
    """
    Show GES algorithm evolution.
    """

    def plot_ges_search(sample_size, regularization):
        """
        Plot GES search progress.
        """
        fig, (ax_left, ax_right, ax_comments) = plt.subplots(
            1, 3, figsize=(19, 5)
        )
        # Simulate BIC scores during search.
        iterations = np.arange(0, 15)
        bic_scores = -100 + 20 * np.sin(iterations / 3) - 0.5 * iterations
        # Left: DAG evolution.
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3, 4])
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        pos = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0)}
        nx.draw_networkx_nodes(
            G, pos, node_color="lightblue", node_size=500, ax=ax_left
        )
        nx.draw_networkx_edges(G, pos, ax=ax_left, width=2, arrowsize=15)
        nx.draw_networkx_labels(G, pos, font_size=11, ax=ax_left)
        ax_left.set_title(
            f"Current DAG Structure\n"
            f"N={sample_size}, Regularization={regularization:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax_left.axis("off")
        # Right: BIC trajectory.
        ax_right.plot(
            iterations,
            bic_scores,
            "o-",
            color="steelblue",
            linewidth=2,
            markersize=6,
        )
        ax_right.axvline(
            x=8, color="red", linestyle="--", label="Forward/Backward transition"
        )
        ax_right.set_xlabel("Iteration", fontsize=11)
        ax_right.set_ylabel("BIC Score", fontsize=11)
        ax_right.set_title(
            "Score Trajectory (GES Search)", fontsize=12, fontweight="bold"
        )
        ax_right.grid(True, alpha=0.3)
        ax_right.legend()
        # Comments panel.
        ax_comments.axis("off")
        ax_comments.set_title("Comments", fontsize=12, fontweight="bold")
        detail = (
            f"sample size N = {sample_size}\n"
            f"regularization = {regularization:.2f}\n\n"
            "GES searches DAG space with a\n"
            "greedy forward-backward heuristic:\n"
            "forward adds edges, backward\n"
            "removes low-value edges.\n\n"
            "Greedy search can get stuck in\n"
            "local optima; multiple random\n"
            "starts improve robustness."
        )
        htutori.add_fitted_text_box(
            ax_comments, detail, max_fontsize=11, min_fontsize=8
        )
        plt.tight_layout()
        plt.show()

    # Interactive widgets.
    sample_slider, sample_box = htutori.build_widget_control(
        name="N",
        description="Sample Size (N)",
        min_val=50,
        max_val=10000,
        step=100,
        initial_value=100,
        is_float=False,
    )
    regularization_slider, regularization_box = htutori.build_widget_control(
        name="reg",
        description="Regularization",
        min_val=0.0,
        max_val=2.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    output = Output()

    def update(change):
        with output:
            clear_output(wait=True)
            plot_ges_search(sample_slider.value, regularization_slider.value)

    sample_slider.observe(update, names="value")
    regularization_slider.observe(update, names="value")
    display(
        VBox(
            [
                HBox([sample_box, regularization_box]),
                output,
            ]
        )
    )
    update(None)


# #############################################################################
# Cell 6: LiNGAM Non-Gaussian
# #############################################################################
def cell6_lingam_nongaussian():
    """
    Show how non-Gaussianity enables full DAG recovery.
    """

    def plot_lingam(seed, skewness, snr):
        """
        Plot LiNGAM with non-Gaussian noise.
        """
        fig = plt.figure(figsize=(19, 5))
        gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.3)
        # Generate data with non-Gaussian noise.
        np.random.seed(seed)
        n_samples = 200
        # Generate skewed noise.
        if skewness > 0:
            noise = np.random.exponential(scale=1, size=n_samples) - 1
            noise = noise * skewness / np.std(noise)
        else:
            noise = np.random.normal(0, 1, n_samples)
        X = np.random.normal(0, 1, n_samples) * np.sqrt(snr)
        Y = 0.8 * X + noise * np.sqrt(1 - snr)
        Z = 0.8 * Y + noise * np.sqrt(1 - snr)
        # Plot three scatter plots.
        for idx, (data_x, data_y, title) in enumerate(
            [(X, Y, "X -> Y"), (Y, Z, "Y -> Z"), (X, Z, "X -> Z (indirect)")]
        ):
            ax = fig.add_subplot(gs[0, idx])
            ax.scatter(data_x, data_y, alpha=0.6, s=30, color="steelblue")
            ax.set_xlabel("Input", fontsize=10)
            ax.set_ylabel("Output", fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)
        # Jarque-Bera test for non-Gaussianity.
        jb_result = scipy.stats.jarque_bera(noise)
        is_non_gaussian = jb_result.pvalue < 0.05  # type: ignore
        # Comments panel.
        ax_comments = fig.add_subplot(gs[0, 3])
        ax_comments.axis("off")
        ax_comments.set_title("Comments", fontsize=11, fontweight="bold")
        detail = (
            f"seed = {seed}\n"
            f"skewness = {skewness:.1f}\n"
            f"SNR = {snr:.2f}\n\n"
            f"Jarque-Bera stat = {jb_result.statistic:.2f}\n"  # type: ignore
            f"p-value = {jb_result.pvalue:.4f}\n"  # type: ignore
            f"data are "
            f"{'NON-GAUSSIAN' if is_non_gaussian else 'GAUSSIAN'}\n\n"
            "Non-Gaussianity breaks the\n"
            "directional symmetry: LiNGAM\n"
            "exploits it to orient edges."
        )
        htutori.add_fitted_text_box(
            ax_comments, detail, max_fontsize=10, min_fontsize=7
        )
        plt.suptitle(
            "LiNGAM: Non-Gaussianity Reveals Causal Direction",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    # Interactive widgets.
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    skewness_slider, skewness_box = htutori.build_widget_control(
        name="skewness",
        description="Skewness",
        min_val=0.0,
        max_val=5.0,
        step=0.5,
        initial_value=0.0,
        is_float=True,
    )
    snr_slider, snr_box = htutori.build_widget_control(
        name="snr",
        description="Signal-to-Noise Ratio",
        min_val=0.1,
        max_val=0.99,
        step=0.05,
        initial_value=0.8,
        is_float=True,
    )
    output = Output()

    def update(change):
        with output:
            clear_output(wait=True)
            plot_lingam(
                seed_slider.value, skewness_slider.value, snr_slider.value
            )

    seed_slider.observe(update, names="value")
    skewness_slider.observe(update, names="value")
    snr_slider.observe(update, names="value")
    display(
        VBox(
            [
                seed_box,
                HBox([skewness_box, snr_box]),
                output,
            ]
        )
    )
    update(None)


# #############################################################################
# Cell 7: Comparing Algorithms
# #############################################################################
def cell7_algorithm_comparison():
    """
    Compare PC, GES, and LiNGAM outputs.
    """

    def plot_comparison(dataset_type, sample_size):
        """
        Plot three algorithm outputs side-by-side.
        """
        fig, axes = plt.subplots(1, 4, figsize=(19, 5))
        algorithms = [
            "PC (Constraint-Based)",
            "GES (Score-Based)",
            "LiNGAM (Functional)",
        ]
        edge_counts = [4, 5, 6]
        ambiguity = [2, 0, 0]
        for idx, (ax, algo, edges, ambig) in enumerate(
            zip(axes[:3], algorithms, edge_counts, ambiguity)
        ):
            # Draw sample DAG.
            G = nx.DiGraph()
            G.add_nodes_from([1, 2, 3])
            if idx == 0:  # PC: mixed edges.
                G.add_edges_from([(1, 2), (2, 3)])
            elif idx == 1:  # GES: all directed.
                G.add_edges_from([(1, 2), (2, 3)])
            else:  # LiNGAM: full DAG with weights.
                G.add_edges_from([(1, 2), (2, 3)])
            pos = {1: (0, 0), 2: (1, 1), 3: (2, 0)}
            nx.draw_networkx_nodes(
                G, pos, node_color="lightblue", node_size=600, ax=ax
            )
            nx.draw_networkx_edges(G, pos, ax=ax, width=2, arrowsize=20)
            nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
            # Summary text.
            summary = (
                f"{algo}\n"
                f"Edges: {edges}\n"
                f"Ambiguities: {ambig}\n"
                f"Sample Size: {sample_size}"
            )
            ax.text(
                0.5,
                -0.3,
                summary,
                ha="center",
                fontsize=10,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
            )
            ax.set_title(algo, fontsize=12, fontweight="bold")
            ax.axis("off")
        # Comments panel.
        axes[3].axis("off")
        axes[3].set_title("Comments", fontsize=12, fontweight="bold")
        detail = (
            f"dataset type = {dataset_type}\n"
            f"sample size N = {sample_size}\n\n"
            "PC: good for exploratory\n"
            "analysis, returns equivalence\n"
            "class.\n\n"
            "GES: more directed edges,\n"
            "assumes no hidden confounders.\n\n"
            "LiNGAM: requires non-\n"
            "Gaussianity, full DAG recovery."
        )
        htutori.add_fitted_text_box(
            axes[3], detail, max_fontsize=10, min_fontsize=7
        )
        plt.suptitle(
            "Comparing Causal Discovery Algorithms",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    # Interactive widgets.
    dataset_dropdown = Dropdown(
        options=["Linear Gaussian", "Linear Non-Gaussian", "Nonlinear"],
        value="Linear Gaussian",
        description="Dataset Type:",
    )
    sample_slider, sample_box = htutori.build_widget_control(
        name="N",
        description="Sample Size (N)",
        min_val=100,
        max_val=5000,
        step=100,
        initial_value=100,
        is_float=False,
    )
    output = Output()

    def update(change):
        with output:
            clear_output(wait=True)
            plot_comparison(dataset_dropdown.value, sample_slider.value)

    dataset_dropdown.observe(update, names="value")
    sample_slider.observe(update, names="value")
    display(
        VBox(
            [
                HBox([dataset_dropdown, sample_box]),
                output,
            ]
        )
    )
    update(None)


# #############################################################################
# Cell 8: Validating DAGs
# #############################################################################
def cell8_validation():
    """
    Show validation tests for discovered DAGs.
    """

    def plot_validation(alpha_threshold, confounder_strength):
        """
        Plot validation dashboard.
        """
        fig = plt.figure(figsize=(19, 5))
        gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.3)
        # CI validation test results.
        ax1 = fig.add_subplot(gs[0, 0])
        tests = ["X_|_Z|Y", "X_|_W|Y", "Y_|_Z"]
        p_values = [0.12, 0.03, 0.45]
        colors = ["green" if p > alpha_threshold else "red" for p in p_values]
        ax1.barh(tests, p_values, color=colors, alpha=0.7)
        ax1.axvline(
            x=alpha_threshold, color="black", linestyle="--", label="Alpha"
        )
        ax1.set_xlabel("p-value", fontsize=10)
        ax1.set_title("CI Test Validation", fontsize=11, fontweight="bold")
        ax1.legend()
        # Placebo test result.
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(
            0.5,
            0.5,
            "Placebo Test Result:\n"
            "Shuffled data found 0 edges\n(expected: 0)\nOK PASS",
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
            transform=ax2.transAxes,
        )
        ax2.set_title("Placebo Test", fontsize=11, fontweight="bold")
        ax2.axis("off")
        # Sensitivity analysis.
        ax3 = fig.add_subplot(gs[0, 2])
        confounder_values = np.linspace(0, 1, 50)
        robustness = 1 - (confounder_values * confounder_strength)
        ax3.plot(
            confounder_values, robustness, "o-", color="steelblue", linewidth=2
        )
        ax3.axvline(
            x=confounder_strength,
            color="red",
            linestyle="--",
            label=f"Current: {confounder_strength:.2f}",
        )
        ax3.set_xlabel("Confounder Strength", fontsize=10)
        ax3.set_ylabel("Robustness", fontsize=10)
        ax3.set_title("Sensitivity Analysis", fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        # Comments panel.
        validation_score = 0.67
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis("off")
        ax4.set_title("Comments", fontsize=11, fontweight="bold")
        detail = (
            f"alpha = {alpha_threshold:.2f}\n"
            f"confounder strength = {confounder_strength:.2f}\n\n"
            f"validation score = {validation_score:.1%}\n"
            "of implied CIs confirmed\n\n"
            "A discovered DAG is a\n"
            "hypothesis; validation checks\n"
            "consistency with the data."
        )
        htutori.add_fitted_text_box(ax4, detail, max_fontsize=10, min_fontsize=7)
        plt.suptitle(
            "Validating Discovered DAGs with Refutation Tests",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    # Interactive widgets.
    alpha_slider, alpha_box = htutori.build_widget_control(
        name="alpha",
        description="Alpha (CI threshold)",
        min_val=0.01,
        max_val=0.2,
        step=0.01,
        initial_value=0.05,
        is_float=True,
    )
    confounder_slider, confounder_box = htutori.build_widget_control(
        name="confounder",
        description="Confounder Strength",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.3,
        is_float=True,
    )
    output = Output()

    def update(change):
        with output:
            clear_output(wait=True)
            plot_validation(alpha_slider.value, confounder_slider.value)

    alpha_slider.observe(update, names="value")
    confounder_slider.observe(update, names="value")
    display(
        VBox(
            [
                HBox([alpha_box, confounder_box]),
                output,
            ]
        )
    )
    update(None)


# #############################################################################
# Cell 9: Domain Knowledge Integration
# #############################################################################
def cell9_domain_knowledge():
    """
    Show impact of domain knowledge constraints.
    """

    def plot_domain_constraints(prior_strength):
        """
        Plot discovery with and without constraints.
        """
        # 1xN layout: unconstrained DAG, constrained DAG, Comments.
        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        # Without constraints.
        ax = axes[0]
        G_unconstrained = nx.DiGraph()
        G_unconstrained.add_nodes_from([1, 2, 3, 4])
        G_unconstrained.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
        pos = {1: (0, 1), 2: (1, 2), 3: (1, 0), 4: (2, 1)}
        nx.draw_networkx_nodes(
            G_unconstrained, pos, node_color="lightblue", node_size=600, ax=ax
        )
        nx.draw_networkx_edges(
            G_unconstrained, pos, ax=ax, width=2, arrowsize=15
        )
        nx.draw_networkx_labels(G_unconstrained, pos, font_size=11, ax=ax)
        ax.set_title("No Constraints", fontsize=11, fontweight="bold")
        ax.axis("off")
        # With constraints.
        ax = axes[1]
        G_constrained = nx.DiGraph()
        G_constrained.add_nodes_from([1, 2, 3, 4])
        G_constrained.add_edges_from([(1, 2), (1, 3), (2, 4)])
        nx.draw_networkx_nodes(
            G_constrained, pos, node_color="lightgreen", node_size=600, ax=ax
        )
        nx.draw_networkx_edges(G_constrained, pos, ax=ax, width=2, arrowsize=15)
        nx.draw_networkx_labels(G_constrained, pos, font_size=11, ax=ax)
        ax.set_title("With Constraints", fontsize=11, fontweight="bold")
        ax.axis("off")
        # Comments panel: expert constraints + impact summary.
        axes[2].axis("off")
        axes[2].set_title("Comments", fontsize=11, fontweight="bold")
        detail = (
            "Expert constraints:\n"
            "- Forbidden: 3 -> 1\n"
            "  (outcome cannot cause treatment)\n"
            "- Required: 1 -> 2\n"
            "  (treatment causes outcome)\n"
            "- Temporal order: 1 -> {2,3} -> 4\n\n"
            f"Impact (prior strength = {prior_strength:.2f}):\n"
            "- Search space reduction: 60%\n"
            "- Ambiguous edges removed: 2\n"
            "- Convergence time: 40% faster\n"
            "- Accuracy improvement: ~25%"
        )
        htutori.add_fitted_text_box(
            axes[2], detail, max_fontsize=10, min_fontsize=7
        )
        plt.suptitle(
            "Domain Knowledge Integration: Constraints and Prior DAGs",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    # Interactive widgets.
    prior_slider, prior_box = htutori.build_widget_control(
        name="prior",
        description="Prior Strength",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    output = Output()

    def update(change):
        with output:
            clear_output(wait=True)
            plot_domain_constraints(prior_slider.value)

    prior_slider.observe(update, names="value")
    display(VBox([prior_box, output]))
    update(None)


# #############################################################################
# Cell 10: End-to-End Workflow
# #############################################################################
def cell10_end_to_end_workflow():
    """
    Show complete discovery pipeline.
    """

    def plot_workflow(dataset_name, selected_algorithms, progress_stage):
        """
        Plot multi-stage workflow.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        stages = [
            "Data Preparation",
            "Algorithm Selection",
            "Consensus",
            "Refinement",
            "Validation",
            "Final DAG",
        ]
        # Color-code stages by completion status.
        colors = [
            "lightgreen" if i <= progress_stage else "lightgray"
            for i in range(len(stages))
        ]
        # Stage 1: Data Preparation.
        ax = fig.add_subplot(gs[0, 0])
        ax.set_facecolor(colors[0])
        ax.axis("off")
        ax.set_title("Stage 1: Data Prep", fontsize=10, fontweight="bold")
        htutori.add_fitted_text_box(
            ax,
            "Variables: 5\nSamples: 500\nNon-Gaussian: yes",
            max_fontsize=10,
            min_fontsize=7,
        )
        # Stage 2: Algorithm Selection.
        ax = fig.add_subplot(gs[0, 1])
        ax.set_facecolor(colors[1])
        ax.axis("off")
        ax.set_title("Stage 2: Algorithms", fontsize=10, fontweight="bold")
        algo_text = "\n".join(selected_algorithms)
        htutori.add_fitted_text_box(
            ax, algo_text, max_fontsize=10, min_fontsize=7
        )
        # Stage 3: Consensus.
        ax = fig.add_subplot(gs[0, 2])
        ax.set_facecolor(colors[2])
        ax.axis("off")
        ax.set_title("Stage 3: Consensus", fontsize=10, fontweight="bold")
        htutori.add_fitted_text_box(
            ax,
            "Edges found by 2+ algos\n(more trustworthy)",
            max_fontsize=10,
            min_fontsize=7,
        )
        # Stage 4: Refinement.
        ax = fig.add_subplot(gs[1, 0])
        ax.set_facecolor(colors[3])
        ax.axis("off")
        ax.set_title("Stage 4: Refinement", fontsize=10, fontweight="bold")
        htutori.add_fitted_text_box(
            ax, "Expert review\nAdd constraints", max_fontsize=10, min_fontsize=7
        )
        # Stage 5: Validation.
        ax = fig.add_subplot(gs[1, 1])
        ax.set_facecolor(colors[4])
        ax.axis("off")
        ax.set_title("Stage 5: Validation", fontsize=10, fontweight="bold")
        htutori.add_fitted_text_box(
            ax,
            "Refutation tests\nSensitivity analysis",
            max_fontsize=10,
            min_fontsize=7,
        )
        # Stage 6: Final DAG.
        ax = fig.add_subplot(gs[1, 2])
        ax.set_facecolor(colors[5])
        ax.axis("off")
        ax.set_title("Stage 6: Final DAG", fontsize=10, fontweight="bold")
        htutori.add_fitted_text_box(
            ax,
            "Refined & validated\nReady for inference",
            max_fontsize=10,
            min_fontsize=7,
        )
        # Progress bar at bottom.
        ax = fig.add_subplot(gs[2, :])
        progress_pct = (progress_stage + 1) / len(stages)
        ax.barh([0], [progress_pct], height=0.3, color="steelblue", alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.text(
            progress_pct / 2,
            0,
            f"{progress_pct:.0%}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
        )
        ax.set_xlabel("Workflow Progress", fontsize=11)
        ax.set_yticks([])
        ax.set_title("Overall Progress", fontsize=11, fontweight="bold")
        plt.suptitle(
            f"End-to-End Causal Discovery Workflow: {dataset_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    # Interactive widgets.
    dataset_dropdown = Dropdown(
        options=[
            "Synthetic: Linear Gaussian",
            "Synthetic: Non-Gaussian",
            "Real: Economic Data",
        ],
        value="Synthetic: Linear Gaussian",
        description="Dataset:",
    )
    algo_pc = Checkbox(value=True, description="PC")
    algo_ges = Checkbox(value=True, description="GES")
    algo_lingam = Checkbox(value=False, description="LiNGAM")
    progress_slider, progress_box = htutori.build_widget_control(
        name="stage",
        description="Progress Stage",
        min_val=0,
        max_val=5,
        step=1,
        initial_value=0,
        is_float=False,
    )
    output = Output()

    def update(change):
        with output:
            clear_output(wait=True)
            selected = []
            if algo_pc.value:
                selected.append("PC")
            if algo_ges.value:
                selected.append("GES")
            if algo_lingam.value:
                selected.append("LiNGAM")
            if not selected:
                selected = ["PC", "GES"]
            plot_workflow(
                dataset_dropdown.value, selected, progress_slider.value
            )

    dataset_dropdown.observe(update, names="value")
    algo_pc.observe(update, names="value")
    algo_ges.observe(update, names="value")
    algo_lingam.observe(update, names="value")
    progress_slider.observe(update, names="value")
    display(
        VBox(
            [
                dataset_dropdown,
                HBox([algo_pc, algo_ges, algo_lingam]),
                progress_box,
                output,
            ]
        )
    )
    update(None)
