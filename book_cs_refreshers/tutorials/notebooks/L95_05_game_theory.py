# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Game Theory: From Payoff Matrices to Evolutionary Dynamics
#
# - This notebook builds strategic-reasoning intuition around one reusable
#   tool: an editable 2x2 payoff-matrix sandbox
# - The flow is:
#   - Read a payoff matrix -> detect dominant strategies -> find Nash
#     equilibria by best response -> recognize classic games as points in the
#     same payoff space -> mix strategies when no pure equilibrium exists ->
#     solve sequential games by backward induction -> watch equilibrium
#     emerge from selection instead of reasoning -> see decentralized play
#     fail to reach the social optimum
# - Scope: foundational, most visualizable chapters (normal form, equilibrium
#   concepts, classic games, sequential games, evolutionary game theory,
#   network games). Cooperative game theory, Bayesian/signaling games, and the
#   applied economics/politics/business chapters are left to the lecture
#   slides

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# System libraries.
import logging

# Third-party libraries.

# %%
import helpers.hmodule as hmodule

hmodule.install_module_if_not_present(
    "networkx", use_activate=True, use_sudo=False, venv_path="/opt/venv"
)

# %%
import helpers.htutorial as htutori

import class_cs_refreshers.tutorials.notebooks.L95_05_game_theory_utils as utils

htutori.config_notebook()

# Initialize logger.
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Part 1: The Payoff Matrix and the Game Sandbox

# %% [markdown]
# ## Cell 1.1: Reading a Payoff Matrix
#
# **Goal**:
# - Ground the normal-form representation with a concrete example (Prisoners'
#   Dilemma) before any interactivity is introduced
# - Understand that each cell holds (row player payoff, column player payoff)
#
# **Plots**:
# - _Payoff matrix_: 2x2 table for the Prisoners' Dilemma, rows = Player 1's
#   action, columns = Player 2's action
# - _Comments_: payoff convention (row player's payoff listed first)

# %%
# Display the Prisoners' Dilemma payoff matrix as a static reference.
utils.cell1_1_show_payoff_matrix()

# %% [markdown]
# **Key observations**:
# - The matrix packs 4 numbers per outcome (2 per player) into one compact
#   structure
# - Row Player 1 best cell in one column is not necessarily best in another
#   column: payoffs are interdependent
# - Reading a payoff matrix is the prerequisite for every equilibrium concept
#   that follows

# %% [markdown]
# ## Cell 1.2: The 2x2 Game Sandbox: Editable Payoffs
#
# **Goal**:
# - Build the reusable tool used through Parts 1-3: an editable 2x2
#   normal-form game where changing any payoff instantly updates the picture
# - Build intuition that Prisoners' Dilemma, Battle of the Sexes, and other
#   classic games are all points in the same 8-dimensional payoff space
#
# **Plots**:
# - _Payoff matrix_: live-updating 2x2 grid
# - _Comments_: current 8 payoff values, whether the game is zero-sum
#
# **Parameters**:
# - `game`: preset dropdown (Custom, Prisoners' Dilemma, Battle of the Sexes,
#   Stag Hunt, Matching Pennies)
# - `p1_11, p1_12, p1_21, p1_22`: row player's payoffs (-5 to 5)
# - `p2_11, p2_12, p2_21, p2_22`: column player's payoffs (-5 to 5)

# %%
# Build the editable 2x2 payoff-matrix sandbox.
utils.cell1_2_game_sandbox()

# %% [markdown]
# **Key observations**:
# - Every classic game in this notebook is one setting of these 8 numbers
# - A small change (turning +1 into -1 in one cell) can flip a game from
#   non-zero-sum to zero-sum
# - This sandbox is reused with different lenses in Cells 2.1, 2.2, and 3.1

# %% [markdown]
# # Part 2: Dominant Strategies and Nash Equilibrium

# %% [markdown]
# ## Cell 2.1: Highlighting Dominant and Dominated Strategies
#
# **Goal**:
# - Apply the sandbox from Cell 1.2 to detect a dominant strategy: an action
#   that is best against every opponent action
# - Extend Cell 1.2 by adding automatic classification instead of plain
#   display
#
# **Plots**:
# - _Payoff matrix_: same grid as Cell 1.2, with the dominant row and/or
#   column (if one exists) outlined in a bold border
# - _Comments_: whether each player has a dominant strategy, and which one
#
# **Parameters**:
# - Same 8 payoff sliders and `game` preset dropdown as Cell 1.2

# %%
# Outline the dominant row and/or column, if one exists.
utils.cell2_1_dominant_strategy_widget()

# %% [markdown]
# **Key observations**:
# - Prisoners' Dilemma preset: Defect is dominant for both players even
#   though mutual cooperation is better for both
# - Not every game has a dominant strategy: switch to Battle of the Sexes and
#   watch the outline disappear
# - A dominant strategy makes prediction trivial since a rational player
#   needs no belief about the opponent at all

# %% [markdown]
# ## Cell 2.2: Finding Nash Equilibria via Best Response
#
# **Goal**:
# - Introduce the best-response method: highlight each player's best action
#   given every opponent choice, and mark cells where both coincide
# - Connect dominant strategies (Cell 2.1) to the more general Nash
#   equilibrium concept
#
# **Plots**:
# - _Payoff matrix_: row payoff boxed blue where it is Player 1's best
#   response, column payoff boxed orange where it is Player 2's best
#   response; cells with both boxes shaded gold as Nash equilibria
# - _Comments_: number of pure-strategy Nash equilibria and their coordinates
#
# **Parameters**:
# - Same 8 payoff sliders and `game` preset dropdown as Cell 1.2

# %%
# Highlight best responses and pure Nash equilibria.
utils.cell2_2_nash_equilibrium_widget()

# %% [markdown]
# **Key observations**:
# - Prisoners' Dilemma has exactly one Nash equilibrium, (Defect, Defect),
#   matching its dominant strategy
# - Battle of the Sexes has two pure Nash equilibria: multiplicity creates an
#   equilibrium-selection problem
# - Matching Pennies has zero pure Nash equilibria: the best-response marks
#   chase each other around the matrix with no cell where both coincide

# %% [markdown]
# # Part 3: A Gallery of Classic 2x2 Games

# %% [markdown]
# ## Cell 3.1: Prisoners' Dilemma, Battle of the Sexes, and Stag Hunt Side by Side
#
# **Goal**:
# - Compare three canonical games at once to see how equilibrium structure
#   differs even though all three are 2x2 games
# - Apply the best-response/Nash analysis from Cells 2.1-2.2 to three games
#   simultaneously
#
# **Plots**:
# - _Three payoff matrices_: Prisoners' Dilemma, Battle of the Sexes, and
#   Stag Hunt side by side, each with best-response highlighting
# - _Comments_: per game, the number of pure Nash equilibria and whether each
#   is Pareto optimal
#
# **Parameters**:
# - `highlight`: toggle to show or hide best-response highlighting

# %%
# Compare three canonical games side by side.
utils.cell3_1_classic_games_gallery()

# %% [markdown]
# **Key observations**:
# - Prisoners' Dilemma: unique Nash equilibrium, not Pareto optimal since
#   mutual cooperation is better for both but unreachable
# - Battle of the Sexes: two Nash equilibria, both Pareto optimal, but the
#   players disagree on which they prefer
# - Stag Hunt: two Nash equilibria, but only (Stag, Stag) is Pareto optimal:
#   (Hare, Hare) is Pareto-dominated even though it is risk dominant

# %% [markdown]
# ## Cell 3.2: Stag Hunt: Payoff Dominance vs Risk Dominance
#
# **Goal**:
# - Zoom into the equilibrium-selection problem left open by Cell 3.1's Stag
#   Hunt panel: which of the two Nash equilibria should rational players
#   expect
# - Show that the answer depends on trust in the opponent, not on the payoff
#   matrix alone
#
# **Plots**:
# - _Expected payoff vs belief_: expected payoff from Stag as a function of
#   the believed probability $q$ the opponent plays Stag, compared against
#   the constant payoff from Hare
# - _Comments_: current $V$, current belief $q$, and which action is
#   currently better
#
# **Parameters**:
# - `V`: stag payoff when both hunt stag (2-6)
# - `q`: believed probability the opponent chooses Stag (0.0-1.0)

# %%
# Explore the belief threshold between Stag and Hare.
utils.cell3_2_stag_hunt_dominance()

# %% [markdown]
# **Key observations**:
# - The crossing point of the two lines is the belief threshold above which
#   Stag becomes the rational choice
# - A higher $V$ shifts the threshold, making Stag rational under more doubt
#   about the opponent
# - Payoff dominant does not mean safe: below the threshold, Hare remains the
#   individually rational choice

# %% [markdown]
# # Part 4: Mixed Strategy Equilibria

# %% [markdown]
# ## Cell 4.1: Why Randomize? Matching Pennies Has No Pure Equilibrium
#
# **Goal**:
# - Show, by exhaustive best response, that Matching Pennies has no pure Nash
#   equilibrium, motivating randomization as the only stable strategy
# - Apply the Cell 2.2 best-response method to a game where it fails to find
#   a pure Nash equilibrium
#
# **Plots**:
# - _Payoff matrix_: Matching Pennies with best-response highlights chasing
#   each other with no coinciding cell
# - _Comments_: confirmation that 0 pure Nash equilibria exist

# %%
# Show the best-response cycle in Matching Pennies.
utils.cell4_1_matching_pennies_best_response()

# %% [markdown]
# **Key observations**:
# - Whatever the row player is expected to do, switching is always
#   profitable: the best response cycles Heads -> Tails -> Heads
# - This cycling is the signature of a game with no pure Nash equilibrium
# - The next cell shows that allowing randomization restores an equilibrium

# %% [markdown]
# ## Cell 4.2: Computing the Equilibrium Mix by Indifference
#
# **Goal**:
# - Derive the mixed Nash equilibrium via the indifference principle: find
#   the opponent's mixing probability that makes both of your actions
#   equally good
# - Compute and visualize the crossing point predicted qualitatively in
#   Cell 4.1
#
# **Plots**:
# - _Expected payoff vs opponent's probability_: expected payoff from Heads
#   and from Tails, plotted against $q$ = probability the opponent plays
#   Heads
# - _Comments_: computed $q^*$, expected value of the game at equilibrium
#
# **Parameters**:
# - `stakes`: win/lose payoff magnitude (1-5), rescales both lines without
#   moving the crossing point

# %%
# Derive the mixed equilibrium by the indifference principle.
utils.cell4_2_mixed_equilibrium_indifference()

# %% [markdown]
# **Key observations**:
# - The crossing point is exactly $q^* = 1/2$ for symmetric Matching Pennies,
#   matching the lecture's derivation
# - Changing the stakes rescales both lines but never moves the equilibrium
#   probability: mixing probabilities are pinned down by the opponent's
#   payoffs, not your own
# - At the crossing, both actions give equal expected payoff: exactly the
#   definition of indifference

# %% [markdown]
# # Part 5: Sequential Games and Backward Induction

# %% [markdown]
# ## Cell 5.1: Solving the Entry Game by Backward Induction
#
# **Goal**:
# - Introduce the extensive form (game tree) using the market-entry example
#   from the lecture, and solve it by reasoning from the leaves back to the
#   root
# - Contrast with Parts 1-4, where every game was simultaneous: here timing
#   matters
#
# **Plots**:
# - _Game tree_: Entrant's node at the root branching to Out / Enter; Enter
#   leads to the Incumbent's node branching to Accommodate / Fight
# - _Comments_: payoff at each leaf and which branch backward induction
#   selects at each internal node
#
# **Parameters**:
# - `out_payoff`: Entrant's payoff if it stays Out (-2 to 2)
# - `fight_payoff`: Incumbent's payoff if it Fights (-3 to 1)

# %%
# Solve the market-entry game by backward induction.
utils.cell5_1_entry_game_backward_induction()

# %% [markdown]
# **Key observations**:
# - Backward induction solves the Incumbent's node first (Accommodate beats
#   Fight when fighting is costly), then the Entrant's node using that result
# - The solved path (Enter, Accommodate) highlights in green while the pruned
#   branch (Fight) stays gray
# - Raising the Incumbent's fight payoff above its accommodate payoff flips
#   the solved path entirely, showing how one number changes the whole game

# %% [markdown]
# ## Cell 5.2: Non-Credible Threats and Subgame Perfection
#
# **Goal**:
# - Show why the Nash equilibrium (Out, Fight) is not subgame perfect: the
#   threat to Fight is not credible once the Entrant has actually entered
# - Contrast Cell 5.1's solved tree with an unsolved, threat-based Nash
#   equilibrium on the same game
#
# **Plots**:
# - _Two trees side by side_: left tree shows the baseline subgame perfect
#   outcome; right tree shows the (Out, Fight) equilibrium with the Fight
#   branch marked non-credible in red, unless the commitment device is on
# - _Comments_: payoff comparison of the two equilibria for the Entrant
#
# **Parameters**:
# - `commitment`: toggle for whether the Incumbent has a credible commitment
#   device that raises its Fight payoff enough to make Fight genuinely
#   optimal

# %%
# Contrast a non-credible threat with a genuine commitment device.
utils.cell5_2_credible_commitment()

# %% [markdown]
# **Key observations**:
# - A strategy profile can be a Nash equilibrium and still rely on a threat
#   the threatener would not actually carry out
# - Every subgame perfect equilibrium is a Nash equilibrium, but this example
#   is a Nash equilibrium that is not subgame perfect
# - Toggling on a genuine commitment device is what turns an empty threat
#   into a credible one, changing the subgame perfect equilibrium itself

# %% [markdown]
# # Part 6: Evolutionary Game Theory

# %% [markdown]
# ## Cell 6.1: Hawk-Dove and Evolutionarily Stable Strategies
#
# **Goal**:
# - Introduce equilibrium without rationality: the Hawk-Dove game, where
#   fitness rather than reasoning determines which strategy survives
# - Connect the evolutionarily stable strategy condition to the Nash
#   equilibrium concept from Part 2
#
# **Plots**:
# - _Payoff matrix_: Hawk-Dove matrix with cells computed from $V$ and $C$
# - _Fitness vs population share_: a Hawk's expected fitness against a
#   Dove's expected fitness as a function of the population fraction playing
#   Hawk
# - _Comments_: current $V$, $C$, ESS type, and ESS fraction
#
# **Parameters**:
# - `V`: resource value (1-10)
# - `C`: fight cost (1-10)

# %%
# Explore the Hawk-Dove game and its evolutionarily stable strategy.
utils.cell6_1_hawk_dove_ess()

# %% [markdown]
# **Key observations**:
# - When $V > C$, Hawk is a pure evolutionarily stable strategy: escalation
#   always pays
# - When $V < C$, the evolutionarily stable strategy is a mix of Hawks and
#   Doves at the fraction where both have equal fitness
# - The evolutionarily stable fraction depends only on the ratio $V/C$, not
#   on the absolute payoff scale

# %% [markdown]
# ## Cell 6.2: Replicator Dynamics: Watching a Population Converge
#
# **Goal**:
# - Simulate how the population share of Hawks evolves over time under the
#   replicator equation, showing convergence to the evolutionarily stable
#   strategy found in Cell 6.1 without any individual doing the reasoning
# - Make the fixed point from Cell 6.1 concrete as a trajectory that starts
#   anywhere and ends at the same place
#
# **Plots**:
# - _Population trajectory_: Hawk fraction $x_t$ over generations, starting
#   from the chosen initial fraction and converging to the evolutionarily
#   stable fraction
# - _Comments_: current generation, current Hawk fraction, distance to the
#   evolutionarily stable fraction
#
# **Parameters**:
# - `x0`: initial fraction of Hawks (0.0-1.0)
# - `V`: resource value (1-10)
# - `C`: fight cost (1-10)
# - `n_generations`: number of generations simulated (10-200)

# %%
# Simulate the population converging to the evolutionarily stable fraction.
utils.cell6_2_replicator_dynamics()

# %% [markdown]
# **Key observations**:
# - Every starting fraction $x_0$ converges to the same evolutionarily
#   stable fraction: the fixed point is stable to perturbation, exactly what
#   evolutionary stability means
# - Convergence is fast near the middle of the range and slows near the
#   boundaries (0 or 1)
# - No individual animal computes a best response: selection alone drives the
#   population toward the same point Nash equilibrium analysis predicts

# %% [markdown]
# # Part 7: Price of Anarchy in Network Routing

# %% [markdown]
# ## Cell 7.1: Braess's Paradox: Adding a Road That Hurts Everyone
#
# **Goal**:
# - Close with a network game showing decentralized optimization can fail
#   even without any single irrational player
# - Contrast selfish equilibrium routing against the socially optimal routing
#
# **Plots**:
# - _Network diagram_: 4-node network (source, two middle nodes, sink) with a
#   toggleable zero-cost shortcut edge
# - _Travel time comparison_: total travel time under selfish equilibrium
#   routing with the shortcut off vs on, next to the socially optimal routing
# - _Comments_: current parameters, equilibrium and social-optimum totals,
#   price of anarchy
#
# **Parameters**:
# - `shortcut`: toggle to add or remove the zero-cost shortcut edge
# - `n_drivers`: number of drivers (10-100)

# %%
# Explore Braess's paradox on a small routing network.
utils.cell7_1_braess_paradox()

# %% [markdown]
# **Key observations**:
# - Adding the shortcut edge makes every driver's equilibrium travel time go
#   up, not down
# - Once the shortcut exists, the socially optimal routing (drivers split
#   evenly, ignoring the shortcut) beats the selfish equilibrium for everyone
# - Price of anarchy formalizes this gap as selfish equilibrium cost divided
#   by optimal cost, here strictly greater than 1

# %% [markdown]
# # Summary: The Mental Model
#
# - A normal-form game is a payoff matrix; a dominant strategy is a row or
#   column that wins against everything; a Nash equilibrium is a cell where
#   every player's best response coincides
# - Classic games (Prisoners' Dilemma, Battle of the Sexes, Stag Hunt,
#   Matching Pennies) are specific points in the same payoff space,
#   distinguished by how many pure equilibria they have and whether those
#   equilibria are efficient
# - When no pure equilibrium exists, the indifference principle pins down a
#   mixed equilibrium: the opponent's payoffs, not your own, determine your
#   mixing probability
# - Sequential games add timing: backward induction finds the subgame perfect
#   equilibrium, which rules out empty threats that ordinary Nash equilibrium
#   allows
# - Evolutionary game theory reaches the same equilibrium concepts through
#   population dynamics instead of individual rationality, and network games
#   show that decentralized equilibrium play can be strictly worse than the
#   social optimum
