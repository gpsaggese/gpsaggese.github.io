# Plan: Reorganize Notebooks to Match Lesson09.8 / Lesson09.9 Flow

- The current setup of the content is described in
  research/Implement_MonteCarlo_Tree_Search_and_Alpha_Zero/README.md

- Invariants
  - remove direct references to Lesson09.8 and 09.9 in the comments of the notebooks
    but follow their logical flow and their notation
  - move content from old notebooks to new notebooks and from old files to new files,
    since we want to **move** the content
  - for every change update
    research/Implement_MonteCarlo_Tree_Search_and_Alpha_Zero/README.md
  - if you are unsure, ask questions to the user

## [x] Step 1: 

- Achieve the following organization
  - game.py -> base interface of Game
  - game_examples.py -> TicTacToe + ConnectFour
  - game.API.ipynb -> explain the API of game and show the examples of TicTacToe and
    Connect4 moving code from mcts.API.ipynb to game.API.ipynb
  - **Frame the game before searching it.** 
     explicitly map `TicTacToe`/`ConnectFour` onto
     Lesson09.8's 5-tuple (`s0`, `Actions`, `Result`, `IsTerminal`,
     `Utility`) and names the tree as an **AND-OR tree** (Max = OR, Min =
     AND), before jumping into Minimax.

## [x] Step 2:

Create a search_algorithms.API.ipynb notebook that explains the code in
search_algorithms_utils.py using .claude/skills/notebook.create_api_intro/SKILL.md

Explain the API of SearchNode

Explain the API of build_tree_graph, build a fake data structure, explain the colors

Explain bit by bit how the minimax tree is built using the demo_state and the
fork_state (as a switch) using a widget that builds the tree little by little calling
the functions
- You can use connect 3 as an example

- Result: `search_algorithms.02.API.ipynb` (backed by
  `search_algorithms_API_utils.py`); added `ConnectThree` to
  `game_examples.py` for the connect-3 example

## [x] Step 3

In build_tree_graph is it possible to show also the state of the board calling
  render?

- In cell4_2_build_minimax_step_widget remove the histogram and put the Comments
  on the side of the graph

- In part 3 of search_algorithms.02.API.ipynb build_tree_graph there should be
  only what is needed to explain minimax (only blue and green nodes)
  - Then there is a part explaining alpha-beta search (gray nodes)
  - Then there is a part explaining a depth limited (amber nodes)

- Result: `build_tree_graph()` and `_build_partial_tree_graph()` both take
  an opt-in `game=` switch that stacks `game.render(node.state)` on each
  node (monospace font); the step widget got a "show board" checkbox for
  it, on by default. The widget's histogram (`_plot_step_progress()`) was
  removed; the Comments panel now sits beside the tree diagram via
  `ipywidgets.HBox([graph_output, comment_output])` instead of below it.
  Part 3 of `search_algorithms.02.API.ipynb` now grows one hand-built tree
  across 3 cells, one algorithm's color at a time (3.1 minimax:
  blue/green, 3.2 alpha-beta: grey/dashed, 3.3 depth-limited: amber),
  then 3.4 demos the board switch on the finished tree.

## 

The search algorithms should not refer to mcts since mcts comes later,
it should be the opposite, i.e., mcts referring to the search algorithms

Move flat monte carlo from search_algorithms_utils.py to mcts.

Number the files so that the order is clarified

- Result: flat Monte Carlo moved to `mcts_utils.py` (reuses
  `search_algorithms_utils.SearchNode` / `pick_best_move()`);
  `search_algorithms_utils.py` no longer imports `mcts_utils`. Notebooks
  renamed `<name>.<ID>.<type>.ipynb`: `game.01.API.ipynb`,
  `search_algorithms.02.API.ipynb` / `.02.example.ipynb`,
  `mcts.03.API.ipynb` / `.03.example.ipynb`; `*_utils.py` core/widget
  modules are unnumbered since they are imported as Python modules.

## Step 3

- classical_search_algorithms_utils.py
- classical_search_algorithms.example.py / ipynb
  - For each algo show TicTacToe and ConnectFour

- mcts.API.ipynb

Key finding: Lesson09.9 is a newer lecture that split out of Lesson09.8 (see
the `// From: Lesson09.8-...` provenance comments in the `.smd`). Minimax,
alpha-beta, depth-limited search, and evaluation-function content now has its
detailed, canonical treatment in Lesson09.9, with two entirely new subsections
(Move Ordering, Horizon Effect) that no notebook currently covers. The
notebooks still cite "Lesson09.8" for this material and don't reflect the
split.

## Steps

- **Split "Depth-Limited Search" into its two lecture sections.** Rename
   Part 5 to "Depth-Limited Game Search" for the `Cutoff`/`Eval`
   formalism, then add a distinct "Evaluation Functions for Connect 4"
   subsection generalizing `evaluate_tic_tac_toe` as an instance of the
   linear form $Eval(s) = \sum w_i f_i(s)$.

- **Add the missing "Move Ordering" section.** New cell(s) in Part 4
   (Alpha-Beta) showing perfect/random/worst move ordering's effect on
   nodes explored ($O(b^{m/2})$ vs $O(b^{3m/4})$ vs $O(b^m)$). Requires a
   small helper in `search_algorithms_utils.py` to reorder children
   before search.

5. **Add the missing "Horizon Effect" section.** New cell(s) after the
   eval-function material, using a forced-loss-delayed-by-quiet-moves
   position to show depth-limited search reporting a healthy score right
   up to the point the loss crosses the horizon.

6. **Re-scope "Part 6: Flat Monte Carlo".** Keep it (needed for the Part 7
   comparison), but retitle it to make explicit that it is Lesson09.8
   content living inside the Lesson09.9-flow notebook, and cross-link both
   directions.

7. **Tighten Part headers to lecture header names.** Rename notebook
   `# Part N: ...` headers to match `.smd` `*`-level headers verbatim
   where they already correspond 1:1 (e.g. "Minimax", "Alpha-Beta
   Pruning"), so the mapping is mechanically checkable.

8. **Update `README.md`.** Refresh the file-description table and
   "Running the Notebooks" order to describe the two-lecture arc
   explicitly: `mcts.API` → `mcts.example` cover Lesson09.8's MCTS core;
   `search_algorithms.example` covers Lesson09.9's classical/adversarial
   search plus the closing MCTS comparison.

9. **Re-sync and renumber.** After editing each `.py`, run jupytext sync
   to regenerate the paired `.ipynb`, renumber `Cell X.Y` labels
   consecutively, and re-run each notebook top to bottom to confirm
   outputs still hold.

10. **Re-run the Docker e2e test.** `test/test_docker_template.py` runs
    each notebook by name with no cell-order assumptions, so it stays
    valid as-is — just re-run it after the reorg to catch any broken cell
    dependency introduced by the new sections.
