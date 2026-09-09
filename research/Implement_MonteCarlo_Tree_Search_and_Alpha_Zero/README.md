# Implement Monte Carlo Tree Search and Alpha Zero

This project implements the Monte Carlo Tree Search (MCTS) algorithm and the Alpha
Zero algorithm for game playing

- A game-agnostic `Game` interface (`game.py`) is the contract every search
  algorithm below is built against

- Three concrete games (`game_examples.py`) plug into that interface:
  - Tic-tac-toe
  - Connect Three (`ConnectFour`'s gravity-drop rule on a 3x3, 3-in-a-row
    board, small enough to search exhaustively)
  - Connect Four

- `search_algorithms_utils.py`
  - Classical searches: minimax, alpha-beta pruning, and depth-limited search
  - `SearchNode` and `build_tree_graph()`: the search-tree node type and Graphviz
    renderer flat Monte Carlo and MCTS reuse rather than duplicating

- `mcts_utils.py`
  - Game-agnostic MCTS engine: selection, expansion, rollout, backpropagation
  - Flat Monte Carlo (`build_flat_mc_tree()`): MCTS with a tree of depth one,
    kept alongside MCTS since both reuse `random_rollout()`

## Structure of the Dir

| File    | Description                                                         |
| ------- | ------------------------------------------------------------------- |
| `test/` | Docker-based end-to-end test that runs every notebook top to bottom |

## Description of Files

| File                                    | Description                                                                                          | Cluster            |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------ |
| `game.py`                                | `Game` interface (`State`, `Move`, 6 abstract methods)                                                 | Core Engine        |
| `game_examples.py`                       | `TicTacToe`, `ConnectFour`, and `ConnectThree` implementations                                         | Game Rules         |
| `game_API_utils.py`                      | Widget helpers for `game.01.API.ipynb`                                                                 | Notebook Utilities |
| `game.01.API.ipynb`                      | API tour of `game.py`; frames both games as an AND-OR tree                                             | Notebooks          |
| `search_algorithms_utils.py`             | Minimax, alpha-beta, depth-limited search, and the search-tree node/renderer (`SearchNode`, `build_tree_graph()`) | Core Engine |
| `search_algorithms_API_utils.py`         | Widget helpers for `search_algorithms.02.API.ipynb`                                                    | Notebook Utilities |
| `search_algorithms.02.API.ipynb`         | API tour of `search_algorithms_utils.py`: `SearchNode`, `build_tree_graph()`, and a widget that grows a minimax tree step by step | Notebooks |
| `search_algorithms.02.example.ipynb`     | Classical searches on tic-tac-toe, each with its search tree, compared against MCTS                    | Notebooks          |
| `mcts_API_utils.py`                      | Widget and search-tree plotting helpers for `mcts.03.API.ipynb`                                        | Notebook Utilities |
| `mcts_utils.py`                          | Game-agnostic MCTS engine (`MCTSNode`, `run_mcts()`) and flat Monte Carlo (`build_flat_mc_tree()`)     | Core Engine        |
| `mcts.03.API.ipynb`                      | API tour of `mcts_utils.py`                                                                            | Notebooks          |
| `mcts.03.example.ipynb`                  | Milestone 1: MCTS vs. a random player on tic-tac-toe                                                   | Notebooks          |

## Flow of the Code / Notebooks

Files are grouped into 3 numbered arcs, all built on the same `Game`
interface; the number prefixing each notebook (`01`/`02`/`03`) is the group
it belongs to. The `*_utils.py` core/widget modules are not numbered, since
they are imported as Python modules (`import ...search_algorithms_utils`),
and a leading digit is not a legal module name.

- **01, frame the game** (`game.py`, `game_examples.py`, `game.01.API.ipynb`):
  define a game as a search problem
- **02, search it exactly and approximately**
  (`search_algorithms_utils.py`, `search_algorithms.02.API.ipynb`,
  `search_algorithms.02.example.ipynb`): solve it with minimax / alpha-beta /
  depth-limited search, then compare against flat Monte Carlo and MCTS
- **03, search it by sampling** (`mcts_utils.py`, `mcts.03.API.ipynb`,
  `mcts.03.example.ipynb`): solve it by sampling (MCTS)

### 01: Frame the game

1. `game.py`
   - `Game`: the abstract interface every game implements (`get_initial_state()`,
     `get_legal_moves()`, `apply_move()`, `is_terminal()`, `get_winner()`,
     `get_current_player()`, `render()`)
   - Players are `1` and `-1`; a draw is `0`

2. `game_examples.py`
   - `TicTacToe`, `ConnectFour`, `ConnectThree`: concrete `Game` implementations
   - `ConnectThree` is `ConnectFour`'s gravity-drop rule on a 3x3 board,
     small enough to search exhaustively

3. `game.01.API.ipynb`: explore the `Game` interface
   - Backed by `game_API_utils.py`
   - Part 1: Library overview and mental model
   - Part 2: `Game` and `TicTacToe`
   - Part 3: Connect Four (same interface, bigger board)
   - Part 4: Framing the game as a search problem ($s_0$, $Actions$,
     $Result$, $IsTerminal$, $Utility$), naming the tree an AND-OR tree

### 02: Search it exactly and approximately

1. `search_algorithms_utils.py`
   - Classical searches: minimax, alpha-beta pruning, and depth-limited search
   - `SearchNode` and `build_tree_graph()`: the search-tree node type and
     Graphviz renderer that flat Monte Carlo and MCTS reuse rather than
     duplicate

2. `search_algorithms.02.API.ipynb`: the `SearchNode` / `build_tree_graph()`
   API, and how a search actually builds a tree
   - Backed by `search_algorithms_API_utils.py`
   - Part 1: Library overview and mental model
   - Part 2: `SearchNode`
   - Part 3: `build_tree_graph()`: coloring a search tree
   - Part 4: Building the minimax tree step by step (tic-tac-toe)
   - Part 5: A second game: Connect Three

3. `search_algorithms.02.example.ipynb`: classical and adversarial search
   - Part 2: The game
   - Part 3: Minimax
   - Part 4: Alpha-beta pruning
   - Part 5: Depth-limited search
   - Part 6: Flat Monte Carlo
   - Part 7: Comparing all four (plus MCTS)
   - Part 8: Full-game sanity check

### 03: Search it by sampling

1. `mcts_utils.py`
   - Game-agnostic MCTS engine: `MCTSNode`, `run_mcts()` (selection,
     expansion, rollout, backpropagation)
   - Flat Monte Carlo (`build_flat_mc_tree()`): MCTS with a tree of depth
     one, sharing `random_rollout()` with MCTS

2. `mcts.03.API.ipynb`: the MCTS engine
   - Part 1: Library overview and mental model
   - Part 2: `MCTSNode`
   - Part 3: `run_mcts()`
   - Part 4: Composing players and games
   - Part 5: Evaluation API (win rate over games)
   - Part 6: Connect Four (same engine, bigger board)

3. `mcts.03.example.ipynb`: MCTS end to end
   - Part 2: The game
   - Part 3: MCTS on a single position
   - Part 4: Full games (MCTS vs. random, MCTS vs. MCTS)
   - Part 5: Evaluation (win rate over 300 games)

## Running the Notebooks

- Build the Docker image:

  ```bash
  > ./docker_build.sh
  ```

- Launch Jupyter inside the container:

  ```bash
  > ./docker_jupyter.sh
  ```

- From the Jupyter file browser, open the 5 notebooks in the order listed in
  "Flow of the Code / Notebooks" above (`game.01.API.ipynb` ->
  `search_algorithms.02.API.ipynb` -> `search_algorithms.02.example.ipynb`
  -> `mcts.03.API.ipynb` -> `mcts.03.example.ipynb`)

- For more information on the Docker build system refer to
  [Project template readme](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/class_project/project_template/docker_scripts.README.md)
