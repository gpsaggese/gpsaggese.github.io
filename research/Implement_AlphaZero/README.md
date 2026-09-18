# Implement AlphaZero

Game representations for AlphaZero: player-relative board encodings,
legal-action masks, and exact terminal outcomes. The API notebook explains
each function through a complete hand-played Tic-Tac-Toe game.

The project reuses the `Game` interface and board games from the
[MCTS project](../Implement_MonteCarlo_Tree_Search_and_Alpha_Zero/README.md).
Game rules remain separate from the representation utilities. Start with
`alphazero_utils.py`, then explore the API notebook.

## Structure of the Directory

| Directory | Description |
| :--- | :--- |
| `test/` | Unit tests and the reference project's Docker notebook test pattern |

## Description of Files

| File | Description |
| :--- | :--- |
| `alphazero_utils.py` | Player-relative encoding, legal-action mask, and exact terminal value |
| `alphazero.API.ipynb` | Guided API tour and a hand-played Tic-Tac-Toe example |
| `test/test_alphazero_utils.py` | Core representation and game-integration tests |
| `test/test_docker_template.py` | Docker build/script checks and notebook execution using shared helpers |
| `requirements.txt` | Reference requirements plus `pytest<9` for the shared test hooks |
| `Dockerfile` | Python 3.12 slim CPU image with Jupyter and project dependencies |
| `.dockerignore` | Shared project-template build exclusions |
| `docker_name.sh` | Local image name: `gpsaggese/implement_alphazero` |
| `bashrc`, `etc_sudoers`, `version.sh`, `utils.sh` | Unchanged reference symlinks to shared configuration and utilities |
| `docker_*.sh`, `run_jupyter.sh` | Reference Docker build, shell, command, notebook, and lifecycle scripts |

## API and Assumptions

The game rules come directly from
[`game_examples.py`](../Implement_MonteCarlo_Tree_Search_and_Alpha_Zero/game_examples.py).

| Function | Output | Convention |
| :--- | :--- | :--- |
| `encode_state(game, state)` | Fresh flat `float32` array | Own pieces `+1`, opponent pieces `-1`, empty cells `0` |
| `get_legal_action_mask(game, state, action_size)` | Boolean array of fixed action size | True for legal indices; all False at terminality |
| `get_terminal_value(game, state)` | `Optional[float]` | Player-to-move outcome; `None` if unfinished |

For Tic-Tac-Toe, input shape is `(9,)` and action size is `9`. Coordinates
remain row-major when perspective changes. Encodings are model inputs, never
replacement states for game methods. Callers use reachable states and legal
transitions; arbitrary malformed boards are outside this small API's contract.
The encoder assumes flat signed board cells, not every conceivable `Game`.

After X wins, the existing turn convention reports O as next, so the value
is `-1.0`. After O wins, it likewise reports `-1.0` for X. A draw is `0.0`.
Alternating-player search negates values between child and parent.
This convention differs from the existing MCTS node's incoming-player values.

## Run Locally

Use a checkout with the `helpers_root` submodule present. Run these commands
from the repository root. The `>` characters below denote shell prompts.

- Install dependencies in your Python environment:

  ```bash
  > python -m pip install -r research/Implement_AlphaZero/requirements.txt \
      jupyterlab
  > export PYTHONPATH="$PWD:$PWD/helpers_root${PYTHONPATH:+:$PYTHONPATH}"
  > export MPLBACKEND=Agg
  ```

- Run the core tests using the repository's shared pytest configuration:

  ```bash
  > python -m pytest -o addopts='' \
      research/Implement_AlphaZero/test/test_alphazero_utils.py -q
  ```

  The override avoids requiring the optional `--new-first` pytest plugin.
  Keep the root `conftest.py` enabled: the shared test base uses its fixtures.

- Open the API notebook:

  ```bash
  > jupyter lab research/Implement_AlphaZero/alphazero.API.ipynb
  ```

- Execute the notebook top to bottom, saving the executed copy outside the source:

  ```bash
  > jupyter nbconvert --to notebook --execute \
      research/Implement_AlphaZero/alphazero.API.ipynb \
      --output alphazero.API.executed.ipynb --output-dir /tmp
  ```

## Docker Commands

The local `Dockerfile` uses `python:3.12-slim` for CPU execution, with Git,
CA certificates, Jupyter, and the dependencies in `requirements.txt`.
The bash scripts use the shared project template, mount the complete checkout,
and set `PYTHONPATH` for the repository and `helpers_root`.

| Command | Description |
| :--- | :--- |
| `docker_build.sh` | Build the project's CPU image |
| `docker_jupyter.sh` | Launch Jupyter with the repository mounted |
| `docker_bash.sh` | Open a shell in the image |
| `docker_cmd.sh` | Execute a command in the image |
| `docker_exec.sh` | Open a shell in a running container |
| `docker_clean.sh` | Clean project Docker resources; inspect help before use |
| `docker_push.sh` | Publish the configured image when explicitly requested |
| `run_jupyter.sh` | Start Jupyter inside the container |
| `version.sh` | Report installed package versions during the build |

- Build and launch from the project directory:

  ```bash
  > cd research/Implement_AlphaZero
  > ./docker_build.sh
  > ./docker_jupyter.sh
  ```

- Run core tests through the project's container:

  ```bash
  > ./docker_cmd.sh 'cd /git_root && MPLBACKEND=Agg python -m pytest -o addopts="" research/Implement_AlphaZero/test/test_alphazero_utils.py -q'
  ```

- From the repository root in the shared development environment, explicitly
  run Docker integration checks (requires a working Docker daemon):

  ```bash
  > python -m pytest -o addopts='' \
      research/Implement_AlphaZero/test/test_docker_template.py -s -v
  ```

See the [shared Docker guide](../../class_project/project_template/docker_scripts.README.md)
for additional script options.

## Validation

| Check | Result |
| :--- | :--- |
| Core tests in the local development environment | 15 passed |
| Core tests in the built CPU image | 15 passed |
| Notebook execution in fresh local and Docker kernels | Passed |
| Docker integration checks | Build, shell, command, and notebook checks passed |
| Notebook schema | Valid |
| Python formatting, shell syntax, and symlink targets | Passed |

The shared helpers emit deprecation warnings for `datetime.utcnow()` and the
root pytest hook's legacy `path` argument. The latter is why this project's
requirements constrain pytest below version 9. Headless test runs use
`MPLBACKEND=Agg`, because the shared test base initializes Matplotlib during
cleanup even though these utilities do not plot anything.
