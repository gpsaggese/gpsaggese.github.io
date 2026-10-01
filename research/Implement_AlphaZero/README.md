# Implement AlphaZero

Game representations, a CPU policy/value network, PUCT search, self-play,
replay-based training, and paired-seat evaluation for AlphaZero.
The API notebook explains board encodings, legal priors, terminal outcomes,
search statistics, action selection, self-play, replay updates, and network
checkpoints with small Tic-Tac-Toe examples.

The project reuses the `Game` interface and board games from the
[MCTS project](../Implement_MonteCarlo_Tree_Search_and_Alpha_Zero/README.md).
Game rules remain separate from representation and evaluation. All AlphaZero
implementation lives in `alphazero_utils.py`. The single `alphazero.API.ipynb`
tutorial progresses from board conventions and the uniform evaluator to
hand-traced search, a complete game played by search, and fitting a network
to four controlled examples, collecting self-play games, training from replay,
reloading a saved network, and comparing fixed agents in both seats.
The companion [tutorial article](alphazero.tutorial.md) explains the complete
implementation and records the measured comparison and its limitations.

## Structure of the Directory

| Directory | Description |
| :--- | :--- |
| `test/` | Unit tests and the reference project's Docker notebook test pattern |

## Description of Files

| File | Description |
| :--- | :--- |
| `alphazero_utils.py` | Board representations, evaluators, PUCT, CPU policy/value network, minibatch learning, self-play, replay training, checkpoints, and paired-seat evaluation |
| `alphazero.API.ipynb` | Guided API tour with search traces, a complete game, controlled fitting, self-play targets, replay training, checkpoint reloads, and baseline comparisons |
| `alphazero.tutorial.md` | End-to-end explanation, reproducible evaluation setup, measured results, and limitations |
| `test/test_alphazero_utils.py` | Representation, evaluation, search, network, learning, self-play, replay, training-loop, and checkpoint tests |
| `test/test_docker_template.py` | Docker build/script checks and notebook execution using shared helpers |
| `requirements.txt` | Project dependencies, PyTorch 2.6.0, and `pytest<9` for shared test hooks |
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

## Policy/Value Evaluation

An evaluator accepts `(game, state)` and returns a `PolicyValuePrediction`.
The `PolicyValueEvaluator` type alias describes this callable signature.
Pass the game's original state, just as with the representation utilities.

| API | Output or behavior |
| :--- | :--- |
| `normalize_policy(weights, legal_action_mask)` | Fresh `float64` legal probabilities from finite nonnegative weights |
| `PolicyValuePrediction(policy, value)` | Validated policy vector and finite scalar value in `[-1, 1]` |
| `UniformEvaluator(action_size)` | Callable baseline with equal legal priors and neutral nonterminal values |

The normalizer masks illegal actions before summing legal weights. It uses
uniform legal probabilities if their mass is zero, and an all-zero vector
if no action is legal. Scaling by the largest legal weight keeps normalization
finite even when a direct sum would overflow. Inputs are never modified.
Weights must be nonnegative; raw network logits are not accepted.

The prediction constructor checks numerical constraints and copies its policy.
The evaluator ensures that priors match the fixed action space, assign zero
probability to illegal actions, and sum to one for unfinished games. A terminal
prediction instead has an all-zero policy and an exact player-to-move outcome.
Do not sample a move from a terminal policy.

`UniformEvaluator` returns `0.0` for unfinished games as a neutral estimate,
not a claim that the game will draw. It uses the exact outcome at terminality:
after either player wins at Tic-Tac-Toe, the next player has value `-1.0`.
This deterministic baseline performs no search, random sampling, or learning.

```python
import research.Implement_AlphaZero.alphazero_utils as rialzut
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game_examples as rimtsaazge

game = rimtsaazge.TicTacToe()
evaluator = rialzut.UniformEvaluator(9)
prediction = evaluator(game, game.get_initial_state())
# prediction.policy has nine entries of 1/9; prediction.value is 0.0.
```

## PUCT Search

`build_search_tree()` searches a reachable nonterminal position using the
supplied evaluator. It supports strictly alternating two-player, zero-sum,
deterministic games with the existing integer-action interface. The sign
convention does not apply to extra-turn or single-player games.

| API | Behavior |
| :--- | :--- |
| `AlphaZeroNode(state, prior)` | Holds state, incoming prior, children, visit count, and value sum |
| `get_puct_scores(node, exploration_constant)` | Returns each child's selection score from the parent's perspective |
| `build_search_tree(game, state, evaluator, action_size=..., num_simulations=...)` | Builds a fresh tree and returns its root |
| `get_visit_policy(root, action_size)` | Normalizes root child visits into a fixed-size policy |

Selection scores an action as
`-child.mean_value + c * child.prior * sqrt(max(1, parent.visit_count)) / (1 + child.visit_count)`.
The child mean is negated because it favors the opponent. The count floor of
one makes the first selection honor the priors. Unvisited children have mean
zero, and equal scores choose the lowest action index. Zero-prior actions are
legal children, but receive no exploration bonus; PUCT does not guarantee
that every action will be visited within a finite budget.

Root expansion evaluates the current state and creates all legal children
before the counted simulations. Its initial value estimate is not backed up.
Each simulation descends to a leaf, expands it if unfinished, and backs up a
value with alternating signs. Exact terminal outcomes bypass the evaluator.
Nonterminal priors are masked and normalized before expansion. Search performs
no random rollouts or training. Root noise is disabled by default; enable it
explicitly for exploration during self-play.

Root visits and the sum of root child visits both equal `num_simulations`.
A nonroot node's first visit evaluates that node without selecting one of
its children; its own count therefore need not equal its children's counts.
At its default temperature of one, `get_visit_policy()` returns root priors
with zero simulations or normalized child counts with positive simulations.
These probabilities can be used
with `np.argmax()` to choose the most-visited action. Terminal roots are
rejected; check `game.is_terminal()` before requesting another move.

```python
import numpy as np

state = (1, 1, 0, -1, -1, 0, 0, 0, 0)
root = rialzut.build_search_tree(
    game, state, evaluator, action_size=9, num_simulations=100
)
policy = rialzut.get_visit_policy(root, 9)
move = int(np.argmax(policy))
# X selects action 2, completing the top row.
```

The uniform evaluator supplies neutral estimates at unfinished leaves. Search
can discover tactical outcomes by reaching terminal positions, but a finite
budget does not guarantee optimal play. With noise disabled, repeating a search with the same
deterministic evaluator and configuration produces the same result. With noise
enabled, reproduce it by starting from the same explicit RNG state. Each
call builds a new tree; it does not retain statistics across moves.

## Policy/Value Network and Supervised Learning

`PolicyValueNetwork` is a CPU float32 MLP with two shared ReLU layers,
a linear policy head, and a tanh value head. It accepts one board or a batch.
The explicit initialization seed preserves the caller's CPU random stream.
Its board and action sizes are independent: Tic-Tac-Toe uses 9 and 9, while
Connect Four uses 42 and 7. The model has no dropout or batch normalization.

| API | Behavior |
| :--- | :--- |
| `PolicyValueNetwork(input_size, action_size, hidden_size=..., seed=...)` | Construct the network with explicit width and initialization seed |
| `network(encoded_tensor)` | Return unmasked logits `(A,)` or `(B, A)` and values `()` or `(B,)` |
| `NetworkEvaluator(network)` | Encode the original state, mask illegal logits before softmax, and return a `PolicyValuePrediction` |
| `train_batch(network, optimizer, inputs, policies, values, l2_coefficient=...)` | Update the network once and return pre-update loss components |

Network inference creates no gradient graph and preserves model mode and
existing gradients. The evaluator retains the model by reference, so updates
are immediately visible to subsequent searches. Terminal states bypass the
network and return exact outcomes with zero policies.

`train_batch()` accepts NumPy arrays with shapes `(B, input_size)`,
`(B, action_size)`, and `(B,)`. Policy targets are nonnegative distributions
summing to one, including soft targets; value labels lie in `[-1, 1]` and
refer to each board's player to move. Callers ensure targets are legal, since
the training function receives encodings rather than game rules. Terminal
all-zero policies are not valid training distributions.

The objective is mean policy cross-entropy plus mean squared value error
plus `l2_coefficient * sum(parameter ** 2)`. The L2 term includes biases.
Cross-entropy uses unmasked logits over the full action space, penalizing
probability on illegal actions when their target mass is zero. Keep optimizer
weight decay zero to avoid double regularization, and reuse one optimizer
across batches to preserve its state. Returned `loss`, `policy_loss`,
`value_loss`, and weighted `l2_loss` are measured before the update.

```python
import numpy as np
import torch

network = rialzut.PolicyValueNetwork(9, 9, hidden_size=32, seed=7)
network_evaluator = rialzut.NetworkEvaluator(network)
optimizer = torch.optim.Adam(network.parameters(), lr=0.02, weight_decay=0.0)
# X can win immediately at action 2, so its value target is +1.
state = (1, 1, 0, -1, -1, 0, 0, 0, 0)
inputs = rialzut.encode_state(game, state)[None, :]
policies = np.eye(9, dtype=np.float32)[[2]]
values = np.array([1.0], dtype=np.float32)
metrics = rialzut.train_batch(
    network, optimizer, inputs, policies, values, l2_coefficient=1e-4
)
prediction = network_evaluator(game, state)
```

The notebook fits four hand-built positions covering wins for either player,
a draw, and a forced loss. It shows before/after probabilities, value estimates,
loss curves, and integration with PUCT. Fitting these examples demonstrates
learning mechanics; it does not establish generalization or playing strength.

## Self-Play Data Collection

`play_self_play_game()` plays both seats with a fixed evaluator and collects
training records after a complete game. It builds a fresh tree each move and
never applies optimizer updates. It supports the same strictly alternating,
two-player, zero-sum game interface as PUCT.

| API | Behavior |
| :--- | :--- |
| `TrainingExample(state, policy, value)` | Owns the original pre-move state, a copied policy, and a final outcome label |
| `SelfPlayConfig(...)` | Validates budgets, temperature schedule, root noise, and maximum game length |
| `SelfPlayResult` | Contains aligned `examples` and `moves`, plus `final_state` and `winner` |
| `play_self_play_game(game, evaluator, config, rng, action_size=...)` | Starts at the game's initial state and returns a completed result |

At every root, optional noise mixes each legal prior as
`(1 - root_noise_fraction) * prior + root_noise_fraction * noise`.
The Dirichlet sample has one entry per legal action, drawn in ascending action
order with concentration `dirichlet_alpha`. Mixing happens after root expansion
and before the first simulation; deeper priors are untouched. Noise remains
active at every new root, including after the temperature schedule ends.
Ordinary `build_search_tree()` calls retain noise fraction zero. Enabled noise
requires an explicit `np.random.Generator`; disabled noise consumes no draws.

`get_visit_policy(..., temperature=tau)` uses visit weights proportional to
`count ** (1 / tau)` for positive temperature. Log-space scaling avoids overflow
for very small temperatures. Temperature zero yields a one-hot policy on the
most-visited action, with lowest-index ties. Unvisited actions retain zero mass
when any visits exist. Standalone zero-budget search uses the same temperature
rule on priors; self-play requires a positive simulation budget.

The first `temperature_moves` plies use the configured temperature; subsequent
plies use zero. A ply is one move by either player. The saved policy is the
**temperature-adjusted policy used to choose the move**, including one-hot
policies for greedy moves. Positive temperature samples with the supplied RNG;
zero temperature uses argmax without consuming a sampling draw. The collector
shares that generator between root noise and move sampling and never seeds a
global random stream. Resetting the seed reproduces a game with a deterministic
evaluator and the same settings; reusing a generator advances its stream.

After terminality, each record receives `winner * player_to_move(record.state)`.
Decisive outcomes therefore alternate signs across plies; draws label every
record zero. The terminal board is stored separately, not as an all-zero policy
example. An initially terminal game returns empty examples and moves.
If a game is unfinished at `max_moves`, collection raises and returns no
partially labeled data. The evaluator must remain fixed and side-effect free
throughout collection; the supplied network evaluator preserves parameters,
mode, and existing gradients.

```python
config = rialzut.SelfPlayConfig(
    num_simulations=16, temperature=1.0, temperature_moves=3,
    dirichlet_alpha=0.3, root_noise_fraction=0.25, max_moves=9,
)
result = rialzut.play_self_play_game(
    game, network_evaluator, config, np.random.default_rng(7), action_size=9
)
# Align each pre-move board with its policy and final player-relative outcome.
inputs = np.stack([rialzut.encode_state(game, e.state) for e in result.examples])
policies = np.stack([e.policy for e in result.examples])
values = np.array([e.value for e in result.examples], dtype=np.float32)
```

The notebook traces one seeded game's moves, legal policies, and labels,
checks reproducibility, and assembles batch arrays without updating the model.
Collected games establish data alignment, not playing strength.

## Training From Replay

`ReplayBuffer(capacity)` keeps the newest positions in FIFO order. Capacity
counts positions rather than games, so old games can be partially evicted.
`add()` copies each record and checks consistent board/action dimensions.
`get_examples()` returns independent copies in oldest-to-newest order.
`sample(batch_size, rng)` samples uniformly **with replacement** and returns
independent copies, including when the same position is selected twice. This
allows full minibatches while a new buffer contains only a few examples.
Sampling an empty buffer is rejected.

| API | Behavior |
| :--- | :--- |
| `TrainingConfig(...)` | Positive iteration/game/update/batch budgets, L2 coefficient, and `SelfPlayConfig` |
| `TrainingMetrics` | Iteration, games, generated positions, retained replay size, updates, and mean loss components |
| `train_alphazero(game, network, optimizer, replay, config, rng)` | Collect complete games, add their targets to replay, then update from sampled minibatches |

All games within an iteration use fixed network parameters. Updates happen
only after collection, and the next iteration's games use the updated model.
The network, optimizer, replay, and NumPy generator are supplied by the caller
and updated in place. Reuse all four to continue an in-memory run without
resetting Adam moments, retained examples, or the sampling stream. The generator
is shared by root noise, action sampling, and replay sampling. Model
initialization has its own explicit seed. Existing replay must contain targets
from the same game and match the model's board and action dimensions.

The optimizer must own exactly the model's parameters and use zero weight decay;
L2 is included explicitly in `train_batch()`. Metrics contain one record per
iteration, numbered from one for each function call. `generated_examples`
counts positions before FIFO eviction. Loss fields average **pre-update**
minibatch losses within that iteration. Training data changes between updates,
so these losses need not decrease monotonically or measure playing strength.
Errors propagate; already completed games and updates are retained in memory.

```python
network = rialzut.PolicyValueNetwork(9, 9, hidden_size=32, seed=21)
optimizer = torch.optim.Adam(network.parameters(), lr=0.01, weight_decay=0.0)
replay = rialzut.ReplayBuffer(32)
rng = np.random.default_rng(21)
training_config = rialzut.TrainingConfig(
    iterations=4, games_per_iteration=2, updates_per_iteration=6,
    batch_size=16, l2_coefficient=1e-4,
    self_play=rialzut.SelfPlayConfig(
        num_simulations=16, temperature=1.0, temperature_moves=3,
        dirichlet_alpha=0.3, root_noise_fraction=0.25, max_moves=9,
    ),
)
history = rialzut.train_alphazero(
    game, network, optimizer, replay, training_config, rng
)
```

The notebook runs this budget on CPU: eight games and 24 optimizer steps,
with at most 32 retained positions. It displays replay contents, iteration
metrics, and loss curves. This is an integration demonstration, not evidence
of competitive play. Pass `show_progress=False` to suppress the progress bar.

## Network Checkpoints

`save_checkpoint(network, path)` saves architecture dimensions and parameter
tensors in a versioned payload. It serializes to a sibling temporary file and
atomically replaces the destination after success, preserving the previous
file if serialization fails. The destination directory must already exist.

`load_checkpoint(path)` uses CPU mapping and `weights_only=True`, checks the
format version, requires matching parameter names/shapes, and rejects nonfinite
weights. It reconstructs an independent CPU float32 model in evaluation mode,
without gradients and without advancing the caller's CPU random stream.

```python
import os
import tempfile

checkpoint_dir = tempfile.mkdtemp(prefix="alphazero-model-")
checkpoint_path = os.path.join(checkpoint_dir, "alphazero.policy_value.pt")
rialzut.save_checkpoint(network, checkpoint_path)
restored_network = rialzut.load_checkpoint(checkpoint_path)
restored_evaluator = rialzut.NetworkEvaluator(restored_network)
```

These are **model checkpoints** for inference or fresh optimization. They do
not contain optimizer moments, replay, RNG state, gradients, or training
progress, so loading does not exactly resume a training run. The notebook
checks identical predictions and deterministic PUCT policies after reloading.
Its checkpoint lives outside the source tree in the displayed temporary path.

## Evaluating Fixed Agents

`EvaluationAgent` accepts `(game, state, rng)` and returns an integer legal
move. The evaluation factories keep network parameters fixed and use no
self-play noise or temperature sampling:

| Factory | Behavior |
| :--- | :--- |
| `make_random_agent()` | Wrap the original uniform-random legal player |
| `make_mcts_agent(num_simulations)` | Wrap the original rollout MCTS, retaining UCT constant `sqrt(2)` and its tie behavior |
| `make_minimax_agent()` | Exact minimax through the original alpha-beta implementation, caching actions by game object and state |
| `make_policy_agent(evaluator)` | Highest legal policy probability; lowest-index ties |
| `make_search_agent(evaluator, action_size=..., num_simulations=...)` | PUCT with noise disabled, greedy visit selection, and lowest-index ties |

Reference random/MCTS functions use Python's global RNG. Their serial adapter
seeds each call from the supplied NumPy generator and restores the Python RNG
state afterward, including on errors. Do not run these adapters concurrently
with other Python global-RNG consumers. Exact search is practical here for
Tic-Tac-Toe; it is not a scalable default for larger games. Cached game rules
must remain unchanged.

`evaluate_agent(game, agent, opponent, seeds=..., max_moves=...)` plays the
candidate as X and then O for each distinct nonnegative seed. Each seed/seat
pair derives two independent streams using `SeedSequence([seed, seat_index])`:
one for the candidate and one for the opponent. Matching experiments therefore
use the same opponent seed schedule, without coupling it to a candidate's
random consumption. All moves are checked for legality. An unfinished game at
the move limit raises rather than being scored as a draw.

`EvaluationResult.games` retains seeds, seats, move sequences, terminal states,
winners, and candidate-relative outcomes. `summary()` returns game count,
wins, draws, losses, and `score_rate = (wins + 0.5 * draws) / games`.
Use `summary(agent_player=1)` for X and `summary(agent_player=-1)` for O.

```python
candidate = rialzut.make_search_agent(
    rialzut.NetworkEvaluator(restored_network), action_size=9, num_simulations=64
)
result = rialzut.evaluate_agent(
    game, candidate, rialzut.make_random_agent(), seeds=list(range(10)), max_moves=9
)
print(result.summary())
print(result.summary(agent_player=-1))
```

The notebook compares random play, uniform PUCT, original rollout MCTS, and
untrained/trained policy-only and network-guided PUCT agents. Each faces common
random and exact-play opponents, with ten games per seat and 64 simulations per
move for search candidates. The trained model comes from the notebook's eight
games and 24 updates; untrained controls use the same initial seed and width.
All per-seat results and budgets are displayed. The
[tutorial article](alphazero.tutorial.md) records the observed results.

These are small descriptive experiments with one network initialization.
Different seeds cannot vary a fully deterministic matchup, and repeated
identical games are not independent evidence. Exact minimax also uses one tie
convention; a draw against that opponent does not prove a candidate is unbeatable
against every strategy. Equal simulation counts are not equal runtime or equal
compute: rollout MCTS, neural PUCT, and exhaustive minimax do different work.

## Run Locally

Use a checkout with the `helpers_root` submodule present. Run these commands
from the repository root. The `>` characters below denote shell prompts.

- Install dependencies in your Python environment:

  ```bash
  > python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
  > python -m pip install -r research/Implement_AlphaZero/requirements.txt \
      jupyterlab
  > export PYTHONPATH="$PWD:$PWD/helpers_root${PYTHONPATH:+:$PYTHONPATH}"
  > export MPLBACKEND=Agg
  > export OMP_NUM_THREADS=1
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
It installs PyTorch 2.6.0 from the CPU wheel index before resolving the remaining
requirements, so CUDA libraries are unnecessary. Rebuild after dependency changes.
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
  > ./docker_cmd.sh 'cd /git_root && OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -o addopts="" research/Implement_AlphaZero/test/test_alphazero_utils.py -q'
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
| Core tests in the local development environment | 119 passed: 106 existing tests and 13 evaluation/adapter/scoring tests |
| Core tests in the CPU image | 119 passed with PyTorch 2.6.0+cpu |
| Notebook execution in fresh local and Docker kernels | Passed |
| Docker integration checks | Command and notebook execution passed; image build and shared shell checks passed previously |
| Overall and per-seat evaluation tables | Local and Docker results match exactly |
| Notebook schema | Valid |
| Python formatting, shell syntax, and symlink targets | Passed |

Evaluator tests cover legal masking, uniform fallback, large and small finite
weights, malformed predictions, independent output arrays, terminal outcomes,
and the different action spaces of Tic-Tac-Toe and Connect Four.
Search tests cover hand-computed PUCT scores, one- and two-ply backups,
terminal evaluator bypass, simulation accounting, zero-budget priors,
determinism, action masking, and immediate wins and forced blocks for both
players. Network tests check shapes, RNG isolation, masking before softmax,
terminal bypass, PUCT integration, hand-computed losses and gradients, L2,
gradient clearing, input validation, and fitting four controlled examples.
Self-play tests cover hand-computed temperatures, legal root-only noise,
seeded sampling, policy/trajectory alignment, wins for both players, draws,
move-limit handling, Connect Four column actions, and unchanged network state.
Training-loop tests cover FIFO bounds, record ownership, replacement sampling,
collection/update ordering, finite metrics, parameter changes, optimizer
continuation, and repeatable seeded runs. Checkpoint tests verify prediction
and search round trips, independent CPU models, atomic replacement, and
rejection of incompatible or nonfinite weights.
Evaluation tests cover known scoring, both seats, reference adapter equivalence,
legal trajectories, separate role RNGs, reproducibility, deterministic inference,
unchanged network state, exact-play draws, and move/seed validation.
The notebook executes the complete implementation, compares seven candidates
against two opponents, and reports budgets and limitations. Its overall and
per-seat evaluation tables match between local and CPU Docker runs.

The shared helpers emit deprecation warnings for `datetime.utcnow()` and the
root pytest hook's legacy `path` argument. The latter is why this project's
requirements constrain pytest below version 9. Headless test runs use
`MPLBACKEND=Agg`, because the shared test base initializes Matplotlib during
cleanup even though these utilities do not plot anything.
