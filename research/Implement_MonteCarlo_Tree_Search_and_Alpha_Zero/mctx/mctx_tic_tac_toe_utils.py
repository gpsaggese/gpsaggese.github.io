"""
JAX environment model and Mctx search helpers for tic-tac-toe.

The public player returned by `make_mctx_player()` follows the same
`(game, state) -> move` interface as the players in `alphazero_utils.py`.
Mctx itself operates on batches, so `run_mctx_search()` accepts a sequence of
states and searches every position in parallel.

Import as:

import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.mctx.mctx_tic_tac_toe_utils as rimtsaazmmtttu
"""

import functools
import logging
import statistics
import time
from typing import Callable, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import mctx
import numpy as np

import helpers.hdbg as hdbg
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.alphazero_utils as rimtsaazau

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


# Tic-tac-toe has one action for each cell in its 3x3 board.
NUM_ACTIONS = 9

# Mctx simulations used when a caller does not specify a search budget.
DEFAULT_NUM_SIMULATIONS = 128

# Reproducible default PRNG seed for Gumbel MuZero search.
DEFAULT_SEED = 0

# All rows, columns, and diagonals that can win a tic-tac-toe game.
_WIN_LINES = jnp.asarray(
    [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ],
    dtype=jnp.int32,
)


# #############################################################################
# Batched tic-tac-toe model
# #############################################################################


def _has_winning_line(boards: jax.Array, player: int) -> jax.Array:
    """
    Check a batch of canonical boards for a winning line.

    :param boards: board embeddings with shape `[batch_size, 9]`
    :param player: cell value to check, either `1` or `-1`
    :return: Boolean array with shape `[batch_size]`
    """
    line_values = boards[:, _WIN_LINES]
    has_won = jnp.any(jnp.all(line_values == player, axis=-1), axis=-1)
    return has_won


def _is_terminal(boards: jax.Array) -> jax.Array:
    """
    Check a batch of canonical boards for wins or draws.

    :param boards: board embeddings with shape `[batch_size, 9]`
    :return: Boolean array with shape `[batch_size]`
    """
    has_winner = _has_winning_line(boards, 1) | _has_winning_line(boards, -1)
    is_full = jnp.all(boards != 0, axis=-1)
    is_terminal = has_winner | is_full
    return is_terminal


def _get_prior_logits(boards: jax.Array) -> jax.Array:
    """
    Assign uniform logits to legal moves and mask every illegal move.

    Terminal boards use finite dummy logits so normalization stays defined.
    These are absorbing search states, not playable game positions.

    :param boards: board embeddings with shape `[batch_size, 9]`
    :return: logits with shape `[batch_size, 9]`
    """
    is_terminal = _is_terminal(boards)
    is_legal = (boards == 0) & ~is_terminal[:, None]
    logits = jnp.where(is_legal, 0.0, -jnp.inf)
    logits = jnp.where(is_terminal[:, None], 0.0, logits)
    return logits


def _states_to_canonical_boards(
    states: Sequence[rimtsaazau.State],
) -> jax.Array:
    """
    Convert Python states to boards from each current player's perspective.

    On a canonical board, the player about to move always owns the `1` cells
    and the opponent owns the `-1` cells. This lets one recurrent model handle
    both X and O positions.

    :param states: non-empty sequence of nonterminal tic-tac-toe states
    :return: integer JAX array with shape `[batch_size, 9]`
    """
    states = list(states)
    hdbg.dassert(states, "At least one state is required")
    game = rimtsaazau.TicTacToe()
    canonical_boards = []
    for state in states:
        hdbg.dassert_eq(len(state), NUM_ACTIONS)
        hdbg.dassert(
            set(state).issubset({-1, 0, 1}),
            "State contains an invalid cell value: %s",
            state,
        )
        hdbg.dassert(
            not game.is_terminal(state),
            "Mctx search requires a nonterminal root state",
        )
        current_player = game.get_current_player(state)
        canonical_board = np.asarray(state, dtype=np.int32) * current_player
        canonical_boards.append(canonical_board)
    boards = jnp.asarray(np.stack(canonical_boards))
    return boards


def build_mctx_root(
    states: Sequence[rimtsaazau.State],
) -> Tuple[mctx.RootFnOutput, jax.Array]:
    """
    Build the root model inputs for a batch of tic-tac-toe positions.

    The tutorial intentionally uses uniform priors and zero nonterminal values
    so that it demonstrates Mctx's search mechanics without introducing a
    neural network yet.

    :param states: batch of nonterminal tic-tac-toe states
    :return: `(root, invalid_actions)` suitable for an Mctx policy
    """
    boards = _states_to_canonical_boards(states)
    prior_logits = _get_prior_logits(boards)
    batch_size = boards.shape[0]
    root = mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=jnp.zeros([batch_size], dtype=jnp.float32),
        embedding=boards,
    )
    invalid_actions = ~jnp.isfinite(prior_logits)
    return root, invalid_actions


def mctx_recurrent_fn(
    params,
    rng_key: jax.Array,
    actions: jax.Array,
    embeddings: jax.Array,
) -> Tuple[mctx.RecurrentFnOutput, jax.Array]:
    """
    Apply one batched tic-tac-toe transition for Mctx.

    Each input embedding is canonicalized for the player taking `actions`.
    After applying that player's mark, the board is negated so the returned
    embedding is canonicalized for the opponent. A nonterminal discount of
    `-1` makes value backup switch perspective at every ply; terminal nodes use
    discount `0` because their reward completely determines the outcome.
    Repeated expansion of a terminal node preserves its board and returns zero
    reward and discount, making it an absorbing state inside the search tree.

    :param params: unused model parameters required by the Mctx API
    :param rng_key: unused PRNG key required by the Mctx API
    :param actions: integer actions with shape `[batch_size]`
    :param embeddings: canonical boards with shape `[batch_size, 9]`
    :return: `(recurrent_output, next_embeddings)`
    """
    del params, rng_key
    was_terminal = _is_terminal(embeddings)
    moves = jax.nn.one_hot(actions, NUM_ACTIONS, dtype=embeddings.dtype)
    boards_after_move = embeddings + jnp.where(was_terminal[:, None], 0, moves)
    player_won = _has_winning_line(boards_after_move, 1)
    board_is_full = jnp.all(boards_after_move != 0, axis=-1)
    is_terminal = was_terminal | player_won | board_is_full
    rewards = (player_won & ~was_terminal).astype(jnp.float32)
    discounts = jnp.where(is_terminal, 0.0, -1.0).astype(jnp.float32)
    next_embeddings = jnp.where(
        was_terminal[:, None], embeddings, -boards_after_move
    )
    recurrent_output = mctx.RecurrentFnOutput(
        reward=rewards,
        discount=discounts,
        prior_logits=_get_prior_logits(next_embeddings),
        value=jnp.zeros_like(rewards),
    )
    return recurrent_output, next_embeddings


# #############################################################################
# Mctx search
# #############################################################################


@functools.partial(jax.jit, static_argnames=("num_simulations",))
def _run_gumbel_muzero_policy(
    root: mctx.RootFnOutput,
    invalid_actions: jax.Array,
    rng_key: jax.Array,
    *,
    num_simulations: int,
) -> mctx.PolicyOutput:
    """Run a compiled Gumbel MuZero search for one fixed batch shape."""
    max_num_considered_actions = min(NUM_ACTIONS, num_simulations)
    policy_output = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=mctx_recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        max_depth=NUM_ACTIONS,
        max_num_considered_actions=max_num_considered_actions,
        gumbel_scale=0.0,
    )
    return policy_output


def run_mctx_search(
    states: Sequence[rimtsaazau.State],
    *,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> mctx.PolicyOutput:
    """
    Search a batch of tic-tac-toe states with Mctx.

    The first call for a new batch shape and simulation count includes XLA
    compilation. Later calls reuse the compiled executable.

    :param states: batch of nonterminal tic-tac-toe states
    :param num_simulations: tree expansions performed for each state
        - Default: `DEFAULT_NUM_SIMULATIONS`
    :param seed: JAX PRNG seed
        - Default: `DEFAULT_SEED`
    :return: Mctx actions, action weights, and search trees
    """
    hdbg.dassert_lt(0, num_simulations, "num_simulations must be positive")
    root, invalid_actions = build_mctx_root(states)
    rng_key = jax.random.PRNGKey(seed)
    policy_output = _run_gumbel_muzero_policy(
        root,
        invalid_actions,
        rng_key,
        num_simulations=num_simulations,
    )
    return policy_output


def make_mctx_player(
    *,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> Callable[[rimtsaazau.Game, rimtsaazau.State], rimtsaazau.Move]:
    """
    Build a seeded Mctx player compatible with `alphazero_utils.play_game()`.

    :param num_simulations: Mctx tree expansions run for every move
        - Default: `DEFAULT_NUM_SIMULATIONS`
    :param seed: seed used for the first move; later moves increment it
        - Default: `DEFAULT_SEED`
    :return: `(game, state) -> move` player function
    """
    hdbg.dassert_lt(0, num_simulations, "num_simulations must be positive")
    next_seed = seed

    def player(
        game: rimtsaazau.Game, state: rimtsaazau.State
    ) -> rimtsaazau.Move:
        nonlocal next_seed
        hdbg.dassert(
            isinstance(game, rimtsaazau.TicTacToe),
            "The Mctx model currently supports TicTacToe only",
        )
        policy_output = run_mctx_search(
            [state], num_simulations=num_simulations, seed=next_seed
        )
        next_seed += 1
        move = int(np.asarray(policy_output.action)[0])
        hdbg.dassert_in(move, game.get_legal_moves(state))
        return move

    return player


def benchmark_mctx_search(
    states: Sequence[rimtsaazau.State],
    *,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    num_repeats: int = 3,
) -> Dict[str, object]:
    """
    Benchmark compiled Mctx search after a separate JIT warm-up.

    :param states: batch of states searched together
    :param num_simulations: tree expansions performed for each state
        - Default: `DEFAULT_NUM_SIMULATIONS`
    :param num_repeats: number of synchronized timed runs
        - Default: `3`
    :return: backend, device, batch size, median time, and throughput
    """
    hdbg.dassert_lt(0, num_repeats, "num_repeats must be positive")
    states = list(states)
    # Compile once before recording execution time.
    warmup_output = run_mctx_search(
        states, num_simulations=num_simulations, seed=DEFAULT_SEED
    )
    warmup_output.action.block_until_ready()
    elapsed_times = []
    for repeat_idx in range(num_repeats):
        start_time = time.perf_counter()
        policy_output = run_mctx_search(
            states,
            num_simulations=num_simulations,
            seed=DEFAULT_SEED + repeat_idx + 1,
        )
        policy_output.action.block_until_ready()
        elapsed_times.append(time.perf_counter() - start_time)
    median_seconds = statistics.median(elapsed_times)
    results: Dict[str, object] = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "batch_size": len(states),
        "num_simulations": num_simulations,
        "median_seconds": median_seconds,
        "positions_per_second": len(states) / median_seconds,
    }
    _LOG.info("Mctx benchmark: %s", results)
    return results
