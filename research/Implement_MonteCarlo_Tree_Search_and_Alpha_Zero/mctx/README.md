# Mctx Tic-Tac-Toe

This subproject uses [Mctx](https://github.com/google-deepmind/mctx) to search
batches of tic-tac-toe positions with JAX. It reuses the game rules and
evaluation helpers from the parent MCTS tutorial while keeping the JAX/CUDA
environment in a separate Docker image.

## Files

- `mctx_tic_tac_toe_utils.py`: batched JAX environment model and Mctx player.
- `mctx_tic_tac_toe.ipynb`: executable tutorial and CPU/GPU benchmark.
- `Dockerfile`: Python 3.12 image with Mctx and CUDA-enabled JAX.
- `requirements.txt`: dependencies pinned for the Mctx image.
- `all.learn_mctx_by_building_a_batched_tic_tac_toe_agent.how_to_guide.md`:
  tutorial-style blog.

## Build the Image

From this directory, run:

```bash
./docker_build.sh
```

The image is named `gpsaggese/umd_alphazero_mctx`. CUDA and cuDNN user-space
libraries are installed by `jax[cuda12]`; an NVIDIA CUDA base image is not
required.

## Select a Device

The launch scripts read `MCTX_DEVICE`:

- `auto` (default): use an NVIDIA GPU when the NVIDIA Docker runtime is
  available, otherwise fall back to CPU.
- `gpu`: require an NVIDIA GPU and fail if it is unavailable.
- `cpu`: force JAX to use CPU.

For GPU execution, install the NVIDIA Container Toolkit and use a host NVIDIA
driver compatible with CUDA 12.

```bash
# Automatically select GPU or CPU.
./docker_cmd.sh 'python -c "import jax; print(jax.default_backend(), jax.devices())"'

# Require GPU execution.
MCTX_DEVICE=gpu ./docker_jupyter.sh

# Force the portable CPU fallback.
MCTX_DEVICE=cpu ./docker_jupyter.sh
```

Open the URL printed by Jupyter and run `mctx_tic_tac_toe.ipynb` from top to
bottom. The notebook reports the actual backend and device before running any
searches or benchmarks.

## Design

Mctx expects batched model inputs rather than Python tree nodes. The adapter:

1. Expresses every board from the current player's perspective.
2. Masks occupied cells with negative-infinity policy logits.
3. Applies moves through a JAX-compatible recurrent function.
4. Uses a reward of `1` for a winning move, a discount of `-1` while play
   continues, and a discount of `0` at terminal states.
5. Uses uniform priors and zero nonterminal values so this milestone focuses
   on search rather than neural-network training.

The next AlphaZero milestone can replace those uniform priors and values with
the outputs of a learned policy/value network without changing the Mctx search
interface.
