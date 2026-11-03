# Gymnasium Tutorial

- This folder teaches the `Gymnasium` API for reinforcement learning environments
  through runnable notebooks in a Docker container

## Quick Start

- Build the image and launch Jupyter:

  ```bash
  > cd tutorials/gymnasium
  > ./docker_build.sh
  > ./docker_jupyter.sh
  ```

- The image is the shared one built from `class_project/project_template`. Each
  notebook installs its pinned `gymnasium` and `pygame-ce` in its first code cell,
  so the first run needs network access
  - The example notebook also installs a CPU-only `torch` and `stable-baselines3`:
    about 200 MB, 1 to 3 minutes on the first run

- Work through the notebooks in order:
  1. [`gymnasium.01.API.Env.ipynb`](gymnasium.01.API.Env.ipynb):
     `reset()`, `step()`, and the episode loop
  2. [`gymnasium.02.API.Registry.ipynb`](gymnasium.02.API.Registry.ipynb):
     `gym.make()`, `gym.register()`, and `EnvSpec`
  3. [`gymnasium.03.API.Spaces.ipynb`](gymnasium.03.API.Spaces.ipynb):
     every `Space` class, seeding, and flattening
  4. [`gymnasium.04.API.Wrappers.ipynb`](gymnasium.04.API.Wrappers.ipynb):
     observation, action, and reward wrappers, and how they stack
  5. [`gymnasium.05.API.Vector.ipynb`](gymnasium.05.API.Vector.ipynb):
     batched envs, autoreset modes, and `AsyncVectorEnv`
  6. [`gymnasium.example.ipynb`](gymnasium.example.ipynb):
     a custom env, `check_env()`, and a `PPO` agent against a random baseline

- [`gymnasium_utils.py`](gymnasium_utils.py): helper functions used by the notebooks

- For more on the Docker build system, see the
  [project template README](/class_project/project_template/docker_scripts.README.md)

## Changelog

- 2026-11-03: Added the end-to-end example notebook
- 2026-11-02: Installed `gymnasium` in the notebooks and switched to the shared
  template image
- 2026-11-02: Fixed and cleaned up the 5 API notebooks, added utils and tests
- 2026-06-11: Initial release
