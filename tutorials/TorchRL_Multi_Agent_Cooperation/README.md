TorchRL + PettingZoo MPE — Multi-Agent Communication

 This folder contains a course-style tutorial project demonstrating an end-to-end multi-agent reinforcement learning (MARL) workflow using TorchRL and PettingZoo (MPE) with communication.


Quick Start

 From the root of the repository, create and activate a virtual environment:
  > conda create -n torchrl-mac python=3.10 -y
  > conda activate torchrl-mac

 Install dependencies:
  > pip install -r requirements.txt

 Launch Jupyter Notebook:
  > jupyter notebook

 Open and run the notebooks in this order:

  i. TorchRL_MAC.API.ipynb
    Start here to understand the wrapper layer and core utilities.

  ii. TorchRL_MAC.example.ipynb
    Run this notebook to execute end-to-end training + evaluation.

 Notebooks are designed to be executed using Restart & Run All.


Docker Quick Start

 Build the Docker image:
 > ./docker_build.sh

 Run the container and launch Jupyter:
 > ./docker_bash.sh

 Open the printed Jupyter URL in your browser and run the notebooks.

 Note: Docker runs on CPU by default.


Project Structure

 TorchRL_MAC_utils.py
  Environment creation, wrapper logic, actor/critic helpers, rollout/training utilities, evaluation, and communication metrics.

 TorchRL_MAC.API.ipynb / .md
  Minimal API-style tutorial demonstrating wrapper contracts.

 TorchRL_MAC.example.ipynb / .md
  Full training + evaluation example with results and diagnostics.

 Dockerfile
  Reproducible container environment.

 docker_build.sh / docker_bash.sh
  Docker helper scripts.

Project Objective

 Train multiple agents to cooperate in an MPE environment (e.g., simple_reference) and evaluate cooperation using:
  i. Task success rate (success)
  ii. Goal distance diagnostics
  iii. Communication behavior metrics:
   > message_entropy
   > message_change_rate
   > observation-derived verification metrics


What This Tutorial Covers

 i. Integration of PettingZoo MPE with TorchRL
 ii. CTDE (Centralized Training, Decentralized Execution)
 iii. Communication verification and debugging
 iv. Interpreting structured metrics beyond raw returns
 v. Practical MARL failure modes and iteration process


Evaluation Metrics

 This project prioritizes structured evaluation over noisy return curves:
  > success — binary success based on goal distance threshold
  > distances=[...] — debug visibility into goal proximity
 Communication structure:
  > message_entropy
  > message_change_rate
  > observation-derived variants (*_obs)


Notes on Training Iteration

 Initial experiments included:
  > Single-worker A3C
  > Multi-worker A3C
  > Reward shaping (distance-to-goal shaping)
  > Communication wiring verification

 Observed MARL failure mode:
  > Loss changing without meaningful improvement in success or structured communication.

 The final notebook reflects the stabilized and interpretable configuration after these iterations.


Reproducibility

 For stable runs:
  > Use fixed seeds (config exposes seed)
  > Execute notebooks from a clean kernel state
  > Prefer CPU inside Docker for deterministic behavior


Troubleshooting

 Success always 0.0
  > Check success_dist
  > Print goal distances
  > Verify environment version (simple_reference_v3)

 Communication metrics look random
  > Increase training duration
  > Tune entropy coefficient
  > Run communication sanity checks

 Wrapper/spec errors
  > Confirm installed versions (pip freeze)
  > Run API notebook first to inspect environment specs