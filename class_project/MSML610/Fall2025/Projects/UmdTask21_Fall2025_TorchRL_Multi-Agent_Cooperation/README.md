# TorchRL + PettingZoo MPE — Multi-Agent Communication (Course Tutorial Project)

This repository is a course-style tutorial project demonstrating how to build an end-to-end **multi-agent reinforcement learning (MARL)** workflow using **TorchRL** and **PettingZoo’s Multi-Agent Particle Environment (MPE)**. The focus is **cooperative learning with communication**, along with practical debugging and evaluation techniques.

---

## Project Objective

Train multiple agents to **collaborate** in an MPE task (e.g., `simple_reference`) and evaluate cooperation using:

* **Task success rate** (goal completion)
* **Communication behavior metrics** (message entropy, message change rate)
* (Optional) **communication ablations** (disable/random comm)

---

## Repository Contents

This project follows the course submission structure:

* `TorchRL_MAC_utils.py`
  Reusable utilities and wrapper logic: environment creation, actor/critic helpers, rollout/training helpers, evaluation, and communication metrics.

* `TorchRL_MAC.API.ipynb` + `TorchRL_MAC.API.md`
  “API tutorial” demonstrating the wrapper layer (contract-style usage) with minimal cells.

* `TorchRL_MAC.example.ipynb` + `TorchRL_MAC.example.md`
  End-to-end training + evaluation example demonstrating the full pipeline and results.

* `Dockerfile`
  Container for reproducible execution.

* `docker_build.sh` / `docker_bash.sh`
  Helper scripts for building and running the Docker container.

---

## Quickstart (Local)

### 1) Create environment

Using conda (recommended):

```bash
conda create -n torchrl-mac python=3.10 -y
conda activate torchrl-mac
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run notebooks

```bash
jupyter notebook
```

Open and run (top-to-bottom):

* `TorchRL_MAC.API.ipynb`
* `TorchRL_MAC.example.ipynb`

**Important:** notebooks are designed to be executed via **Restart & Run All**.

---

## Quickstart (Docker)

### Build the image

Using the helper script (recommended):

```bash
./docker_build.sh
```

Or manually:

```bash
docker build -t torchrl-mac .
```

### Run the container (launch Jupyter)

Using the helper script (recommended):

```bash
./docker_bash.sh
```

Or manually:

```bash
docker run --rm -it -p 8888:8888 torchrl-mac
```

You should see a Jupyter URL printed in the terminal. Open it in your browser and run the notebooks.

> Note: Docker runs on **CPU** by default. Local runs may use hardware acceleration depending on availability.

---

## What You’ll Learn

By completing the notebooks, you will learn:

1. How to integrate **PettingZoo MPE** environments with **TorchRL** wrappers
2. How to implement a **CTDE** (centralized training, decentralized execution) MARL structure
3. How to compute and interpret **communication metrics**:

   * `message_entropy`
   * `message_change_rate`
   * observation-derived variants (`*_obs`) for verification
4. How to validate that “communication is real” using:

   * wiring sanity checks
   * action-vs-observation consistency checks
   * optional ablations (`full_comm`, `disable_comm`, `random_comm`)

---

## Evaluation Metrics (What Matters)

Returns can be noisy or misleading in shaped-reward environments. This project prioritizes:

* **Success** (`success`): binary success based on whether all agents reach within `success_dist` of their goals
* **Goal distances debug** (`distances=[...]` vs `success_dist`) for interpretability
* **Communication structure metrics**:

  * `message_entropy`, `message_change_rate`
  * `message_entropy_obs`, `message_change_rate_obs` (should align if comm is wired correctly)

---

## Notes on the A3C Attempt (Academic Iteration)

We initially implemented:

* **single-worker A3C**
* **multi-worker A3C**

and then performed extensive:

* hyperparameter tuning
* reward shaping experiments (e.g., distance-to-goal shaping)
* communication verification (watching actions/messages, wiring checks)

In multiple runs, we observed a common MARL failure mode: training loss changing without meaningful improvements in **success** or message structure. Based on these diagnostics, we transitioned to the **current stable setup** used in the example notebook, which produced interpretable results.

This iteration process is documented in `TorchRL_MAC.example.md`.

---

## Reproducibility

For reproducible results:

* use fixed seeds (configs expose `seed`)
* run notebooks from a clean kernel state (“Restart & Run All”)
* prefer CPU inside Docker for stability

---

## Common Troubleshooting

### “Success is always 0.0”

Check:

* `success_dist` threshold (too strict thresholds will show 0 success)
* goal distance debug prints (`distances=[...]`)
* ensure you’re evaluating the correct environment version (`simple_reference_v3` vs others)

### Communication metrics look random (high entropy, high change rate)

This can be normal early in training. If entropy/change rate stay high late into training:

* try increasing training duration
* tune entropy coefficient
* verify comm wiring using the sanity check in utils

### TorchRL/PettingZoo wrapper mismatch

This project includes wrapper fallbacks in the utilities to handle minor version differences. If you see spec/key errors:

* confirm installed versions from `pip freeze`
* run the API notebook first (it prints env specs and expected keys)

---
