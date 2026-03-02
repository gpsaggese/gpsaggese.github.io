# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a FilterPy tutorial project demonstrating Bayesian state estimation with financial applications. It covers four Kalman filter variants: Linear KF, Extended KF (EKF), Unscented KF (UKF), and Ensemble KF (EnKF).

## Architecture

- `filterpy_api_utils.py` - All widget/plotting logic; the notebook calls only these functions
- `filterpy.api.ipynb` - Main Jupyter notebook (do not put logic here; call `filterpy_api_utils`)
- `filterpy.api.md` - Script describing the notebook cell-by-cell structure
- `filterpy.api.py` - Python script version of the notebook

## Docker Workflow

```bash
# Build the Docker image
./docker_build.sh

# Launch Jupyter Lab (default port 8888)
./docker_jupyter.sh

# Launch on a custom port with a specific directory mounted
./docker_jupyter.sh -p 8889 -d /path/to/notebooks

# Run an arbitrary command in the container
./docker_cmd.sh python filterpy_api_utils.py
```

The Docker container mounts:
- Current directory → `/curr_dir`
- Git root → `/git_root`
- `PYTHONPATH` set to `/git_root:/git_root/helpers_root`

## Dependencies

Python packages (see `requirements.txt`): `filterpy`, `ipywidgets`, `matplotlib`, `numpy`, `pandas`, `seaborn`, plus `jupyterlab`.

## Code Conventions

- All interactive widget and plotting logic belongs in `filterpy_api_utils.py`, not in the notebook cells
- Each notebook cell calls a single function from `filterpy_api_utils.py`
- Plots use fixed axis limits to prevent layout jumps when widget sliders change
- Interactive widgets use `ipywidgets` sliders connected via `widgets.interactive_output`
