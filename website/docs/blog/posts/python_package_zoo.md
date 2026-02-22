---
title: "Python Package Tools in 30 mins"
authors:
  - gpsaggese
date: 2026-02-14
description:
categories:
  - Python
---

TL;DR: A comprehensive guide to Python package managers and virtual environment
tools, from built-in solutions like `pip` and `venv` to modern alternatives like
`Poetry` and `uv`.

<!-- more -->

- Python's ecosystem offers numerous tools for managing packages and virtual
  environments
- Choosing the right tool depends on:
  - Project requirements
  - Need for simple dependency management
  - Cross-language support requirements
  - Performance needs
- This guide organizes tools by category to help select the best fit for your
  workflow

## Built-In and Standard Tools

### `pip`
- **What it does**: Default Python package installer that downloads and installs
  packages from PyPI (Python Package Index)
- Installs packages from PyPI
- Works with `requirements.txt` for dependency specifications
  ```bash
  # Install a single package from PyPI
  > pip install requests

  # Install all packages listed in requirements.txt (common for project setup)
  > pip install -r requirements.txt

  # Export all installed packages with their versions to requirements.txt (for sharing/reproducing environment)
  > pip freeze > requirements.txt

  # Install a specific version of a package (for version pinning)
  > pip install requests==2.28.0

  # Upgrade a package to the latest version
  > pip install --upgrade requests

  # Uninstall a package
  > pip uninstall requests
  ```

- **Best for**: Basic package installation and simple projects

### `venv`
- **What it does**: Creates isolated Python environments to separate project
  dependencies and avoid conflicts between projects
- Built-in virtual environment tool (Python 3.3+)
- Creates isolated environments with their own Python interpreter and packages
  ```bash
  # Create a new virtual environment named 'venv' in the current directory
  > python -m venv venv

  # Activate the virtual environment on macOS/Linux (makes Python and pip point to the venv)
  > source venv/bin/activate

  # Activate on Windows
  > venv\Scripts\activate.bat

  # Deactivate the current virtual environment (returns to system Python)
  > deactivate

  # Create a venv without pip (if you want to install pip separately)
  > python -m venv venv --without-pip

  # Remove a virtual environment (just delete the directory)
  > rm -rf venv
  ```

- **Best for**: Basic environment isolation without extra dependencies

### `virtualenv`
- **What it does**: Creates isolated Python environments like `venv`, but with
  more features and flexibility
- Older alternative to `venv` but still maintained and enhanced
- More features including Python version discovery and environment templates
- Slightly faster environment creation
  ```bash
  # Install virtualenv globally (required before first use)
  > pip install virtualenv

  # Create a new virtual environment named 'myenv'
  > virtualenv myenv

  # Create a virtualenv with a specific Python version (if multiple Pythons are installed)
  > virtualenv -p python3.11 myenv

  # Create a virtualenv using system site-packages (inherits globally installed packages)
  > virtualenv --system-site-packages myenv

  # Activate the virtualenv (same as venv)
  > source myenv/bin/activate

  # List available Python interpreters that virtualenv can use
  > virtualenv --discovery cached
  ```

- **Best for**: Advanced virtual environment needs or pre-Python 3.3 projects

## Modern Dependency and Environment Managers

### `pipenv`
- **What it does**: Unifies package management and virtual environment creation
  into a single tool, automatically managing both for you
- Combines `pip` + `venv` functionality
- Uses `Pipfile` and `Pipfile.lock` for deterministic dependency resolution
- Automatic virtual environment management
  ```bash
  # Install a package and automatically create/update Pipfile and Pipfile.lock
  > pipenv install requests

  # Activate a subshell within the virtual environment (creates venv if it doesn't exist)
  > pipenv shell

  # Install dev dependencies (e.g., testing tools, linters)
  > pipenv install --dev pytest

  # Install all dependencies from Pipfile.lock (for reproducible deployments)
  > pipenv sync

  # Update all packages to their latest compatible versions
  > pipenv update

  # Run a command within the virtual environment without activating shell
  > pipenv run python script.py
  ```

- **Best for**: Simple project workflows requiring both dependency and
  environment management

### Poetry
- **What it does**: Comprehensive tool for dependency management, packaging, and
  publishing Python projects using modern standards
- Modern dependency management with sophisticated resolution
- Uses `pyproject.toml` (PEP 518 standard)
- Built-in dependency resolution and lockfile generation
- Handles packaging and publishing to PyPI
  ```bash
  # Initialize a new Poetry project interactively (creates pyproject.toml)
  > poetry init

  # Add a package as a dependency (updates pyproject.toml and poetry.lock)
  > poetry add requests

  # Activate a shell within the virtual environment
  > poetry shell

  # Add a dev dependency (e.g., testing or linting tools)
  > poetry add --group dev pytest

  # Install all dependencies from poetry.lock (for setting up project on new machine)
  > poetry install

  # Build distribution packages (wheel and sdist) for publishing
  > poetry build

  # Publish package to PyPI
  > poetry publish

  # Update dependencies to their latest compatible versions
  > poetry update

  # Run a command within the virtual environment without activating shell
  > poetry run python script.py
  ```

- **Best for**: Modern Python applications and libraries requiring robust
  dependency management and publishing

### Conda
- **What it does**: Cross-language package and environment manager that handles
  both Python packages and system dependencies (C libraries, compilers, etc.)
- Cross-language package + environment manager
- Popular in data science and scientific computing communities
- Manages Python versions and binary dependencies
- Installs pre-compiled binaries for faster installation
  ```bash
  # Create a new environment with a specific Python version
  > conda create -n myenv python=3.11

  # Activate an environment (makes it the active Python)
  > conda activate myenv

  # Install a package from conda channels (pre-compiled binaries)
  > conda install numpy

  # Install multiple packages at once (recommended for better dependency resolution)
  > conda install numpy pandas matplotlib

  # Export environment to a file (for sharing/reproducing)
  > conda env export > environment.yml

  # Create environment from a YAML file
  > conda env create -f environment.yml

  # Deactivate current environment (returns to base)
  > conda deactivate

  # List all environments
  > conda env list

  # Remove an environment
  > conda env remove -n myenv
  ```

- **Best for**: Data science and scientific computing projects requiring complex
  binary dependencies

### Mamba
- **What it does**: Drop-in replacement for Conda with a faster C++ dependency
  solver, providing the same functionality with significantly better performance
- Faster alternative to conda
- Compatible with conda commands and packages
- Faster dependency solver (written in C++)
- Particularly beneficial for large, complex environments
  ```bash
  # Create a new environment (same as conda but much faster)
  > mamba create -n myenv python=3.11

  # Activate environment (use conda activate, not mamba activate)
  > conda activate myenv

  # Install packages with faster dependency resolution than conda
  > mamba install numpy pandas scikit-learn

  # Update packages (significantly faster than conda update)
  > mamba update --all

  # Search for available packages
  > mamba search tensorflow

  # Install from a YAML file (faster than conda)
  > mamba env create -f environment.yml
  ```

- **Best for**: Large scientific environments where conda's solver is too slow

### `uv`
- **What it does**: Ultra-fast Rust-based package installer and resolver that
  serves as a drop-in replacement for pip with 10-100x faster performance
- Extremely fast Rust-based package manager
- Drop-in `pip` replacement with compatible CLI
- Can manage virtual environments
- Supports lockfiles for reproducible installs
  ```bash
  # Create a virtual environment (10-100x faster than venv)
  > uv venv

  # Install packages (significantly faster than pip)
  > uv pip install requests

  # Install from requirements.txt (with parallel downloads)
  > uv pip install -r requirements.txt

  # Compile dependencies to a lockfile (for reproducible installs)
  > uv pip compile requirements.in -o requirements.txt

  # Sync environment to exactly match lockfile (removes unlisted packages)
  > uv pip sync requirements.txt

  # Install a specific version with fast resolution
  > uv pip install "django>=4.0,<5.0"
  ```

- **Best for**: Fast modern workflows where performance matters, especially for
  CI/CD pipelines

### `pipx`
- **What it does**: Installs and runs Python CLI applications in isolated
  environments, making them globally available without dependency conflicts
- Installs Python CLI tools globally in isolated environments
- Each tool gets its own virtual environment
- Tools are globally accessible as commands
- Prevents dependency conflicts between CLI tools
  ```bash
  # Install a CLI tool globally in its own isolated environment
  > pipx install black

  # Install another tool (gets separate venv, no dependency conflicts)
  > pipx install pytest

  # Run a tool temporarily without installing it (downloads, runs, then cleans up)
  > pipx run cowsay "Hello!"

  # Upgrade an installed tool to the latest version
  > pipx upgrade black

  # List all installed tools and their versions
  > pipx list

  # Uninstall a tool and its isolated environment
  > pipx uninstall black

  # Upgrade all installed tools at once
  > pipx upgrade-all
  ```

- **Best for**: Installing and managing CLI tools like black, pytest, or
  cookiecutter globally

## Comparison Overview
| Tool       | Manages Packages | Manages Venv | Lockfile | Python Version Mgmt | Best For                      |
| :--------- | :--------------- | :----------- | :------- | :------------------ | :---------------------------- |
| pip        | Yes              | No           | No       | No                  | Basic installs                |
| venv       | No               | Yes          | No       | No                  | Environment isolation only    |
| virtualenv | No               | Yes          | No       | No                  | Legacy / advanced isolation   |
| pipenv     | Yes              | Yes          | Yes      | No                  | Simple project workflows      |
| poetry     | Yes              | Yes          | Yes      | No                  | Modern apps and libraries     |
| conda      | Yes              | Yes          | No       | Yes                 | Data science                  |
| mamba      | Yes              | Yes          | No       | Yes                 | Large scientific environments |
| uv         | Yes              | Yes          | Yes      | No                  | Fast modern workflows         |
| pipx       | Yes              | Auto         | No       | No                  | Installing CLI tools          |
