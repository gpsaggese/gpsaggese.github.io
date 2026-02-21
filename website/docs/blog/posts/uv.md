---
title: "UV in 30 mins"
authors:
  - gpsaggese
date: 2026-02-14
description:
categories:
  - Python
---

TL;DR `uv` is a blazingly fast Python package manager written in Rust that
replaces `pip`, `pip-tools`, and `virtualenv` with a single tool that is 10-100x
faster.

<!-- more -->

## Introduction

- This tutorial explores why `uv` is becoming the go-to package manager for
  Python developers and how to use it effectively

- Python's packaging ecosystem challenges:
  - Long criticized for complexity
  - Slow performance

- `uv` addresses these pain points:
  - Unified solution
  - Fast performance
  - Modern dependency management
  - Efficient virtual environment handling

## Why UV?

- **Speed**: Written in Rust with highly optimized algorithms, `uv` is 10-100x
  faster than `pip` for dependency resolution and installation
- **All-in-one tool**: Replaces multiple tools (`pip`, `pip-tools`,
  `virtualenv`, `pyenv`) with a single unified interface
- **Drop-in replacement**: Compatible with `pip` and `requirements.txt` format,
  making migration seamless
- **Better dependency resolution**: Uses a modern SAT solver for more reliable
  dependency resolution
- **Cross-platform**: Works seamlessly on Linux, macOS, and Windows
- **Disk space efficient**: Smart caching reduces redundant downloads and
  storage
- **Project management**: Built-in support for creating and managing Python
  projects

Here's a speed comparison for installing packages:

| Tool               | Time (seconds) | Notes                       |
| :----------------- | :------------- | :-------------------------- |
| `pip install`      | 45.2           | Standard installation       |
| `poetry install`   | 38.7           | With lock file              |
| `uv pip install`   | 2.1            | First install with caching  |
| `uv pip install`   | 0.3            | Subsequent install (cached) |

## Installation

- On macOS and Linux using the official installer:

  ```bash
  > curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- Using Homebrew on macOS:

  ```bash
  > brew install uv
  ```

- Using `pip` (ironically):

  ```bash
  > pip install uv
  ```

### Verifying installation

- Verify the installation:
  ```bash
  > uv --version
  uv 0.9.5 (Homebrew 2025-10-21
  ```

## Basic Usage

### Creating a Virtual Environment

- Create a virtual environment:
  ```bash
  > uv venv
  Using CPython 3.14.2 interpreter at: /opt/homebrew/opt/python@3.14/bin/python3.14
  Creating virtual environment at: .venv
  Activate with: source .venv/bin/activate
  ```
  - This creates a `.venv` directory with a fresh Python environment
  ```bash
  > ls .venv/
  bin          CACHEDIR.TAG lib          pyvenv.cfg
  ```

- Activate the environment:
  ```bash
  > source .venv/bin/activate
  > which python
  /Users/USER/.venv/bin/python
  ```

- Create a virtual environment with a specific Python version:
  ```bash
  > uv venv --python 3.9 --clear; source .venv/bin/activate; python --version
  Using CPython 3.9.6 interpreter at: /Library/Developer/CommandLineTools/usr/bin/python3
  Creating virtual environment at: .venv
  Activate with: source .venv/bin/activate
  Python 3.9.6
  ```

### Installing Packages

- Install a package using `uv pip`:
  ```bash
  > uv venv --python 3.14 --clear; source .venv/bin/activate
  Using CPython 3.14.2 interpreter at: /opt/homebrew/opt/python@3.14/bin/python3.14
  Creating virtual environment at: .venv
  Activate with: source .venv/bin/activate

  > which python; python --version
  Python 3.14.2

  > python -c "import requests; print(requests.__version__)"
  Traceback (most recent call last):
    File "<string>", line 1, in <module>
      import requests; print(requests.__version__)
      ^^^^^^^^^^^^^^^
  ModuleNotFoundError: No module named 'requests'

  > uv pip install requests
  Resolved 5 packages in 86ms
  Installed 5 packages in 14ms
   + certifi==2026.1.4
   + charset-normalizer==3.4.4
   + idna==3.11
   + requests==2.32.5
   + urllib3==2.6.3

  > python -c "import requests"

  > python -c "import requests; print(requests.__version__)"
  2.32.5
  ```

- Install multiple packages:
  ```bash
  > uv pip install requests pandas numpy
  ```

- Install from a requirements file:
  ```bash
  > more requirements.txt
  requests
  pandas
  numpy

  > uv pip install -r requirements.txt
  ```

- Install with specific version constraints:
  ```bash
  > uv pip install "django>=4.0,<5.0"
  ```

- Install from a requirements file with pinned versions:
  ```bash
  > more requirements.txt
  requests==2.31.0
  pandas==2.2.0
  numpy==1.26.4

  > uv pip install -r requirements.txt
  ```

## Project Management

### Initializing a New Project

- Create a new project with scaffolding:

  ```bash
  > uv init myproject
  > cd myproject
  ```

  - This creates a new project with:
    - `pyproject.toml`: Project configuration and dependencies
    - `README.md`: Project documentation
    - `src/myproject/`: Source code directory
    - `.python-version`: Python version specification

### Dev vs Prod Dependencies

- Production Dependencies = required to run your application in production
  - In `pyproject.toml`:
    ```
    [project]
    dependencies = [
      "fastapi"
    ]
    ```

- Development Dependencies = needed only for development (testing, linting,
  formatting)
  - Not required in production
  - Example: `pytest`
  - In `pyproject.toml`:
    ```
    [project.optional-dependencies]
    dev = [
      "pytest"
    ]
    ```

### Working with pyproject.toml

- Example `pyproject.toml` configuration:

  ```toml
  [project]
  name = "myproject"
  version = "0.1.0"
  requires-python = ">=3.11"
  dependencies = [
      "requests>=2.31.0",
      "pandas>=2.0.0",
  ]

  [project.optional-dependencies]
  dev = [
      "pytest>=7.0.0",
      "black>=23.0.0",
  ]
  ```

- Add a dependency to your project:

  ```bash
  > uv add requests
  ```

- Add a development dependency:
  ```bash
  > uv add --dev pytest
  ```

- Remove a dependency:
  ```bash
  > uv remove requests
  ```

### Syncing Dependencies

- Sync your environment with project dependencies:

  ```bash
  > uv sync
  ```

  - This installs all dependencies specified in `pyproject.toml` and creates or
    updates the lock file

## Dependency Locking

### Creating Lock Files

- Generate a lock file for reproducible builds:

  ```bash
  > uv lock
  ```

  - This creates `uv.lock` with exact versions of all dependencies and their
    transitive dependencies

- Install from lock file:

  ```bash
  > uv sync --frozen
  ```

  - The `--frozen` flag ensures no changes are made to the lock file

- Update dependencies to latest compatible versions:

  ```bash
  > uv lock --upgrade
  ```

- Update a specific package:

  ```bash
  > uv lock --upgrade-package requests
  ```

## Advanced Features

### Python Version Management

- List available Python versions:

  ```bash
  > uv python list
  ```

- Install a specific Python version:

  ```bash
  > uv python install 3.11
  ```

- Use a specific Python version for a project:

  ```bash
  > uv python pin 3.11
  ```

  - This creates a `.python-version` file in your project

### Running Commands

- Run Python scripts without activating the environment:

  ```bash
  > uv run python script.py
  ```

- Run a command with dependencies:

  ```bash
  > uv run --with requests python -c "import requests; print(requests.__version__)"
  ```

  - This temporarily installs `requests` and runs the command

### Tool Management

- Install and run Python tools globally:

  ```bash
  > uv tool install black
  ```

- Run a tool without installing:

  ```bash
  > uvx black .
  ```

  - The `uvx` command (or `uv tool run`) is similar to `npx` for Node.js

- List installed tools:

  ```bash
  > uv tool list
  ```

### Caching

- Show cache information:

  ```bash
  > uv cache dir
  ```

- Clean the cache:

  ```bash
  > uv cache clean
  ```

- Clean cache for a specific package:

  ```bash
  > uv cache clean requests
  ```

## Practical Examples

### Migrating from pip

- If you have an existing project with `requirements.txt`:

  ```bash
  > uv venv
  > source .venv/bin/activate
  > uv pip install -r requirements.txt
  ```

- Generate a lock file from requirements:

  ```bash
  > uv pip compile requirements.in -o requirements.txt
  ```

### Creating a requirements.txt

- Freeze current environment to requirements:

  ```bash
  > uv pip freeze > requirements.txt
  ```

### Working with Multiple Environments

- Create separate environments for different purposes:

  ```bash
  > uv venv .venv-dev --python 3.11
  > uv venv .venv-test --python 3.10
  ```

### Building Distributions

- Build a package for distribution:

  ```bash
  > uv build
  ```

  - This creates wheel and source distributions in the `dist/` directory

### Installing from Local Path

- Install a package in development mode:

  ```bash
  > uv pip install -e .
  ```

- Install from a local wheel:

  ```bash
  > uv pip install dist/mypackage-0.1.0-py3-none-any.whl
  ```

## Configuration

### Project Configuration

- Configure `uv` behavior in `pyproject.toml`:

  ```toml
  [tool.uv]
  index-url = "https://pypi.org/simple"
  extra-index-url = ["https://my-private-index.example.com/simple"]

  [tool.uv.pip]
  no-binary = ["numpy"]
  ```

### Global Configuration

- Create a global config at `~/.config/uv/uv.toml`:

  ```toml
  [pip]
  index-url = "https://pypi.org/simple"

  [cache]
  dir = "~/.cache/uv"
  ```

### Environment Variables

- Control `uv` behavior with environment variables:

  ```bash
  > export UV_INDEX_URL="https://pypi.org/simple"
  > export UV_CACHE_DIR="~/.cache/uv"
  > export UV_NO_CACHE=1
  ```

## Tips and Tricks

### Offline Installation

- Cache packages for offline use:

  ```bash
  > uv pip download -r requirements.txt -d packages/
  ```

- Install from cached packages:

  ```bash
  > uv pip install --no-index --find-links packages/ -r requirements.txt
  ```

### Upgrading All Packages

- Upgrade all dependencies to latest compatible versions:

  ```bash
  > uv lock --upgrade
  > uv sync
  ```

### Resolving Conflicts

- `uv` provides detailed error messages for dependency conflicts:

  ```bash
  > uv pip install package1 package2
  ```

  - Shows exactly which constraints conflict
  - Suggests resolutions

### Pre-release Versions

- Install pre-release versions:

  ```bash
  > uv pip install --prerelease=allow package_name
  ```

### Platform-Specific Dependencies

- Specify platform-specific dependencies in `pyproject.toml`:

  ```toml
  [project]
  dependencies = [
      "requests",
  ]

  [project.optional-dependencies]
  linux = ["psutil"]
  windows = ["pywin32"]
  ```

## Integration with Development Tools

### Docker Integration

- Example Dockerfile using `uv`:

  ```dockerfile
  FROM python:3.11-slim

  # Install uv
  COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

  # Copy project files
  COPY pyproject.toml uv.lock ./

  # Install dependencies
  RUN uv sync --frozen --no-dev

  # Copy application code
  COPY . .

  CMD ["uv", "run", "python", "main.py"]
  ```

### CI/CD Integration

- Example GitHub Actions workflow:

  ```yaml
  - name: Install uv
    uses: astral-sh/setup-uv@v1

  - name: Set up Python
    run: uv python install 3.11

  - name: Install dependencies
    run: uv sync --frozen

  - name: Run tests
    run: uv run pytest
  ```

### Pre-commit Hooks

- Example pre-commit configuration:

  ```yaml
  repos:
    - repo: local
      hooks:
        - id: uv-lock
          name: uv lock
          entry: uv lock
          language: system
          pass_filenames: false
  ```

## Common Gotchas

### Python Version Not Found

- If `uv` cannot find a Python version:

  ```bash
  > uv python install 3.11
  ```

  - This installs the required Python version

### Lock File Out of Sync

- If dependencies change, regenerate the lock file:

  ```bash
  > uv lock
  ```

### Cache Issues

- If you encounter unexpected behavior, clear the cache:

  ```bash
  > uv cache clean
  ```

### Private Package Indexes

- Configure authentication for private PyPI mirrors:

  ```bash
  > export UV_INDEX_URL="https://user:token@private-index.example.com/simple"
  ```

  - Alternative: use keyring integration for secure credential storage

## Comparison with Other Tools

`uv` vs `pip`:

| Feature              | uv       | pip     |
| :------------------- | :------- | :------ |
| Speed                | +++      | +       |
| Dependency resolver  | Modern   | Legacy  |
| Lock files           | Built-in | Requires pip-tools |
| Virtual environments | Built-in | Requires virtualenv |
| Project management   | Yes      | No      |

`uv` vs `poetry`:

| Feature           | uv      | poetry  |
| :---------------- | :------ | :------ |
| Speed             | +++     | ++      |
| Installation time | Seconds | Minutes |
| Learning curve    | Low     | Medium  |
| Plugin system     | Limited | Rich    |

`uv` vs `pip-tools`:

| Feature        | uv       | pip-tools |
| :------------- | :------- | :-------- |
| Speed          | +++      | +         |
| Ease of use    | Higher   | Medium    |
| Compatibility  | Excellent | Excellent |
| Maintenance    | Active   | Active    |
