# Python Package & Virtual Environment Management Tools
This document provides an overview of the main tools used for managing packages
and virtual environments in Python, grouped by purpose and modern usage.


## 🔹 Built-In & Standard Tools

### 1️⃣ Pip
- Default Python package installer
- Installs packages from PyPI
- Works with `requirements.txt`
```bash
pip install requests
pip install -r requirements.txt
pip freeze > requirements.txt
```

**Best for:** Simple projects and universal compatibility.


### 2️⃣ Venv
- Built-in virtual environment tool (Python 3.3+)
- Creates isolated environments
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

**Best for:** Standard lightweight environment isolation.


### 3️⃣ Virtualenv
- Older alternative to `venv`
- More features
- Slightly faster environment creation
```bash
pip install virtualenv
virtualenv myenv
```

**Best for:** Legacy systems or advanced isolation needs.


## 🔹 Modern Dependency & Environment Managers

### 4️⃣ Pipenv
- Combines `pip` + `venv`
- Uses `Pipfile` and `Pipfile.lock`
- Automatic virtual environment management
```bash
pipenv install requests
pipenv shell
```

**Best for:** Simple dependency + environment workflows.


### 5️⃣ Poetry
- Modern dependency management
- Uses `pyproject.toml`
- Built-in dependency resolution
- Handles packaging & publishing
```bash
poetry init
poetry add requests
poetry shell
```

**Best for:** Modern applications, libraries, and production projects.


### 6️⃣ Conda
- Cross-language package + environment manager
- Popular in data science
- Manages Python versions
```bash
conda create -n myenv python=3.11
conda activate myenv
conda install numpy
```

**Best for:** Data science and scientific computing.


### 7️⃣ Mamba
- Faster alternative to conda
- Same commands
- Faster dependency solver
```bash
mamba create -n myenv python=3.11
```

**Best for:** Large scientific environments.


## 🔹 Lightweight / Emerging Tools

### 8️⃣ Uv
- Extremely fast Rust-based package manager
- Drop-in `pip` replacement
- Can manage virtual environments
```bash
uv venv
uv pip install requests
```

**Best for:** Fast, modern workflows.


### 9️⃣ Pipx
- Installs Python CLI tools globally in isolated environments
```bash
pipx install black
```

**Best for:** Installing developer CLI tools safely.


## 🔹 Comparison Overview
Tool Manages Manages Lockfile Python Version Best For Packages Venv Mgmt

| Tool       | Manages Packages | Manages Venv | Lockfile | Python Version Mgmt | Best For |
|------------|------------------|--------------|----------|---------------------|----------|
| pip        | ✅ Yes           | ❌ No        | ❌ No    | ❌ No               | Basic installs |
| venv       | ❌ No            | ✅ Yes       | ❌ No    | ❌ No               | Environment isolation only |
| virtualenv | ❌ No            | ✅ Yes       | ❌ No    | ❌ No               | Legacy / advanced isolation |
| pipenv     | ✅ Yes           | ✅ Yes       | ✅ Yes   | ❌ No               | Simple project workflows |
| poetry     | ✅ Yes           | ✅ Yes       | ✅ Yes   | ❌ No               | Modern apps & libraries |
| conda      | ✅ Yes           | ✅ Yes       | ❌ No    | ✅ Yes              | Data science |
| mamba      | ✅ Yes           | ✅ Yes       | ❌ No    | ✅ Yes              | Large scientific environments |
| uv         | ✅ Yes           | ✅ Yes       | ✅ Yes   | ❌ No               | Fast modern workflows |
| pipx       | ✅ Yes           | ✅ Auto      | ❌ No    | ❌ No               | Installing CLI tools |
