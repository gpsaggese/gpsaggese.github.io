# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    capture_output=True,
    text=True,
    check=False
)
print(result.stdout or result.stderr)
if result.returncode != 0:
    raise RuntimeError("Dependency conflicts found. Review the output above.")

# %%
import platform
from importlib.metadata import version
from pathlib import Path

packages = [
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "jupyterlab",
    "jupytext",
]

lines = [
    f"Python: {platform.python_version()}",
    "",
    *[f"{package}=={version(package)}" for package in packages],
]

Path("environment_versions.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# %%
