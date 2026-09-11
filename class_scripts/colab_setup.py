"""
Make a `msml610`/`data605` tutorial notebook's repo-local imports work when
opened on Google Colab or Binder, instead of only inside the repo's Docker
container.

Neither Colab nor Binder hosts the whole repo, only the single notebook
file, so `helpers` (the `helpers_root` git submodule, source, not
pip-installed) and the notebook's own paired `_utils.py` file are missing
until fetched. Docker gets both for free (`helpers_root` on `PYTHONPATH`,
cwd = the notebook's own dir); `setup()` reproduces that.

This module is intentionally stdlib-only (no `helpers.*` imports): its job
is to make `helpers` importable in the first place, so it cannot depend on
it.

Import as:

import class_scripts.colab_setup as cscolset

Usage (as the first code cell of a tutorial notebook, before any
`helpers`/`*_utils` import, and before `%load_ext autoreload`):

    import os
    import sys

    ON_COLAB = "google.colab" in sys.modules
    ON_BINDER = "BINDER_LAUNCH_HOST" in os.environ

    if ON_COLAB or ON_BINDER:
        import subprocess

        # This module lives inside the repo, so on Colab it isn't
        # importable until the repo is cloned; Binder and local runs
        # already have it, just not always on `sys.path`.
        if ON_COLAB and not os.path.exists("gpsaggese.github.io"):
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "gp",
                    "https://github.com/gpsaggese/gpsaggese.github.io.git",
                ],
                check=True,
            )
        repo_root = (
            os.path.abspath("gpsaggese.github.io")
            if ON_COLAB
            else subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        # Docker gets this for free via PYTHONPATH; Colab/Binder need it.
        sys.path.insert(0, repo_root)

    import class_scripts.colab_setup as colab_setup

    colab_setup.setup("msml610/tutorials/L03_knowledge_representation")
    colab_setup.maybe_enable_autoreload()
"""

import logging
import os
import subprocess
import sys
from typing import List

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# `helpers_root`'s pinned URL in `.gitmodules` is SSH (`git@github.com:...`),
# which needs a key neither Colab nor Binder has: fetch over HTTPS instead.
_SUBMODULE_HTTPS_URL = "https://github.com/causify-ai/helpers.git"


# #############################################################################
# Helper functions
# #############################################################################


def is_colab() -> bool:
    """
    Return whether the code is running on Google Colab.
    """
    return "google.colab" in sys.modules


def is_binder() -> bool:
    """
    Return whether the code is running on Binder.
    """
    return "BINDER_LAUNCH_HOST" in os.environ


def _repo_root() -> str:
    """
    Return the repo's top-level dir.

    Derived from this module's own path (`class_scripts/colab_setup.py`,
    always one level below the root) rather than passed in, since by the
    time this module is importable the repo is already on disk in a known
    layout.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd: List[str]) -> None:
    """
    Run a command, streaming output, raising on a non-zero exit code.

    :param cmd: command and args, e.g. `["git", "pull"]`
    """
    _LOG.info("> %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _checkout_submodule(repo_root: str) -> None:
    """
    Fetch the `helpers_root` submodule's content into an existing clone.

    A plain `git clone`/`git pull` of the superproject leaves a submodule
    on disk but empty: it needs its own fetch.

    :param repo_root: git superproject's working tree
    """
    _run(
        [
            "git",
            "-c",
            f"submodule.helpers_root.url={_SUBMODULE_HTTPS_URL}",
            "-C",
            repo_root,
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
        ]
    )


# #############################################################################
# Setup
# #############################################################################


def setup(nb_dir: str) -> None:
    """
    Make a tutorial notebook's repo-local imports work on Colab or Binder.

    Fetches the `helpers_root` submodule, wires `sys.path` for `helpers`
    and the notebook's own paired `_utils.py` file, `chdir`s into the
    notebook's dir (so relative file references in the notebook keep
    working), and installs the tutorial's `requirements.txt`.

    :param nb_dir: notebook's dir relative to the repo root, e.g.
        "msml610/tutorials/L03_knowledge_representation"
    """
    repo_root = _repo_root()
    if is_colab():
        # Colab may have cloned this in an earlier run of the same
        # session: pull instead of skipping, so a rerun picks up the
        # latest pushed code. Autoreload is off on Colab (see
        # `maybe_enable_autoreload()`), and modules already imported this
        # session stay cached, so a code update still needs a runtime
        # restart to take effect.
        _run(["git", "-C", repo_root, "pull"])
    _checkout_submodule(repo_root)
    sys.path.insert(0, os.path.join(repo_root, "helpers_root"))
    sys.path.insert(0, os.path.join(repo_root, nb_dir))
    os.chdir(os.path.join(repo_root, nb_dir))
    if is_colab():
        # Binder already built its image from the tutorial's
        # requirements.txt; Colab starts with its own generic env.
        _run(["pip", "install", "-q", "-r", "requirements.txt"])


def maybe_enable_autoreload() -> None:
    """
    Enable IPython's `autoreload` extension, unless it would be a no-op or
    crash.

    Skipped on Colab/Binder: `setup()` must run first to give it repo files
    to watch, and there's nothing to reload before that (call this after
    `setup()`, not before). Also skipped outside a notebook kernel, where
    `get_ipython()` is `None`, e.g. when the paired `.py` runs as a plain
    script.
    """
    if is_colab() or is_binder():
        return
    from IPython.core.getipython import get_ipython

    ip = get_ipython()
    if ip is None:
        return
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
