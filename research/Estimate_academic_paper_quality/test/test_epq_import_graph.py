import ast
import logging
import os
from typing import Dict, List, Set

import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)

_PREFIX = "research.Estimate_academic_paper_quality."

# Allowed imports of the modules of the project, from the architecture doc
# `docs/02-design.architecture.md`, section "Import Graph". Each module points
# down a layer, never up.
_ALLOWED: Dict[str, Set[str]] = {
    # The executables import only modules, never another executable: they talk
    # to each other through files.
    "resolve_papers": {
        "epq_bib_sources",
        "epq_fetcher",
        "epq_issues",
        "epq_paths",
        "epq_report",
        "epq_summary",
        "epq_types",
    },
    "download_papers": {
        "epq_fetcher",
        "epq_issues",
        "epq_paths",
        "epq_report",
        "epq_summary",
        "epq_types",
    },
    "extract_features": {
        "epq_fetcher",
        "epq_issues",
        "epq_paths",
        "epq_report",
        "epq_summary",
        "epq_types",
    },
    "report_papers": {
        "epq_bib_sources",
        "epq_issues",
        "epq_paths",
        "epq_report",
        "epq_summary",
        "epq_types",
    },
    # The pipeline script runs the executables as subprocesses, so it imports
    # none of them.
    "estimate_paper_quality": {"epq_paths"},
    # The helper script fills the tier list by hand, outside the pipeline, and
    # imports no module of the project.
    "populate_affiliation_tiers": set(),
    "epq_report": {"epq_types"},
    "epq_summary": {"epq_paths", "epq_types"},
    "epq_bib_sources": {"epq_types"},
    "epq_fetcher": {"epq_types"},
    "epq_issues": {"epq_types"},
    "epq_paths": set(),
    "epq_types": set(),
}


def _get_project_imports(path: str) -> Set[str]:
    """
    List the modules of the project that a file imports.

    :param path: path of a Python file
    :return: names of the modules, e.g., `epq_types`
    """
    tree = ast.parse(hio.from_file(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name.startswith(_PREFIX):
                imported.add(name[len(_PREFIX) :].split(".")[0])
    return imported


def _find_violations(project_dir: str) -> List[str]:
    """
    Find the imports that the architecture does not allow.

    :param project_dir: dir of the modules of the project
    :return: one text for each import that is not allowed
    """
    violations = []
    for module, allowed in _ALLOWED.items():
        path = os.path.join(project_dir, f"{module}.py")
        if not os.path.isfile(path):
            continue
        for name in sorted(_get_project_imports(path) - allowed):
            violations.append(f"{module} imports {name}")
    return violations


# #############################################################################
# Test_import_graph
# #############################################################################


class Test_import_graph(hunitest.TestCase):
    """
    Test that the modules of the pipeline import only what the architecture
    allows.
    """

    def test1(self) -> None:
        """
        Test the modules of the project follow the import graph.
        """
        # Prepare inputs.
        project_dir = os.path.dirname(hgit.find_file_in_git_tree("epq_types.py"))
        # Run test.
        actual = _find_violations(project_dir)
        # Check outputs.
        self.assertEqual(actual, [])

    def test2(self) -> None:
        """
        Test the check fails on a module that imports a layer above it.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        hio.to_file(
            os.path.join(scratch_dir, "epq_types.py"),
            f"import {_PREFIX}report_papers as repopapers\n",
        )
        # Run test.
        actual = _find_violations(scratch_dir)
        # Check outputs.
        self.assertEqual(actual, ["epq_types imports report_papers"])

    def test3(self) -> None:
        """
        Test every module of the project is in the table of allowed imports.
        """
        # Prepare inputs.
        project_dir = os.path.dirname(hgit.find_file_in_git_tree("epq_types.py"))
        # Run test.
        modules = {
            name[: -len(".py")]
            for name in os.listdir(project_dir)
            if name.endswith(".py")
        }
        # Check outputs.
        self.assertEqual(modules, set(_ALLOWED))

    def test4(self) -> None:
        """
        Test the check fails on an executable that imports another executable.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        hio.to_file(
            os.path.join(scratch_dir, "download_papers.py"),
            f"import {_PREFIX}extract_features as exfeat\n",
        )
        # Run test.
        actual = _find_violations(scratch_dir)
        # Check outputs.
        self.assertEqual(actual, ["download_papers imports extract_features"])
