"""
Check the component layout of `research/Noesis`: allowed imports between
components, and a `README.md` in every component dir.

Import as:

import research.Noesis.test.test_component_imports as rnttcoim
"""

import ast
import logging
import os
from typing import Dict, List, Tuple

import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)

_NOESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PREFIX = "research.Noesis."

# Dirs that hold code of one component each.
_COMPONENTS = [
    "checker",
    "common",
    "gateway",
    "market",
    "monitor",
    "noesis_platform",
    "prober",
]

# Dirs that must have a `README.md`.
_README_DIRS = _COMPONENTS + ["devops", "docs", "scripts"]

# Allowed imports: component -> `research.Noesis.<dir>[.<module>]` prefixes that
# its code (outside `test/`) may import, besides its own component.
_ALLOWED: Dict[str, List[str]] = {
    "common": [],
    "market": [],
    "gateway": ["common"],
    "checker": ["common", "gateway.gateway_providers"],
    "monitor": [
        "checker",
        "common",
        "gateway.gateway_openrouter",
        "gateway.gateway_providers",
    ],
    "noesis_platform": ["market", "gateway.passthrough_proxy"],
    "prober": ["checker", "common", "gateway"],
}


def _get_code_files(component: str) -> List[str]:
    """
    Return the `.py` files of `component`, excluding its `test/` dir.
    """
    files = []
    component_dir = os.path.join(_NOESIS_DIR, component)
    for dir_path, dir_names, file_names in os.walk(component_dir):
        dir_names[:] = [d for d in dir_names if d != "test"]
        for file_name in file_names:
            if file_name.endswith(".py"):
                files.append(os.path.join(dir_path, file_name))
    return sorted(files)


def _get_imported_modules(path: str) -> List[str]:
    """
    Return the `research.Noesis.*` modules imported anywhere in `path`.
    """
    with open(path, "r") as f:
        tree = ast.parse(f.read())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        else:
            continue
        modules.extend(n for n in names if n.startswith(_PREFIX))
    return modules


def _is_allowed(component: str, module: str) -> bool:
    """
    Return whether code of `component` may import `module`.
    """
    rel_module = module[len(_PREFIX) :]
    allowed = [component] + _ALLOWED[component]
    for prefix in allowed:
        if rel_module == prefix or rel_module.startswith(prefix + "."):
            return True
    return False


# #############################################################################
# Test_component_imports
# #############################################################################


class Test_component_imports(hunitest.TestCase):
    def test_allowed_imports1(self) -> None:
        """
        Test that no component imports a component it is not allowed to.
        """
        # Prepare inputs.
        violations: List[Tuple[str, str]] = []
        # Run test.
        for component in _COMPONENTS:
            for path in _get_code_files(component):
                for module in _get_imported_modules(path):
                    if not _is_allowed(component, module):
                        rel_path = os.path.relpath(path, _NOESIS_DIR)
                        violations.append((rel_path, module))
        # Check outputs.
        self.assertEqual(violations, [])

    def test_components_have_code1(self) -> None:
        """
        Test that every component dir holds at least one module.
        """
        for component in _COMPONENTS:
            self.assertTrue(
                _get_code_files(component), msg=f"No code in '{component}'"
            )


# #############################################################################
# Test_readmes
# #############################################################################


class Test_readmes(hunitest.TestCase):
    def test_readme_exists1(self) -> None:
        """
        Test that every component dir has a `README.md`.
        """
        # Prepare inputs.
        missing = []
        # Run test.
        for dir_name in _README_DIRS:
            if not os.path.isfile(
                os.path.join(_NOESIS_DIR, dir_name, "README.md")
            ):
                missing.append(dir_name)
        # Check outputs.
        self.assertEqual(missing, [])
