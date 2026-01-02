"""Compatibility package for tutorial modules."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _load(submodule: str) -> None:
    module_path = _PACKAGE_ROOT / f"retail_sales_forecasting_with_lstms.{submodule}.py"
    if not module_path.exists():
        raise ImportError(f"Missing module file: {module_path}")
    qualified = f"{__name__}.{submodule}"
    loader = SourceFileLoader(qualified, str(module_path))
    spec = importlib.util.spec_from_loader(qualified, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[qualified] = module


for _sub in ("API", "example"):
    try:
        _load(_sub)
    except ImportError:
        pass
