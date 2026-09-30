from __future__ import annotations

import pathlib

import pytest


def test_load_excel_releases_file_handle(tmp_path: pathlib.Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    from src.ingest.file_loader import load_file

    path = tmp_path / "test.xlsx"
    df_in = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df_in.to_excel(path, index=False)

    df_out, meta = load_file(path)
    assert meta["source_format"] == "excel"
    assert list(df_out.columns) == ["a", "b"]

    # On Windows, this fails with PermissionError if the loader leaks file handles.
    path.unlink()

