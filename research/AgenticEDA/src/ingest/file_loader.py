from __future__ import annotations

"""
Multi-format file ingestion for the AgenticEDA pipeline.

Supported formats: CSV, TSV, Excel, JSON, NDJSON, Parquet, log/txt, MongoDB JSON exports.
"""

import csv
import io
import json
import pathlib
import re
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

_EMPTY_METADATA: dict[str, Any] = {
    "source_format": "",
    "original_filename": "",
    "sheet_name": None,
    "encoding": None,
    "row_count": 0,
    "col_count": 0,
    "load_warnings": [],
}


def _make_metadata(
    source_format: str,
    original_filename: str,
    df: pd.DataFrame,
    *,
    sheet_name: str | None = None,
    encoding: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a standardised metadata dict."""
    return {
        "source_format": source_format,
        "original_filename": original_filename,
        "sheet_name": sheet_name,
        "encoding": encoding,
        "row_count": len(df),
        "col_count": len(df.columns),
        "load_warnings": warnings or [],
    }


# ---------------------------------------------------------------------------
# Encoding detection
# ---------------------------------------------------------------------------

def _detect_encoding(path: pathlib.Path, sample_bytes: int = 100_000) -> tuple[str, list[str]]:
    """Return (encoding, warnings).  Falls back to utf-8."""
    warnings: list[str] = []
    try:
        import chardet
    except ImportError:
        warnings.append("chardet is not installed; falling back to utf-8 encoding detection.")
        return "utf-8", warnings

    raw = path.read_bytes()[:sample_bytes]
    if not raw:
        return "utf-8", warnings
    result = chardet.detect(raw)
    enc = result.get("encoding") or "utf-8"
    confidence = result.get("confidence", 0)
    if confidence < 0.5:
        warnings.append(
            f"Low encoding-detection confidence ({confidence:.0%}) for '{enc}'; falling back to utf-8."
        )
        enc = "utf-8"
    return enc, warnings


# ---------------------------------------------------------------------------
# Delimiter sniffing
# ---------------------------------------------------------------------------

def _sniff_delimiter(path: pathlib.Path, encoding: str, default: str = ",") -> str:
    """Use csv.Sniffer on the first 5 lines to guess the delimiter."""
    try:
        with open(path, encoding=encoding, errors="replace") as fh:
            sample = "".join(fh.readline() for _ in range(5))
        if not sample.strip():
            return default
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return default


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_csv(path: pathlib.Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    warnings: list[str] = []
    encoding, enc_warnings = _detect_encoding(path)
    warnings.extend(enc_warnings)

    delimiter = _sniff_delimiter(path, encoding)
    try:
        df = pd.read_csv(path, encoding=encoding, sep=delimiter)
    except Exception as exc:
        warnings.append(f"CSV parse error with encoding={encoding}, delimiter={delimiter!r}: {exc}")
        # Retry with utf-8 and comma as a last resort
        try:
            df = pd.read_csv(path, encoding="utf-8", sep=",")
            warnings.append("Retried with utf-8/comma and succeeded.")
        except Exception as exc2:
            raise ValueError(f"Unable to parse CSV file '{path.name}': {exc2}") from exc2

    meta = _make_metadata("csv", path.name, df, encoding=encoding, warnings=warnings)
    return df, meta


def _load_tsv(path: pathlib.Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    warnings: list[str] = []
    encoding, enc_warnings = _detect_encoding(path)
    warnings.extend(enc_warnings)

    try:
        df = pd.read_csv(path, encoding=encoding, sep="\t")
    except Exception as exc:
        warnings.append(f"TSV parse error: {exc}")
        raise ValueError(f"Unable to parse TSV file '{path.name}': {exc}") from exc

    meta = _make_metadata("tsv", path.name, df, encoding=encoding, warnings=warnings)
    return df, meta


def _load_excel(
    path: pathlib.Path, sheet_name: str | int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load an Excel workbook.  Raises a clear error when openpyxl is missing."""
    warnings: list[str] = []

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError(
            "openpyxl is required to read Excel files. Install it with: pip install openpyxl"
        )

    engine = "openpyxl" if path.suffix.lower() == ".xlsx" else None

    target_sheet = sheet_name if sheet_name is not None else 0

    # Important: keep the ExcelFile open only inside a context manager.
    # On Windows, leaving an ExcelFile handle unclosed prevents deleting the underlying file.
    try:
        with pd.ExcelFile(path, engine=engine) as xls:
            all_sheets = xls.sheet_names

            if len(all_sheets) > 1 and sheet_name is None:
                warnings.append(
                    f"Workbook contains {len(all_sheets)} sheets {all_sheets}; loading the first sheet ('{all_sheets[0]}')."
                )

            try:
                df = xls.parse(sheet_name=target_sheet)
            except Exception as exc:
                warnings.append(f"Excel parse error: {exc}")
                raise ValueError(f"Unable to parse Excel file '{path.name}': {exc}") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to open Excel file '{path.name}': {exc}") from exc

    resolved_sheet = (
        target_sheet
        if isinstance(target_sheet, str)
        else (
            all_sheets[target_sheet]
            if isinstance(target_sheet, int) and 0 <= target_sheet < len(all_sheets)
            else str(target_sheet)
        )
    )

    meta = _make_metadata("excel", path.name, df, sheet_name=resolved_sheet, warnings=warnings)
    return df, meta


def _has_nested(data: list[dict]) -> bool:
    """Return True if any top-level value in any record is a dict or list."""
    for record in data[:50]:  # sample first 50 for speed
        if isinstance(record, dict):
            for v in record.values():
                if isinstance(v, (dict, list)):
                    return True
    return False


def _load_json(path: pathlib.Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    warnings: list[str] = []

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    if not raw_text.strip():
        raise ValueError(f"JSON file '{path.name}' is empty.")

    # Try pandas read_json (records orientation) first
    try:
        df = pd.read_json(io.StringIO(raw_text), orient="records")
        # Verify it produced something sensible
        if df.empty or (len(df.columns) == 1 and df.columns[0] == 0):
            raise ValueError("Unhelpful result from read_json")
        meta = _make_metadata("json", path.name, df, warnings=warnings)
        return df, meta
    except Exception:
        pass

    # Fall back to raw JSON + json_normalize
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse JSON file '{path.name}': {exc}") from exc

    if isinstance(data, list):
        if _has_nested(data):
            df = pd.json_normalize(data, sep=".")
            warnings.append("Nested JSON detected; flattened with json_normalize (sep='.').")
        else:
            df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Single object or wrapper with a data key
        # Try to find the main list
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                if _has_nested(val):
                    df = pd.json_normalize(val, sep=".")
                    warnings.append(f"Used key '{key}' as record list; flattened nested structures.")
                else:
                    df = pd.DataFrame(val)
                    warnings.append(f"Used key '{key}' as record list.")
                break
        else:
            # Wrap single dict as one-row DataFrame
            df = pd.json_normalize(data, sep=".")
            warnings.append("Single JSON object loaded as one-row DataFrame.")
    else:
        raise ValueError(f"Unexpected top-level JSON type ({type(data).__name__}) in '{path.name}'.")

    meta = _make_metadata("json", path.name, df, warnings=warnings)
    return df, meta


def _load_ndjson(path: pathlib.Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    warnings: list[str] = []

    try:
        df = pd.read_json(path, lines=True)
        # Check for nesting
        has_dict_cols = any(df[c].dropna().apply(lambda x: isinstance(x, (dict, list))).any() for c in df.columns)
        if has_dict_cols:
            # Re-parse and normalize
            records = [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").strip().splitlines() if line.strip()]
            df = pd.json_normalize(records, sep=".")
            warnings.append("Nested NDJSON detected; flattened with json_normalize (sep='.').")
    except Exception as exc:
        # Try line-by-line
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            records = [json.loads(line) for line in lines if line.strip()]
            if not records:
                raise ValueError("No valid JSON lines found.")
            if _has_nested(records):
                df = pd.json_normalize(records, sep=".")
                warnings.append("Nested NDJSON detected; flattened with json_normalize (sep='.').")
            else:
                df = pd.DataFrame(records)
        except Exception as exc2:
            raise ValueError(f"Unable to parse NDJSON file '{path.name}': {exc2}") from exc2

    meta = _make_metadata("ndjson", path.name, df, warnings=warnings)
    return df, meta


def _load_parquet(path: pathlib.Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyarrow is required to read Parquet files. Install it with: pip install pyarrow"
        )

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"Unable to parse Parquet file '{path.name}': {exc}") from exc

    meta = _make_metadata("parquet", path.name, df, warnings=[])
    return df, meta


# ---------------------------------------------------------------------------
# Log file parsing
# ---------------------------------------------------------------------------

_LOG_PATTERNS: list[tuple[str, re.Pattern[str], list[str]]] = [
    (
        "apache_nginx",
        re.compile(r'(\S+) - - \[(.+?)\] "(.*?)" (\d+) (\d+)'),
        ["remote_host", "timestamp", "request", "status", "bytes"],
    ),
    (
        "syslog",
        re.compile(r"(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+(\S+?):\s+(.*)"),
        ["timestamp", "hostname", "process", "message"],
    ),
    (
        "generic_timestamp",
        re.compile(r"(\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}[^ ]*)\s+(.*)"),
        ["timestamp", "message"],
    ),
]


def _load_log(path: pathlib.Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    warnings: list[str] = []
    encoding, enc_warnings = _detect_encoding(path)
    warnings.extend(enc_warnings)

    try:
        text = path.read_text(encoding=encoding, errors="replace")
    except Exception as exc:
        raise ValueError(f"Unable to read log file '{path.name}': {exc}") from exc

    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Log file '{path.name}' is empty.")

    # Use first 20 lines for pattern detection, but parse all lines
    sample = lines[:20]
    non_empty_sample = [l for l in sample if l.strip()]
    if not non_empty_sample:
        raise ValueError(f"Log file '{path.name}' contains only blank lines.")

    # Try structured log patterns
    for pattern_name, regex, col_names in _LOG_PATTERNS:
        matches = [regex.match(line) for line in non_empty_sample]
        match_rate = sum(1 for m in matches if m) / len(non_empty_sample)
        if match_rate > 0.8:
            # Parse all lines with this pattern
            rows = []
            for line in lines:
                m = regex.match(line)
                if m:
                    rows.append(m.groups())
            df = pd.DataFrame(rows, columns=col_names)
            skipped = len(lines) - len(rows)
            if skipped:
                warnings.append(f"{skipped} line(s) did not match the '{pattern_name}' pattern and were skipped.")
            meta = _make_metadata("log", path.name, df, encoding=encoding, warnings=warnings)
            return df, meta

    # Try CSV-like detection
    try:
        sample_text = "\n".join(sample)
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
        if csv.Sniffer().has_header(sample_text):
            df = pd.read_csv(path, encoding=encoding, sep=dialect.delimiter)
            warnings.append("Log file detected as CSV-like with a header row.")
            meta = _make_metadata("log", path.name, df, encoding=encoding, warnings=warnings)
            return df, meta
    except csv.Error:
        pass

    # Fallback: single-column raw_line
    df = pd.DataFrame({"raw_line": lines})
    warnings.append(
        "No known log pattern matched; loaded as single-column 'raw_line' DataFrame."
    )
    meta = _make_metadata("log", path.name, df, encoding=encoding, warnings=warnings)
    return df, meta


# ---------------------------------------------------------------------------
# MongoDB JSON export helpers
# ---------------------------------------------------------------------------

def _unwrap_mongo(obj: Any) -> Any:
    """Recursively unwrap MongoDB extended-JSON wrappers to native Python types."""
    if isinstance(obj, dict):
        # Single-key MongoDB wrappers
        if len(obj) == 1:
            key = next(iter(obj))
            val = obj[key]
            if key == "$date":
                # Could be ISO string or {"$numberLong": "..."}
                inner = _unwrap_mongo(val)
                try:
                    return pd.Timestamp(inner)
                except Exception:
                    return inner
            if key == "$oid":
                return str(val)
            if key == "$numberLong":
                return int(val)
            if key == "$numberDecimal":
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return val
        # Recurse into all dict values
        return {k: _unwrap_mongo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap_mongo(item) for item in obj]
    return obj


def _looks_like_mongo_export(records: list[dict]) -> bool:
    """Heuristic: check if any sampled record contains MongoDB-style wrappers."""
    mongo_keys = {"$date", "$oid", "$numberLong", "$numberDecimal"}
    sample_text = json.dumps(records[:20])
    return any(k in sample_text for k in mongo_keys)


def _load_mongo_export(path: pathlib.Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a JSON / NDJSON file containing MongoDB extended-JSON documents."""
    warnings: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"MongoDB export file '{path.name}' is empty.")

    # Try NDJSON first (mongoexport default)
    try:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError:
        try:
            data = json.loads(text)
            records = data if isinstance(data, list) else [data]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unable to parse MongoDB export '{path.name}': {exc}") from exc

    records = [_unwrap_mongo(r) for r in records]
    df = pd.json_normalize(records, sep=".")
    warnings.append("MongoDB extended-JSON wrappers unwrapped to native types.")

    meta = _make_metadata("mongo_export", path.name, df, warnings=warnings)
    return df, meta


# ---------------------------------------------------------------------------
# Extension mapping
# ---------------------------------------------------------------------------

_EXT_MAP: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".jsonl": "ndjson",
    ".ndjson": "ndjson",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".log": "log",
    ".txt": "log",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(path: str | pathlib.Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a data file into a pandas DataFrame with accompanying metadata.

    Supports CSV, TSV, Excel (.xlsx/.xls), JSON, NDJSON/JSON Lines,
    Parquet, common log formats, and MongoDB JSON exports.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file to load.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A tuple of (dataframe, metadata) where metadata contains:
        - source_format: detected format name
        - original_filename: basename of the file
        - sheet_name: Excel sheet name (or None)
        - encoding: detected text encoding (or None for binary formats)
        - row_count: number of rows loaded
        - col_count: number of columns loaded
        - load_warnings: list of warning strings

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file extension is unsupported or the file cannot be parsed.
    ImportError
        If a required optional dependency (openpyxl, pyarrow) is missing.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    ext = path.suffix.lower()
    fmt = _EXT_MAP.get(ext)

    if fmt is None:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported extensions: {', '.join(sorted(_EXT_MAP.keys()))}"
        )

    # For JSON / NDJSON files, check for MongoDB extended-JSON wrappers
    if fmt in ("json", "ndjson"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                try:
                    sample_records = [json.loads(line) for line in text.splitlines()[:20] if line.strip()]
                except json.JSONDecodeError:
                    try:
                        data = json.loads(text)
                        sample_records = data if isinstance(data, list) else [data]
                    except json.JSONDecodeError:
                        sample_records = []
                if sample_records and _looks_like_mongo_export(sample_records):
                    return _load_mongo_export(path)
        except Exception:
            pass  # Fall through to normal loader

    loader_map = {
        "csv": _load_csv,
        "tsv": _load_tsv,
        "excel": _load_excel,
        "json": _load_json,
        "ndjson": _load_ndjson,
        "parquet": _load_parquet,
        "log": _load_log,
    }

    loader = loader_map[fmt]
    return loader(path)


def load_file_excel(
    path: str | pathlib.Path, sheet_name: str | int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a specific sheet from an Excel workbook.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the Excel file.
    sheet_name : str or int
        Name or zero-based index of the sheet to load.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A tuple of (dataframe, metadata).
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return _load_excel(path, sheet_name=sheet_name)
