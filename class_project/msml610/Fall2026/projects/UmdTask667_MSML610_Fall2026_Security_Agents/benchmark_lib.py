"""
Load, validate, and anonymize the Python security benchmark.

Each snippet is one Python file under `benchmark/<type>/` that starts with a
header docstring holding its ground-truth label, e.g.,
```
\"\"\"
id: SEED-001
type: seeded
vulnerable: true
cwe: 89
...
\"\"\"
```
The headers are the single source of truth: `labels.csv` is generated from them
with `build_labels_df()` and checked by `validate_benchmark()`.

Import as:

import benchmark_lib as benlib
"""

import hashlib
import importlib.util
import logging
import os
import re
import types
from typing import Any, Dict, List, Tuple

import pandas as pd

import helpers.hdbg as hdbg
import helpers.hio as hio

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Snippet types, which are also the names of the dirs holding the snippets.
SNIPPET_TYPES = ("seeded", "secure", "cve")

# Fields of the header docstring, in the order they must appear.
HEADER_FIELDS = (
    "id",
    "type",
    "vulnerable",
    "cwe",
    "owasp",
    "severity",
    "bug_lines",
    "attack_scenario",
    "remediation",
    "source",
    "cve_id",
    "fix_commit",
    "pair_id",
    "labelled_by",
    "reviewed_by",
    "first_pass_agree",
)

# Columns of `labels.csv`.
LABEL_COLUMNS = (
    "snippet_id",
    "type",
    "vulnerable",
    "cwe",
    "owasp_2025",
    "severity",
    "bug_lines",
    "attack_scenario",
    "remediation",
    "source",
    "cve_id",
    "fix_commit",
    "pair_id",
    "labelled_by",
    "reviewed_by",
    "first_pass_agree",
    "file",
)

VULNERABLE_SEVERITIES = ("critical", "high", "medium", "low")

# Comment that marks each vulnerable line in a snippet.
LINE_MARKER = "# {!LINE}"

# Bounds on the number of code lines in a snippet, excluding the header.
MIN_CODE_LINES = 15
MAX_CODE_LINES = 80
# Real CVE code is extracted from upstream files, and the fixed version can
# add helpers (e.g., the PyJWT key format checks), so it may be longer.
MAX_CODE_LINES_CVE = 100

# Words that leak the label and must not appear anywhere in the code.
_CODE_HINT_RE = re.compile(
    r"vulnerab|\bcwe\b|cwe-\d|owasp|exploit|insecure|\bbug\b", re.IGNORECASE
)

# Words that leak the label and must not appear in comments.
# E.g., `yaml.unsafe_load()` is valid code but `# unsafe` is a hint.
_COMMENT_HINT_RE = re.compile(
    r"\bsafe\b|\bunsafe\b|injection|attack|sanitiz|malicious", re.IGNORECASE
)

_OWASP_RE = re.compile(r"^A(0[1-9]|10):2025 \S")


# #############################################################################
# Header parsing
# #############################################################################


def _split_header(text: str) -> Tuple[List[str], List[str]]:
    """
    Split a snippet into header lines and code lines.

    The header is the docstring at the top of the file. Blank lines right after
    the header are counted as part of the header.

    :param text: full content of a snippet file
    :return: header lines (including the `\"\"\"` delimiters) and code lines
    """
    lines = text.split("\n")
    hdbg.dassert_lte(1, len(lines))
    hdbg.dassert_eq(lines[0].strip(), '"""', "Snippet must start with a header")
    # Find the closing delimiter of the header.
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == '"""':
            end_idx = idx
            break
    hdbg.dassert_is_not(end_idx, None, "Header docstring is not closed")
    # Skip blank lines after the header.
    code_start = end_idx + 1
    while code_start < len(lines) and lines[code_start].strip() == "":
        code_start += 1
    header_lines = lines[:code_start]
    code_lines = lines[code_start:]
    # Drop the trailing empty line produced by the final newline.
    if code_lines and code_lines[-1] == "":
        code_lines = code_lines[:-1]
    return header_lines, code_lines


def parse_bug_lines(value: str) -> List[int]:
    """
    Parse a list of line numbers.

    E.g., `[12, 15]` -> `[12, 15]`, `12;15` -> `[12, 15]`, `[]` -> `[]`

    :param value: line numbers as `[a, b]` (header) or `a;b` (CSV)
    :return: sorted line numbers
    """
    value = value.strip().strip("[]")
    tokens = [t.strip() for t in re.split(r"[;,]", value) if t.strip()]
    for token in tokens:
        hdbg.dassert(token.isdigit(), "Invalid line number '%s'", token)
    bug_lines = sorted(int(t) for t in tokens)
    return bug_lines


def format_bug_lines(bug_lines: List[int]) -> str:
    """
    Format line numbers for `labels.csv`.

    E.g., `[12, 15]` -> `12;15`
    """
    txt = ";".join(str(line) for line in bug_lines)
    return txt


def parse_header(path: str) -> Dict[str, Any]:
    """
    Parse the header docstring of a snippet.

    Each header line has the form `key: value`; the value is everything after
    the first `: `, so values can contain colons (e.g., `A05:2025 Injection`).

    :param path: path to the snippet file
    :return: header fields, with:
        - `vulnerable` as `bool`
        - `cwe` as `int`
        - `bug_lines` as a sorted list of `int`
        - every other field as `str` (empty if not set)
    """
    text = hio.from_file(path)
    header_lines, _ = _split_header(text)
    header: Dict[str, Any] = {}
    # Skip the opening and closing delimiters and blank lines.
    for line in header_lines[1:]:
        if line.strip() in ('"""', ""):
            continue
        hdbg.dassert_in(":", line, "Invalid header line '%s' in '%s'", line, path)
        key, value = line.split(":", 1)
        key = key.strip()
        hdbg.dassert_not_in(key, header, "Duplicated key '%s' in '%s'", key, path)
        header[key] = value.strip()
    # Convert typed fields.
    if "vulnerable" in header:
        hdbg.dassert_in(header["vulnerable"], ("true", "false"))
        header["vulnerable"] = header["vulnerable"] == "true"
    if "cwe" in header:
        hdbg.dassert(header["cwe"].isdigit(), "Invalid cwe in '%s'", path)
        header["cwe"] = int(header["cwe"])
    if "bug_lines" in header:
        header["bug_lines"] = parse_bug_lines(header["bug_lines"])
    return header


def import_snippet(path: str) -> types.ModuleType:
    """
    Import a snippet file as a fresh module.

    Each call returns a new module, so module-level state (e.g., an in-memory
    database or a value read from an env var) is rebuilt.

    :param path: path to the snippet file
    :return: the imported module
    """
    hdbg.dassert_file_exists(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    module_name = "snippet_" + re.sub(r"\W", "_", stem)
    spec = importlib.util.spec_from_file_location(module_name, path)
    hdbg.dassert_is_not(spec, None)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def find_marked_lines(path: str) -> List[int]:
    """
    Find the line numbers carrying the `# {!LINE}` marker.

    :param path: path to the snippet file
    :return: 1-based line numbers in the original file
    """
    text = hio.from_file(path)
    marked_lines = [
        idx + 1
        for idx, line in enumerate(text.split("\n"))
        if line.rstrip().endswith(LINE_MARKER)
    ]
    return marked_lines


# #############################################################################
# Labels
# #############################################################################


def find_snippet_files(benchmark_dir: str) -> List[str]:
    """
    Find all snippet files in the benchmark.

    :param benchmark_dir: dir containing one subdir per snippet type
    :return: sorted paths of all `.py` files in the snippet type subdirs
    """
    hdbg.dassert_dir_exists(benchmark_dir)
    paths = []
    for snippet_type in SNIPPET_TYPES:
        type_dir = os.path.join(benchmark_dir, snippet_type)
        if not os.path.isdir(type_dir):
            continue
        for file_name in sorted(os.listdir(type_dir)):
            if file_name.endswith(".py"):
                paths.append(os.path.join(type_dir, file_name))
    return paths


def build_labels_df(benchmark_dir: str) -> pd.DataFrame:
    """
    Build the labels table from the snippet headers.

    :param benchmark_dir: dir containing the snippets
    :return: one row per snippet with columns `LABEL_COLUMNS`, sorted by
        `snippet_id`
    """
    rows = []
    for path in find_snippet_files(benchmark_dir):
        header = parse_header(path)
        row = {
            "snippet_id": header.get("id", ""),
            "type": header.get("type", ""),
            "vulnerable": header.get("vulnerable", ""),
            "cwe": header.get("cwe", ""),
            "owasp_2025": header.get("owasp", ""),
            "severity": header.get("severity", ""),
            "bug_lines": format_bug_lines(header.get("bug_lines", [])),
            "attack_scenario": header.get("attack_scenario", ""),
            "remediation": header.get("remediation", ""),
            "source": header.get("source", ""),
            "cve_id": header.get("cve_id", ""),
            "fix_commit": header.get("fix_commit", ""),
            "pair_id": header.get("pair_id", ""),
            "labelled_by": header.get("labelled_by", ""),
            "reviewed_by": header.get("reviewed_by", ""),
            "first_pass_agree": header.get("first_pass_agree", ""),
            "file": os.path.relpath(path, benchmark_dir),
        }
        rows.append(row)
    df = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    df = df.sort_values("snippet_id").reset_index(drop=True)
    return df


def write_labels(benchmark_dir: str, labels_file: str) -> pd.DataFrame:
    """
    Regenerate `labels.csv` from the snippet headers.

    :param benchmark_dir: dir containing the snippets
    :param labels_file: path of the CSV to write
    :return: the labels table that was written
    """
    df = build_labels_df(benchmark_dir)
    df.to_csv(labels_file, index=False)
    _LOG.info("Wrote %s labels to '%s'", len(df), labels_file)
    return df


def load_labels(labels_file: str) -> pd.DataFrame:
    """
    Load `labels.csv` with typed columns.

    :param labels_file: path to `labels.csv`
    :return: labels table with:
        - `vulnerable` as `bool`
        - `cwe` as `int`
        - `bug_lines` as a list of `int`
    """
    hdbg.dassert_file_exists(labels_file)
    df = pd.read_csv(labels_file, dtype=str, keep_default_na=False)
    hdbg.dassert_eq(list(df.columns), list(LABEL_COLUMNS))
    df["vulnerable"] = df["vulnerable"] == "True"
    df["cwe"] = df["cwe"].astype(int)
    df["bug_lines"] = df["bug_lines"].apply(parse_bug_lines)
    return df


# #############################################################################
# Validation
# #############################################################################


def _expected_vulnerable(snippet_type: str, snippet_id: str) -> bool:
    """
    Return the `vulnerable` value implied by the snippet type and id.

    - `seeded` snippets are vulnerable
    - `secure` snippets are safe
    - `cve` snippets are vulnerable iff their id ends with `_vuln`
    """
    if snippet_type == "seeded":
        expected = True
    elif snippet_type == "secure":
        expected = False
    elif snippet_type == "cve":
        hdbg.dassert(
            snippet_id.endswith("_vuln") or snippet_id.endswith("_fixed"),
            "CVE snippet id '%s' must end with `_vuln` or `_fixed`",
            snippet_id,
        )
        expected = snippet_id.endswith("_vuln")
    else:
        raise ValueError(f"Invalid snippet type '{snippet_type}'")
    return expected


def _find_hints(code_lines: List[str]) -> List[str]:
    """
    Find words in the code that leak the label to an agent.

    :param code_lines: code lines after the header, with markers removed
    :return: one message per offending line
    """
    hints = []
    for idx, line in enumerate(code_lines):
        match = _CODE_HINT_RE.search(line)
        if match:
            hints.append(f"code line {idx + 1}: '{match.group(0)}'")
            continue
        if "#" in line:
            comment = line.split("#", 1)[1]
            match = _COMMENT_HINT_RE.search(comment)
            if match:
                hints.append(f"comment on code line {idx + 1}: '{match.group(0)}'")
    return hints


def _validate_snippet(path: str) -> List[str]:
    """
    Validate one snippet file.

    :param path: path to the snippet file
    :return: error messages, empty if the snippet is valid
    """
    errors = []
    header = parse_header(path)
    # Check that all fields are present, in order, with no extra field.
    if tuple(header.keys()) != HEADER_FIELDS:
        missing = [f for f in HEADER_FIELDS if f not in header]
        extra = [f for f in header if f not in HEADER_FIELDS]
        errors.append(
            f"header fields are wrong (missing={missing}, extra={extra}, "
            "or not in the expected order)"
        )
        # The other checks need the fields.
        return errors
    # Check the id and type against the file location.
    file_stem = os.path.splitext(os.path.basename(path))[0]
    if header["id"] != file_stem:
        errors.append(f"id '{header['id']}' does not match file name")
    dir_name = os.path.basename(os.path.dirname(path))
    if header["type"] != dir_name:
        errors.append(f"type '{header['type']}' does not match dir '{dir_name}'")
        return errors
    # Check the vulnerable flag.
    expected = _expected_vulnerable(header["type"], header["id"])
    if header["vulnerable"] != expected:
        errors.append(f"vulnerable should be {expected} for this type and id")
    # Check bug lines, markers, and severity.
    marked_lines = find_marked_lines(path)
    if header["vulnerable"]:
        if not header["bug_lines"]:
            errors.append("vulnerable snippet has no bug_lines")
        if marked_lines != header["bug_lines"]:
            errors.append(
                f"bug_lines {header['bug_lines']} != marked lines {marked_lines}"
            )
        if header["severity"] not in VULNERABLE_SEVERITIES:
            errors.append(f"invalid severity '{header['severity']}'")
    else:
        if header["bug_lines"]:
            errors.append("safe snippet must have empty bug_lines")
        if marked_lines:
            errors.append(f"safe snippet has markers on lines {marked_lines}")
        if header["severity"] != "none":
            errors.append("safe snippet must have severity 'none'")
    # Check the OWASP category.
    if not _OWASP_RE.match(header["owasp"]):
        errors.append(f"invalid owasp '{header['owasp']}'")
    # Check the fields needed to explain and fix the snippet.
    for field in ("attack_scenario", "remediation", "source", "labelled_by"):
        if not header[field]:
            errors.append(f"empty field '{field}'")
    if header["type"] == "cve":
        for field in ("cve_id", "fix_commit"):
            if not header[field]:
                errors.append(f"empty field '{field}' for a CVE snippet")
    # Check the code size and look for hints.
    code, _ = strip_answers(path)
    code_lines = code.rstrip("\n").split("\n")
    n_lines = len(code_lines)
    max_lines = MAX_CODE_LINES_CVE if header["type"] == "cve" else MAX_CODE_LINES
    if not MIN_CODE_LINES <= n_lines <= max_lines:
        errors.append(f"{n_lines} code lines, expected {MIN_CODE_LINES}-{max_lines}")
    for hint in _find_hints(code_lines):
        errors.append(f"label hint in {hint}")
    return errors


def _validate_pairs(df: pd.DataFrame) -> List[str]:
    """
    Check that each `pair_id` links one vulnerable and one safe snippet.

    :param df: labels table from `build_labels_df()`
    :return: error messages, empty if all pairs are valid
    """
    errors = []
    paired = df[df["pair_id"] != ""]
    for pair_id, group in paired.groupby("pair_id"):
        vulnerable = sorted(group["vulnerable"].tolist())
        if vulnerable != [False, True]:
            ids = group["snippet_id"].tolist()
            errors.append(
                f"pair '{pair_id}' must have one vulnerable and one safe "
                f"snippet, got {ids}"
            )
    return errors


def validate_benchmark(benchmark_dir: str, *, labels_file: str = "") -> List[str]:
    """
    Validate all snippets and, optionally, `labels.csv`.

    :param benchmark_dir: dir containing the snippets
    :param labels_file: path to `labels.csv` to compare with the headers
        - Default: `""` (skip the comparison)
    :return: error messages prefixed by the snippet file, empty if valid
    """
    errors = []
    paths = find_snippet_files(benchmark_dir)
    for path in paths:
        rel_path = os.path.relpath(path, benchmark_dir)
        for error in _validate_snippet(path):
            errors.append(f"{rel_path}: {error}")
    df = build_labels_df(benchmark_dir)
    # Check that ids are unique.
    duplicated = df[df["snippet_id"].duplicated()]["snippet_id"].tolist()
    if duplicated:
        errors.append(f"duplicated snippet ids {duplicated}")
    errors.extend(_validate_pairs(df))
    # Check that `labels.csv` is in sync with the headers.
    if labels_file:
        if not os.path.exists(labels_file):
            errors.append(f"labels file '{labels_file}' does not exist")
        else:
            df_csv = pd.read_csv(labels_file, dtype=str, keep_default_na=False)
            if not df_csv.equals(df.astype(str)):
                errors.append(
                    f"'{labels_file}' is out of sync with the snippet headers: "
                    "run `update_labels.py`"
                )
    _LOG.debug("Validated %s snippets: %s errors", len(paths), len(errors))
    return errors


# #############################################################################
# Anonymization
# #############################################################################


def strip_answers(path: str) -> Tuple[str, int]:
    """
    Remove the label from a snippet so that an agent cannot see it.

    Removes:
    - The header docstring and the blank lines after it
    - The `# {!LINE}` markers

    :param path: path to the snippet file
    :return:
        - stripped code
        - line offset, i.e., `original_line = stripped_line + offset`
    """
    text = hio.from_file(path)
    header_lines, code_lines = _split_header(text)
    stripped_lines = []
    for line in code_lines:
        if line.rstrip().endswith(LINE_MARKER):
            line = line.rstrip()[: -len(LINE_MARKER)].rstrip()
        stripped_lines.append(line)
    code = "\n".join(stripped_lines) + "\n"
    offset = len(header_lines)
    return code, offset


def get_case_name(snippet_id: str) -> str:
    """
    Return a neutral file name for a snippet that does not reveal its label.

    E.g., `SAFE-001` -> `case_1a2b3c4d.py`

    :param snippet_id: id of the snippet
    :return: file name derived from a hash of the id
    """
    digest = hashlib.sha1(snippet_id.encode("utf-8")).hexdigest()[:8]
    case_name = f"case_{digest}.py"
    return case_name


def anonymize_benchmark(benchmark_dir: str, dst_dir: str) -> pd.DataFrame:
    """
    Write stripped copies of all snippets with neutral file names.

    Agents are run on `dst_dir`, so they see neither the label, the snippet
    type, nor a revealing file name.

    :param benchmark_dir: dir containing the snippets
    :param dst_dir: dir to write the stripped copies into
    :return: mapping table with columns:
        - `snippet_id`: id of the snippet
        - `case_file`: file name in `dst_dir`
        - `line_offset`: offset to map stripped lines to original lines
    """
    hio.create_dir(dst_dir, incremental=True)
    rows = []
    for path in find_snippet_files(benchmark_dir):
        snippet_id = parse_header(path)["id"]
        code, offset = strip_answers(path)
        case_file = get_case_name(snippet_id)
        hio.to_file(os.path.join(dst_dir, case_file), code)
        rows.append(
            {"snippet_id": snippet_id, "case_file": case_file, "line_offset": offset}
        )
    df = pd.DataFrame(rows, columns=["snippet_id", "case_file", "line_offset"])
    hdbg.dassert_no_duplicates(df["case_file"].tolist())
    _LOG.info("Wrote %s anonymized snippets to '%s'", len(df), dst_dir)
    return df
