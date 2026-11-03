#!/usr/bin/env -S uv run

# /// script
# dependencies = ["pyyaml"]
# ///

"""
Populate `affiliation_tiers.yaml` from an export of the QS World University
Rankings.

The script reads a CSV that a person exports from the QS site, and writes the
universities of the top ranks into the tier list of the affiliation signal:
- Rank 1 to `--tier_5_max_rank` go in `tier_5`
- Rank `--tier_5_max_rank` + 1 to `--tier_4_max_rank` go in `tier_4`

The script never reads a rank from memory and never fetches the QS site, which
blocks scripts. The names and the ranks come from the CSV only.

The script rewrites the whole YAML file and keeps what a person added by hand:
- An entry with no `qs_rank` key is a hand entry (an industry lab, a central
  bank, NBER): it is kept as it is
- The `aliases` of an entry with the same name are kept, since the QS export does
  not have them

Review the result with `git diff` before you rate again.

# Usage Example

- Fill the tiers from a CSV exported from the QS site:
> populate_affiliation_tiers.py \
    --qs_file ~/Downloads/qs_2026.csv \
    --edition "QS World University Rankings 2026" \
    --source "https://www.topuniversities.com/world-university-rankings"

- Show the new file without writing it:
> populate_affiliation_tiers.py \
    --qs_file ~/Downloads/qs_2026.csv \
    --edition "QS World University Rankings 2026" \
    --source "https://www.topuniversities.com/world-university-rankings" \
    --dry_run

- Name the columns of the CSV when the script cannot find them:
> populate_affiliation_tiers.py \
    --qs_file ~/Downloads/qs_2026.csv \
    --edition "QS World University Rankings 2026" \
    --source "https://www.topuniversities.com/world-university-rankings" \
    --rank_col "2026 Rank" \
    --name_col "Institution Name"

For a description of the YAML file and of this script, see the file
`research/Estimate_academic_paper_quality/affiliation_tiers.yaml.README.md`
"""

import argparse
import csv
import datetime
import io
import json
import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Set, Tuple

import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hparser as hparser
import yaml

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Name of the YAML file, which `report_papers.py` also looks up.
_YAML_NAME = "affiliation_tiers.yaml"

# Names of the columns of a QS export that the script finds with no flag, in
# lowercase and in the order of preference.
_RANK_COLUMNS = ["rank", "rank_display", "qs rank", "overall rank"]
_NAME_COLUMNS = [
    "institution",
    "institution name",
    "university",
    "name",
    "title",
]

# Comment at the top of the YAML file.
_HEADER = """\
# Dated list of the institutions that score 5 and 4 in the affiliation signal.
#
# The script `populate_affiliation_tiers.py` writes the entries that have a
# `qs_rank` key, from a CSV exported from the QS World University Rankings site.
# Never fill them from the memory of an LLM (`docs/03-decisions.md`, D15).
# An entry with no `qs_rank` key is a hand entry (an industry lab, a central
# bank, NBER): the script keeps it. Add the `aliases` of an entry by hand: the
# script keeps them.
#
# How to fill the file: `affiliation_tiers.yaml.README.md`."""


# #############################################################################
# Read the QS export
# #############################################################################


def _normalize_name(text: str) -> str:
    """
    Normalize an institution name to compare two names.

    The name is ASCII, lowercase, with no punctuation and single spaces.
    - `École Polytechnique (EPFL)` -> `ecole polytechnique epfl`

    :param text: institution name
    :return: normalized name
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return text


def _find_column(
    headers: List[str], candidates: List[str], *, substring: str = ""
) -> str:
    """
    Find the header of the column that holds a kind of data.

    :param headers: headers of the CSV
    :param candidates: lowercase names to try, in the order of preference
    :param substring: lowercase text; if no candidate matches, accept the
        only header that contains it
    :return: header as written in the CSV
    """
    lowered = {header.strip().lower(): header for header in headers}
    found = ""
    for candidate in candidates:
        if candidate in lowered:
            found = lowered[candidate]
            break
    if not found and substring:
        matches = [header for key, header in lowered.items() if substring in key]
        if len(matches) == 1:
            found = matches[0]
    hdbg.dassert_ne(
        found,
        "",
        "Cannot find the column among %s: pass it with a flag",
        headers,
    )
    return found


def _parse_rank(text: str) -> int:
    """
    Parse the rank of a row of the QS export.

    The site writes a tie as `=20`, a band as `101-110`, and an open band as
    `601+`. The rank is the first number.
    - `=20` -> 20
    - `101-110` -> 101

    :param text: text of the rank cell
    :return: rank, 0 if the text has no number
    """
    match = re.match(r"^\s*=?\s*(\d+)", text)
    rank = int(match.group(1)) if match else 0
    return rank


def _split_name(raw_name: str) -> Tuple[str, List[str]]:
    """
    Split a QS name into the name and the aliases written in parentheses.

    A paper writes `Massachusetts Institute of Technology`, not the name with
    the acronym, so the acronym is an alias, not a part of the name.
    - `Massachusetts Institute of Technology (MIT)` ->
      (`Massachusetts Institute of Technology`, [`MIT`])

    :param raw_name: name as written in the QS export
    :return: pair `(name, aliases)`
    """
    aliases = [
        " ".join(alias.split())
        for alias in re.findall(r"\(([^)]*)\)", raw_name)
        if alias.strip()
    ]
    name = " ".join(re.sub(r"\([^)]*\)", " ", raw_name).split())
    if not name:
        # The whole text is in parentheses: keep it as the name.
        name = " ".join(raw_name.split())
    return name, aliases


def _read_qs_rows(
    path: str, *, rank_col: str, name_col: str
) -> List[Tuple[int, str]]:
    """
    Read the rank and the name of each institution of a QS export.

    A row with no rank number or no name is skipped.

    :param path: path of the CSV
    :param rank_col: header of the rank column, empty to find it
    :param name_col: header of the name column, empty to find it
    :return: pairs `(rank, raw name)` in the order of the file
    """
    # Drop the BOM that a spreadsheet adds at the start of a CSV.
    text = hio.from_file(path).lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text))
    headers = list(reader.fieldnames or [])
    if not rank_col:
        rank_col = _find_column(headers, _RANK_COLUMNS, substring="rank")
    if not name_col:
        name_col = _find_column(headers, _NAME_COLUMNS)
    hdbg.dassert_in(rank_col, headers)
    hdbg.dassert_in(name_col, headers)
    _LOG.info("Rank column='%s', name column='%s'", rank_col, name_col)
    rows = []
    for row in reader:
        rank = _parse_rank(row[rank_col] or "")
        name = (row[name_col] or "").strip()
        if rank == 0 or not name:
            _LOG.warning("Skip the row with rank='%s', name='%s'", rank, name)
            continue
        rows.append((rank, name))
    hdbg.dassert_lt(0, len(rows), "No row with a rank and a name in '%s'", path)
    return rows


# #############################################################################
# Merge with the existing file
# #############################################################################


def _load_entries(data: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    """
    Get the entries of a tier from the YAML dict.

    :param data: content of the existing YAML file, empty if it does not exist
    :param key: `tier_5` or `tier_4`
    :return: entries, empty if the key is empty
    """
    entries = []
    for item in data.get(key) or []:
        entry = dict(item)
        entry["name"] = str(entry["name"])
        entry["aliases"] = [str(alias) for alias in entry.get("aliases") or []]
        entries.append(entry)
    return entries


def _unique_aliases(name: str, groups: List[List[str]]) -> List[str]:
    """
    Merge groups of aliases, with no repeat and no copy of the name.

    :param name: name of the entry
    :param groups: groups of aliases, the first group comes first
    :return: aliases in the order of the groups
    """
    seen = {_normalize_name(name)}
    aliases = []
    for group in groups:
        for alias in group:
            key = _normalize_name(alias)
            if key and key not in seen:
                seen.add(key)
                aliases.append(alias)
    return aliases


def _build_qs_entries(
    rows: List[Tuple[int, str]],
    kept_aliases: Dict[str, List[str]],
    *,
    tier_5_max_rank: int,
    tier_4_max_rank: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Turn the QS rows into the entries of the two tiers.

    :param rows: pairs `(rank, raw name)`
    :param kept_aliases: aliases of the existing file, by normalized name
    :param tier_5_max_rank: last rank of the tier 5
    :param tier_4_max_rank: last rank of the tier 4
    :return: pair `(tier_5 entries, tier_4 entries)`, each by rank
    """
    tier_5: List[Dict[str, Any]] = []
    tier_4: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    # The sort is stable, so the ties stay in the order of the file.
    for rank, raw_name in sorted(rows, key=lambda row: row[0]):
        if rank > tier_4_max_rank:
            continue
        name, name_aliases = _split_name(raw_name)
        key = _normalize_name(name)
        if key in seen:
            _LOG.warning("Skip the repeated name='%s'", name)
            continue
        seen.add(key)
        entry = {
            "name": name,
            "aliases": _unique_aliases(
                name, [name_aliases, kept_aliases.get(key, [])]
            ),
            "qs_rank": rank,
        }
        if rank <= tier_5_max_rank:
            tier_5.append(entry)
        else:
            tier_4.append(entry)
    return tier_5, tier_4


def _keep_hand_entries(
    existing: List[Dict[str, Any]], qs_names: Set[str]
) -> List[Dict[str, Any]]:
    """
    Keep the entries that a person added by hand.

    An entry is by hand if it has no `qs_rank` key. If its name is also a QS
    name, the QS entry replaces it.

    :param existing: entries of one tier of the existing file
    :param qs_names: normalized names of all the QS entries
    :return: hand entries
    """
    kept = [
        entry
        for entry in existing
        if "qs_rank" not in entry
        and _normalize_name(entry["name"]) not in qs_names
    ]
    return kept


# #############################################################################
# Write the YAML
# #############################################################################


def _dump(value: Any) -> str:
    """
    Dump a value as a JSON text, which is also a valid YAML text.

    :param value: string, number, or list of them
    :return: one-line text
    """
    return json.dumps(value, ensure_ascii=False)


def _format_entry(entry: Dict[str, Any]) -> List[str]:
    """
    Format an entry as the lines of a YAML list item.

    JSON strings are valid YAML strings, so they quote the names safely.

    :param entry: entry with the keys `name` and `aliases`, and any other key
    :return: lines of the item
    """
    lines = [
        f"  - name: {_dump(entry['name'])}",
        f"    aliases: {_dump(entry['aliases'])}",
    ]
    for key in sorted(set(entry) - {"name", "aliases"}):
        lines.append(f"    {key}: {_dump(entry[key])}")
    return lines


def _format_tier(
    key: str,
    comment: str,
    qs_entries: List[Dict[str, Any]],
    hand_entries: List[Dict[str, Any]],
) -> List[str]:
    """
    Format one tier of the YAML file.

    :param key: `tier_5` or `tier_4`
    :param comment: comment line above the key
    :param qs_entries: entries from the QS export
    :param hand_entries: entries added by hand
    :return: lines of the tier
    """
    lines = [comment]
    if not qs_entries and not hand_entries:
        lines.append(f"{key}: []")
    else:
        lines.append(f"{key}:")
        for entry in qs_entries:
            lines.extend(_format_entry(entry))
        if hand_entries:
            lines.append("  # Entries added by hand: the script keeps them.")
        for entry in hand_entries:
            lines.extend(_format_entry(entry))
    return lines


def _build_yaml(
    rows: List[Tuple[int, str]],
    existing: Dict[str, Any],
    *,
    as_of: str,
    source: str,
    tier_5_max_rank: int,
    tier_4_max_rank: int,
) -> str:
    """
    Build the text of the YAML file.

    :param rows: pairs `(rank, raw name)` of the QS export
    :param existing: content of the existing YAML file, empty if none
    :param as_of: value of the `as_of` key
    :param source: value of the `source` key
    :param tier_5_max_rank: last rank of the tier 5
    :param tier_4_max_rank: last rank of the tier 4
    :return: text of the file, with a final newline
    """
    existing_5 = _load_entries(existing, "tier_5")
    existing_4 = _load_entries(existing, "tier_4")
    # Keep the aliases that a person wrote for an institution of the file.
    kept_aliases = {
        _normalize_name(entry["name"]): entry["aliases"]
        for entry in existing_5 + existing_4
    }
    qs_5, qs_4 = _build_qs_entries(
        rows,
        kept_aliases,
        tier_5_max_rank=tier_5_max_rank,
        tier_4_max_rank=tier_4_max_rank,
    )
    qs_names = {_normalize_name(entry["name"]) for entry in qs_5 + qs_4}
    hand_5 = _keep_hand_entries(existing_5, qs_names)
    hand_4 = _keep_hand_entries(existing_4, qs_names)
    _LOG.info(
        "tier_5: %d QS and %d hand entries, tier_4: %d QS and %d hand entries",
        len(qs_5),
        len(hand_5),
        len(qs_4),
        len(hand_4),
    )
    if len(qs_5) < tier_5_max_rank:
        _LOG.warning(
            "The CSV has %d institutions of rank 1 to %d: check that it is the "
            "whole table",
            len(qs_5),
            tier_5_max_rank,
        )
    lines = [
        _HEADER,
        f"as_of: {_dump(as_of)}",
        f"source: {_dump(source)}",
    ]
    lines.extend(
        _format_tier(
            "tier_5",
            f"# QS rank 1 to {tier_5_max_rank}, plus leading industry research "
            "labs, central banks, NBER.",
            qs_5,
            hand_5,
        )
    )
    lines.extend(
        _format_tier(
            "tier_4",
            f"# QS rank {tier_5_max_rank + 1} to {tier_4_max_rank}, plus "
            "established industry research groups.",
            qs_4,
            hand_4,
        )
    )
    text = "\n".join(lines) + "\n"
    # Check that the file reads back with the same number of entries.
    parsed = yaml.safe_load(text)
    hdbg.dassert_eq(len(parsed["tier_5"]), len(qs_5) + len(hand_5))
    hdbg.dassert_eq(len(parsed["tier_4"]), len(qs_4) + len(hand_4))
    return text


# #############################################################################
# Script
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--qs_file",
        action="store",
        required=True,
        help="CSV exported from the QS World University Rankings site",
    )
    parser.add_argument(
        "--edition",
        action="store",
        required=True,
        help="Edition of the ranking, e.g., 'QS World University Rankings 2026'",
    )
    parser.add_argument(
        "--source",
        action="store",
        required=True,
        help="URL of the QS page that the CSV comes from",
    )
    parser.add_argument(
        "--accessed",
        action="store",
        default="",
        help="Date when the CSV was exported, as YYYY-MM-DD; default is today",
    )
    parser.add_argument(
        "--yaml_file",
        action="store",
        default="",
        help="YAML file to write; default is '%s' in the Git tree" % _YAML_NAME,
    )
    parser.add_argument(
        "--rank_col",
        action="store",
        default="",
        help="Header of the rank column; default is to find it",
    )
    parser.add_argument(
        "--name_col",
        action="store",
        default="",
        help="Header of the institution name column; default is to find it",
    )
    parser.add_argument(
        "--tier_5_max_rank",
        action="store",
        type=int,
        default=20,
        help="Last QS rank that scores 5",
    )
    parser.add_argument(
        "--tier_4_max_rank",
        action="store",
        type=int,
        default=100,
        help="Last QS rank that scores 4",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Log the new file and do not write it",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    hdbg.dassert_file_exists(args.qs_file)
    hdbg.dassert_lt(0, args.tier_5_max_rank)
    hdbg.dassert_lte(args.tier_5_max_rank, args.tier_4_max_rank)
    yaml_file = args.yaml_file or hgit.find_file_in_git_tree(_YAML_NAME)
    accessed = args.accessed or datetime.date.today().isoformat()
    # Read the inputs.
    rows = _read_qs_rows(
        args.qs_file, rank_col=args.rank_col, name_col=args.name_col
    )
    existing: Dict[str, Any] = {}
    if os.path.isfile(yaml_file):
        existing = yaml.safe_load(hio.from_file(yaml_file)) or {}
    # Build the new file.
    text = _build_yaml(
        rows,
        existing,
        as_of=f"{args.edition}, accessed {accessed}",
        source=args.source,
        tier_5_max_rank=args.tier_5_max_rank,
        tier_4_max_rank=args.tier_4_max_rank,
    )
    if args.dry_run:
        _LOG.info("Dry run: the file '%s' would be:\n%s", yaml_file, text)
    else:
        hio.to_file(yaml_file, text)
        _LOG.info("Wrote the file '%s'", yaml_file)


if __name__ == "__main__":
    _main(_parse())
