#!/usr/bin/env -S uv run

# /// script
# dependencies = ["feedparser", "pymupdf", "requests", "pyyaml", "beautifulsoup4", "pydantic", "tqdm"]
# ///

"""
Turn a CSV of loose paper names into a list of resolved, deduped papers.

The input CSV has one paper per row, with no header needed (a header with a
column `input` is also accepted):
- a title (with the authors in parentheses)
- an arXiv or SSRN link, a DOI, a publisher link, or the path of a local PDF

The files are written in `--output_dir`, with names that the script infers:
- `00-paper_list.csv`: a copy of the input CSV
- `01-resolve.paper_list.json`: one paper per item, from the newest to the oldest,
  with the reference, the base name, and the ordered download inputs
- `01-resolve.issues.json`: one issue per input that did not give a paper
- `01-resolve.summary.txt`: the counts, the output files, and the issues file of
  the run, the same text as the end of the log

The papers file is the input of `download_papers.py`, `extract_features.py`, and
`report_papers.py`. The script `estimate_paper_quality.py` runs all the stages in
order.

# Usage Example

- Resolve a list of papers:
> resolve_papers.py --input papers_in.csv --output_dir out

- Show how each row is classified, with no lookup and no output file:
> resolve_papers.py --input papers_in.csv --output_dir out --dry_run

For a description of this file, see the file
`research/Estimate_academic_paper_quality/resolve_papers.py.README.md`
"""

import argparse
import dataclasses
import difflib
import logging
import os
import re
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

import dev_scripts_helpers.download.download_utils as dshddut
import helpers.hcache_simple as hcacsimp
import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hprint as hprint
import helpers.hstring as hstring
import tqdm

import research.Estimate_academic_paper_quality.epq_bib_sources as epqbibso
import research.Estimate_academic_paper_quality.epq_fetcher as epqfetch
import research.Estimate_academic_paper_quality.epq_issues as epqissue
import research.Estimate_academic_paper_quality.epq_paths as epqpaths
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.epq_summary as epqsumm
import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Minimum title similarity to keep a candidate.
_MATCH_THRESHOLD = 0.9

# Similarity of a title that is a prefix of the other, e.g., a partial title.
_PREFIX_SIMILARITY = 0.95

# Number of candidates to ask each source for, and to list in an issue.
_MAX_CANDIDATES = 5
_MAX_ISSUE_CANDIDATES = 3

# New-style arXiv id, with an optional version.
_ARXIV_ID = r"\d{4}\.\d{4,5}(?:v\d+)?"


# #############################################################################
# Parse the lines
# #############################################################################


def _normalize_title(title: str) -> str:
    """
    Normalize a title to compare it with another title.

    The title is ASCII, lowercase, with no punctuation and single spaces.
    - `Lazy Prices: A Study` -> `lazy prices a study`

    :param title: title as written
    :return: normalized title
    """
    title = hstring.to_ascii(title).lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return title.strip()


def _join_wrapped_lines(raw_lines: List[str]) -> List[Tuple[int, str]]:
    """
    Join the lines of a markdown bullet that was wrapped on several lines.

    An entry starts at a line with a `-` or `*` bullet, or at a line with no
    indent. A line with an indent continues the entry before it. A blank line
    ends an entry.
    - `- Lazy Prices` + `  (Lauren Cohen)` -> `Lazy Prices (Lauren Cohen)`

    :param raw_lines: lines of the input file
    :return: pairs `(number of the first line of the entry, text)`
    """
    entries: List[Tuple[int, str]] = []
    current: Optional[Tuple[int, str]] = None
    for line_no, raw in enumerate(raw_lines, start=1):
        stripped = raw.strip()
        is_comment = stripped.startswith(("#", "//"))
        is_bullet = re.match(r"^\s*[-*]\s+", raw) is not None
        is_continuation = raw[:1] in [" ", "\t"] and not is_bullet
        if not stripped or is_comment:
            # A blank line or a comment ends the entry.
            if current is not None:
                entries.append(current)
                current = None
        elif is_continuation and current is not None:
            current = (current[0], f"{current[1]} {stripped}")
        else:
            if current is not None:
                entries.append(current)
            text = re.sub(r"^[-*]\s+", "", stripped)
            current = (line_no, text)
    if current is not None:
        entries.append(current)
    return entries


def _split_title_and_hints(text: str) -> Tuple[str, Tuple[str, ...]]:
    """
    Split `Title (Author 1; Author 2)` into the title and the author hints.

    :param text: text of a `title` line
    :return: title, and the names in the trailing parentheses
    """
    match = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", text)
    title = text
    hints: Tuple[str, ...] = ()
    if match and match.group(1).strip() and match.group(2).strip():
        title = match.group(1).strip()
        names = match.group(2)
        # The list is separated by `;`, or by `,` if there is no `;`.
        separator = ";" if ";" in names else ","
        hints = tuple(n.strip() for n in names.split(separator) if n.strip())
    return title, hints


def _parse_line(line_no: int, text: str) -> Optional[epqtypes.InputLine]:
    """
    Classify one line of the input file.

    The first test that matches wins, in the order of the download tool:
    - A local PDF, then SSRN, then a DOI, then arXiv, then a URL
    - Anything else is a title
    - A title may end with the authors in parentheses

    :param line_no: number of the line in the input file
    :param text: text of the line
    :return: classified line, `None` for an empty line or a comment
    """
    text = re.sub(r"^[-*]\s+", "", text.strip())
    if not text or text.startswith(("#", "//")):
        return None
    hints: Tuple[str, ...] = ()
    local_path = os.path.expanduser(text)
    arxiv_url = re.search(
        rf"arxiv\.org/(?:abs|pdf)/({_ARXIV_ID}|[a-z\-]+(?:\.[A-Z]{{2}})?/\d{{7}})",
        text,
        re.IGNORECASE,
    )
    arxiv_bare = re.match(rf"^(?:arxiv:\s*)?({_ARXIV_ID})$", text, re.IGNORECASE)
    arxiv_match = arxiv_url or arxiv_bare
    ssrn_id = dshddut.get_ssrn_id(text)
    doi = dshddut.detect_doi(text)
    if text.lower().endswith(".pdf") and os.path.isfile(local_path):
        kind, key = "local_pdf", os.path.abspath(local_path)
    elif ssrn_id:
        kind, key = "ssrn", ssrn_id
    elif doi:
        kind, key = "doi", doi.lower()
    elif arxiv_match:
        # Drop the version and the `.pdf` suffix, to key all the forms alike.
        arxiv_id = re.sub(r"(?:v\d+)?(?:\.pdf)?$", "", arxiv_match.group(1))
        kind, key = "arxiv", arxiv_id
    elif re.match(r"^https?://", text, re.IGNORECASE):
        kind, key = "url", text
    else:
        kind = "title"
        key, hints = _split_title_and_hints(text)
    input_line = epqtypes.InputLine(
        line_no=line_no, text=text, kind=kind, key=key, author_hints=hints
    )
    _LOG.debug(hprint.to_str("input_line"))
    return input_line


def _parse_lines(raw_lines: List[str]) -> List[epqtypes.InputLine]:
    """
    Classify the lines of the input file.

    :param raw_lines: lines of the input file
    :return: classified lines, with wrapped bullets joined
    """
    input_lines = []
    for line_no, text in _join_wrapped_lines(raw_lines):
        input_line = _parse_line(line_no, text)
        if input_line is not None:
            input_lines.append(input_line)
    return input_lines


# #############################################################################
# Identity and matching
# #############################################################################


def _family_key(author: str) -> str:
    """
    Get the normalized last word of the family name of an author.

    The last word makes `Lopez de Prado` and `Prado` the same family name.

    :param author: author name, as `Family, Given` or `Given Family`
    :return: lowercase ASCII word, "" if the name is empty
    """
    family = epqbibso.get_family_name(epqbibso.name_to_family_first(author))
    words = _normalize_title(family).split()
    return words[-1] if words else ""


def _paper_key(record: epqtypes.BibRecord) -> str:
    """
    Compute the identity of a paper: the normalized title plus the first author.

    :param record: record of a bibliographic source
    :return: key, the same for a preprint and its journal version
    """
    first_author = _family_key(record.authors[0]) if record.authors else "none"
    return f"{_normalize_title(record.title)}|{first_author}"


def _paper_aliases(record: epqtypes.BibRecord) -> Tuple[str, ...]:
    """
    List the identifiers that name the same paper as the key.

    :param record: record of a bibliographic source
    :return: aliases like `doi:10.1/x`, `arxiv:1602.04938`
    """
    aliases = []
    if record.doi:
        aliases.append(f"doi:{record.doi.lower()}")
    if record.arxiv_id:
        aliases.append(f"arxiv:{record.arxiv_id.lower()}")
    if record.ssrn_id:
        aliases.append(f"ssrn:{record.ssrn_id}")
    if record.s2_paper_id:
        aliases.append(f"s2:{record.s2_paper_id}")
    return tuple(aliases)


def _title_similarity(title1: str, title2: str) -> float:
    """
    Compute the similarity of two titles, from 0 to 1.

    - 1.0 if the normalized titles are equal
    - 0.95 if one is a prefix of the other, at a word boundary, e.g., a partial
      title or a missing subtitle
    - Else the ratio of `difflib.SequenceMatcher`

    :param title1: first title
    :param title2: second title
    :return: similarity
    """
    norm1 = _normalize_title(title1)
    norm2 = _normalize_title(title2)
    if not norm1 or not norm2:
        similarity = 0.0
    elif norm1 == norm2:
        similarity = 1.0
    elif norm1.startswith(norm2 + " ") or norm2.startswith(norm1 + " "):
        similarity = _PREFIX_SIMILARITY
    else:
        similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
    return similarity


class _UnionFind:
    """
    Group items that share an identifier.
    """

    def __init__(self, n_items: int) -> None:
        self._parent = list(range(n_items))

    def find(self, i: int) -> int:
        """
        Find the representative of the group of an item.

        :param i: index of the item
        :return: index of the representative
        """
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        """
        Put two items in the same group.

        :param i: index of the first item
        :param j: index of the second item
        """
        self._parent[self.find(i)] = self.find(j)


def _group_by_identifiers(
    identifiers: List[Tuple[str, ...]],
) -> List[List[int]]:
    """
    Group items that share at least one identifier.

    :param identifiers: identifiers of each item
    :return: groups of item indices, in the order of the first item
    """
    union_find = _UnionFind(len(identifiers))
    first_seen: Dict[str, int] = {}
    for i, item_ids in enumerate(identifiers):
        for identifier in item_ids:
            if identifier in first_seen:
                union_find.union(i, first_seen[identifier])
            else:
                first_seen[identifier] = i
    groups: Dict[int, List[int]] = {}
    for i in range(len(identifiers)):
        groups.setdefault(union_find.find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


@dataclasses.dataclass(frozen=True)
class _MatchResult:
    """
    Outcome of the match of a title with the candidates.
    """

    # One of `matched`, `unresolved`, `ambiguous`.
    status: str
    # Records of the matched paper, empty if there is no match.
    group: Tuple[epqtypes.BibRecord, ...]
    # One record per distinct paper that passed the test, for an issue.
    candidates: Tuple[epqtypes.BibRecord, ...]
    reason: str


def _match_candidates(
    query: str,
    author_hints: Tuple[str, ...],
    candidates: List[epqtypes.BibRecord],
) -> _MatchResult:
    """
    Match a title with the candidates of the bibliographic sources.

    The rule is the one of decision D3:
    - Drop a candidate with a title similarity below `_MATCH_THRESHOLD`
    - With author hints, drop a candidate that shares no family name with them
    - Group the rest by paper identity
    - One group is a match, zero groups are `unresolved`, two or more groups
      are `ambiguous`, unless exactly one group has a similarity of 1.0

    :param query: title of the line
    :param author_hints: authors in parentheses after the title
    :param candidates: records found by title
    :return: match, with the candidates if the line is ambiguous
    """
    _LOG.debug(hprint.to_str("query author_hints"))
    # Keep the candidates with a close title.
    close = []
    for record in candidates:
        similarity = _title_similarity(query, record.title)
        if similarity >= _MATCH_THRESHOLD:
            close.append((similarity, record))
    reason = f"no candidate has a title like '{query}'"
    # Keep the candidates that share a family name with the hints.
    if author_hints and close:
        hint_families = {_family_key(hint) for hint in author_hints}
        hint_families.discard("")
        with_author = [
            (similarity, record)
            for similarity, record in close
            if hint_families & {_family_key(a) for a in record.authors}
        ]
        if not with_author:
            reason = (
                "no candidate with this title shares an author with the hints"
            )
        close = with_author
    # Group the candidates that are the same paper.
    identifiers = [
        (f"key:{_paper_key(record)}",) + _paper_aliases(record)
        for _, record in close
    ]
    groups = _group_by_identifiers(identifiers)
    group_records = [tuple(close[i][1] for i in group) for group in groups]
    group_similarity = [max(close[i][0] for i in group) for group in groups]
    exact = [i for i, s in enumerate(group_similarity) if s == 1.0]
    if len(groups) == 1:
        result = _MatchResult("matched", group_records[0], (), "")
    elif len(groups) > 1 and len(exact) == 1:
        result = _MatchResult("matched", group_records[exact[0]], (), "")
    elif len(groups) > 1:
        representatives = tuple(records[0] for records in group_records)
        reason = f"{len(groups)} different papers match the title"
        result = _MatchResult("ambiguous", (), representatives, reason)
    else:
        result = _MatchResult("unresolved", (), (), reason)
    return result


# #############################################################################
# Build the papers
# #############################################################################


def _pick_canonical(
    group: Tuple[epqtypes.BibRecord, ...],
) -> epqtypes.BibRecord:
    """
    Pick the record that gives the year, the first author, and the title.

    The order is the Semantic Scholar record, else a Crossref `journal-article`,
    else any Crossref record, else the arXiv record.

    :param group: records of one paper
    :return: canonical record
    """
    by_source = {
        source: [r for r in group if r.source == source]
        for source in ["semantic_scholar", "crossref", "arxiv"]
    }
    journal = [
        r for r in by_source["crossref"] if r.record_type == "journal-article"
    ]
    for options in [
        by_source["semantic_scholar"],
        journal,
        by_source["crossref"],
        by_source["arxiv"],
    ]:
        if options:
            return options[0]
    raise ValueError("A group of records needs at least one known source")


def _format_reference(
    year: int,
    authors: Tuple[str, ...],
    title: str,
    doi: str,
    download_inputs: Tuple[str, ...],
) -> str:
    """
    Format a reference as in `.claude/skills/references.rules.md`.

    The format is `<Year>, <Last name> [et al.], "<Title>" (<Link>)`. The link
    is the first public download input. With none, the link is the DOI URL
    followed by ` ???`, or just `???` with no DOI.

    :param year: publication year, 0 if unknown
    :param authors: authors as `Family, Given`
    :param title: title
    :param doi: DOI, "" if unknown
    :param download_inputs: ordered download inputs
    :return: reference
    """
    first = epqbibso.get_family_name(authors[0]) if authors else "Unknown"
    author_text = f"{first} et al." if len(authors) > 1 else first
    public = [x for x in download_inputs if re.match(r"^https?://", x)]
    if public:
        link = public[0]
    elif doi:
        link = f"https://doi.org/{doi} ???"
    else:
        link = "???"
    year_text = str(year) if year else "n.d."
    return f'{year_text}, {author_text}, "{title}" ({link})'


def _unique(items: List[str]) -> Tuple[str, ...]:
    """
    Remove the duplicates of a list, keeping the first of each.

    :param items: items, possibly with empty ones
    :return: non-empty unique items in the original order
    """
    seen: Dict[str, None] = {}
    for item in items:
        if item and item not in seen:
            seen[item] = None
    return tuple(seen)


def _build_paper(
    group: Tuple[epqtypes.BibRecord, ...],
    input_line: epqtypes.InputLine,
    *,
    unpaywall_url: str,
) -> epqtypes.Paper:
    """
    Build a paper from the records of one matched group.

    The year, the first author, and the title come only from the canonical
    record. The download inputs are, in order: the local PDF of the line, the
    URL of the line, arXiv, the open-access PDF of Semantic Scholar, Unpaywall,
    and the SSRN page.

    :param group: records of one paper
    :param input_line: line that gave the paper
    :param unpaywall_url: open-access PDF found by Unpaywall, "" if none
    :return: paper
    """
    canonical = _pick_canonical(group)
    doi = next((r.doi for r in [canonical, *group] if r.doi), "")
    arxiv_id = next((r.arxiv_id for r in [canonical, *group] if r.arxiv_id), "")
    ssrn_id = next((r.ssrn_id for r in [canonical, *group] if r.ssrn_id), "")
    s2_id = next((r.s2_paper_id for r in group if r.s2_paper_id), "")
    # Collect the download inputs in the order of the design.
    inputs = []
    if input_line.kind == "local_pdf":
        inputs.append(input_line.key)
    if re.match(r"^https?://", input_line.text):
        inputs.append(input_line.text)
    if arxiv_id:
        inputs.append(f"https://arxiv.org/pdf/{arxiv_id}")
    s2_records = [r for r in group if r.source == "semantic_scholar"]
    for record in s2_records + [r for r in group if r not in s2_records]:
        inputs.extend(record.pdf_urls)
    inputs.append(unpaywall_url)
    if ssrn_id:
        inputs.append(
            f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={ssrn_id}"
        )
    download_inputs = _unique(inputs)
    reference = _format_reference(
        canonical.year, canonical.authors, canonical.title, doi, download_inputs
    )
    base_name = dshddut.format_base_filename(
        str(canonical.year) if canonical.year else "",
        list(canonical.authors),
        canonical.title,
    )
    key = _paper_key(canonical)
    aliases = _unique([a for r in group for a in _paper_aliases(r)])
    paper = epqtypes.Paper(
        key=key,
        aliases=aliases,
        line_nos=(input_line.line_no,),
        source=canonical.source,
        year=canonical.year,
        authors=canonical.authors,
        title=canonical.title,
        doi=doi,
        arxiv_id=arxiv_id,
        s2_paper_id=s2_id,
        reference=reference,
        base_name=base_name,
        download_inputs=download_inputs,
    )
    return paper


def _merge_papers(papers: List[epqtypes.Paper]) -> List[epqtypes.Paper]:
    """
    Merge the papers that are the same paper, e.g., two lines for one paper.

    Two papers are the same if they share the key or an alias. The merged paper
    keeps all the line numbers and the union of the download inputs, with a
    local PDF first.

    :param papers: papers from the lines
    :return: one paper per distinct paper, in the order of the first line
    """
    identifiers = [(p.key,) + p.aliases for p in papers]
    merged = []
    for group in _group_by_identifiers(identifiers):
        members = [papers[i] for i in group]
        first = members[0]
        inputs = [x for p in members for x in p.download_inputs]
        local = [x for x in inputs if not re.match(r"^https?://", x)]
        remote = [x for x in inputs if re.match(r"^https?://", x)]
        download_inputs = _unique(local + remote)
        doi = next((p.doi for p in members if p.doi), "")
        merged_paper = dataclasses.replace(
            first,
            aliases=_unique([a for p in members for a in p.aliases]),
            line_nos=tuple(sorted({n for p in members for n in p.line_nos})),
            doi=doi,
            arxiv_id=next((p.arxiv_id for p in members if p.arxiv_id), ""),
            s2_paper_id=next(
                (p.s2_paper_id for p in members if p.s2_paper_id), ""
            ),
            download_inputs=download_inputs,
            reference=_format_reference(
                first.year, first.authors, first.title, doi, download_inputs
            ),
        )
        merged.append(merged_paper)
    return merged


def _sort_papers(papers: List[epqtypes.Paper]) -> List[epqtypes.Paper]:
    """
    Sort papers in reverse chronological order.

    The order is year down, then last name of the first author, then title.

    :param papers: papers
    :return: sorted papers
    """

    def _sort_key(paper: epqtypes.Paper) -> Tuple[int, str, str]:
        first = _family_key(paper.authors[0]) if paper.authors else ""
        return (-paper.year, first, _normalize_title(paper.title))

    return sorted(papers, key=_sort_key)


def _finalize_papers(papers: List[epqtypes.Paper]) -> List[epqtypes.Paper]:
    """
    Merge the papers of the lines and sort them.

    :param papers: papers from the lines
    :return: deduped papers in reverse chronological order
    """
    return _sort_papers(_merge_papers(papers))


# #############################################################################
# Resolve the lines
# #############################################################################


@dataclasses.dataclass(frozen=True)
class _Lookup:
    """
    Records found for one line.
    """

    # True if an id gave the records: they are the paper, with no title test.
    by_id: bool
    # Title to compare with the candidates, for a lookup by title.
    query: str
    author_hints: Tuple[str, ...]
    candidates: Tuple[epqtypes.BibRecord, ...]
    # Why there is no candidate, if there is none.
    note: str


@dataclasses.dataclass(frozen=True)
class _SourceAnswers:
    """
    Answers of independent lookups, and the errors of those that failed.
    """

    answers: Tuple[Any, ...]
    failures: Tuple[str, ...]

    def get_note(self) -> str:
        """
        Build the text that tells a reader that a source was down.

        :return: text to append to a reason, empty if every source answered
        """
        if not self.failures:
            return ""
        return f" (a source failed: {'; '.join(self.failures)})"


def _call_sources(calls: List[Callable[[], Any]]) -> _SourceAnswers:
    """
    Call independent lookups and keep the answers of those that work.

    A source that is down or rate limited must not stop the line when another
    source answers. If every call fails, the last error is raised, so that an
    outage is not hidden behind a line that looks unresolved.

    :param calls: functions without arguments, one for each source
    :return: answers of the calls that worked, and the errors of the others
    """
    answers: List[Any] = []
    errors: List[RuntimeError] = []
    for call in calls:
        try:
            answers.append(call())
        except RuntimeError as e:
            _LOG.warning("A source failed, the others are used: %s", e)
            errors.append(e)
    if errors and not answers:
        raise errors[-1]
    return _SourceAnswers(
        answers=tuple(answers), failures=tuple(str(e) for e in errors)
    )


def _lookup_doi(doi: str, sources: epqbibso.BibSources) -> _Lookup:
    """
    Look up a DOI in Semantic Scholar and Crossref.

    :param doi: DOI
    :param sources: clients of the lookup APIs
    :return: records found, which are the paper with no title test
    """
    result = _call_sources(
        [
            lambda: sources.lookup_s2(f"DOI:{doi}"),
            lambda: sources.lookup_crossref(doi),
        ]
    )
    found = tuple(r for r in result.answers if r is not None)
    note = f"no record for DOI '{doi}'{result.get_note()}"
    return _Lookup(True, "", (), found, note)


def _lookup_title(
    title: str,
    author_hints: Tuple[str, ...],
    sources: epqbibso.BibSources,
) -> _Lookup:
    """
    Search a title in Semantic Scholar and Crossref.

    :param title: title or part of the title
    :param author_hints: authors in parentheses after the title
    :param sources: clients of the lookup APIs
    :return: records found, to be matched with the title
    """
    candidates: List[epqtypes.BibRecord] = []
    note = "the line has no title"
    if title:
        # The authors in the query rank the right work first in Crossref, where
        # a title alone often lists other works with similar words. The title test
        # of `_match_candidates()` still checks every candidate.
        crossref_query = " ".join([title, *author_hints])
        result = _call_sources(
            [
                lambda: sources.search_s2(title, _MAX_CANDIDATES),
                lambda: sources.search_crossref(crossref_query, _MAX_CANDIDATES),
            ]
        )
        for answer in result.answers:
            candidates.extend(answer)
        note = f"no source has a paper titled '{title}'{result.get_note()}"
    return _Lookup(False, title, author_hints, tuple(candidates), note)


def _lookup_candidates(
    input_line: epqtypes.InputLine, sources: epqbibso.BibSources
) -> _Lookup:
    """
    Look up the records of a line, by its kind.

    :param input_line: classified line
    :param sources: clients of the lookup APIs
    :return: records found
    """
    kind, key = input_line.kind, input_line.key
    if kind == "arxiv":
        result = _call_sources(
            [
                lambda: sources.lookup_arxiv(key),
                lambda: sources.lookup_s2(f"ARXIV:{key}"),
            ]
        )
        found = tuple(r for r in result.answers if r is not None)
        note = f"no record for arXiv id '{key}'{result.get_note()}"
        lookup = _Lookup(True, "", (), found, note)
    elif kind in ["doi", "ssrn"]:
        doi = key if kind == "doi" else f"10.2139/ssrn.{key}"
        lookup = _lookup_doi(doi, sources)
    elif kind == "local_pdf":
        title = epqfetch.read_pdf_title(key)
        lookup = _lookup_title(title, (), sources)
        if not title:
            lookup = dataclasses.replace(lookup, note="the PDF has no title")
    elif kind == "url":
        meta = sources.fetch_url_meta(key)
        doi_lookup = _lookup_doi(meta.doi, sources) if meta.doi else None
        if meta.pdf_path:
            # A direct PDF: look up the title of its first page.
            title = epqfetch.read_pdf_title(meta.pdf_path)
            lookup = _lookup_title(title, (), sources)
        elif doi_lookup is not None and doi_lookup.candidates:
            lookup = doi_lookup
        elif meta.title:
            hints = tuple(epqbibso.get_family_name(a) for a in meta.authors)
            lookup = _lookup_title(meta.title, hints, sources)
        else:
            lookup = _Lookup(
                False, "", (), (), "the page has no DOI and no title"
            )
    elif re.match(r"^(?:[/~]|\.{1,2}/).*\.pdf$", key, re.IGNORECASE):
        # A path that names no file: say so, instead of searching it as a title.
        lookup = _Lookup(False, "", (), (), f"the file '{key}' does not exist")
    else:
        hdbg.dassert_eq(kind, "title", "Unknown kind of line")
        lookup = _lookup_title(key, input_line.author_hints, sources)
    return lookup


def _record_to_reference(record: epqtypes.BibRecord) -> str:
    """
    Format a candidate record as a reference, for an issue.

    :param record: record of a bibliographic source
    :return: reference
    """
    return _format_reference(
        record.year, record.authors, record.title, record.doi, record.pdf_urls
    )


def _make_issue(
    input_line: epqtypes.InputLine,
    status: str,
    reason: str,
    candidates: Tuple[epqtypes.BibRecord, ...],
) -> epqtypes.Issue:
    """
    Build the issue of a line that did not give a paper.

    :param input_line: classified line
    :param status: `unresolved` or `ambiguous`
    :param reason: why the line gave no paper
    :param candidates: candidate records, at most 3 are listed
    :return: issue of the stage `resolve`
    """
    return epqtypes.Issue(
        line_no=input_line.line_no,
        input=input_line.text,
        base_name="",
        stage="resolve",
        status=status,
        reason=reason,
        candidates=tuple(
            _record_to_reference(r) for r in candidates[:_MAX_ISSUE_CANDIDATES]
        ),
    )


def _resolve_line(
    input_line: epqtypes.InputLine, sources: epqbibso.BibSources
) -> epqtypes.ResolveResult:
    """
    Resolve one line to a paper, or to an issue.

    An error of a lookup is raised: the caller catches it per line, per
    decision D21.

    :param input_line: classified line
    :param sources: clients of the lookup APIs
    :return: result with one paper, or with one issue
    """
    _LOG.debug(hprint.to_str("input_line"))
    lookup = _lookup_candidates(input_line, sources)
    if lookup.by_id:
        # The id names the paper: no title test is needed.
        if lookup.candidates:
            match = _MatchResult("matched", lookup.candidates, (), "")
        else:
            match = _MatchResult("unresolved", (), (), lookup.note)
    elif lookup.candidates:
        match = _match_candidates(
            lookup.query, lookup.author_hints, list(lookup.candidates)
        )
    else:
        match = _MatchResult("unresolved", (), (), lookup.note)
    if match.status == "matched":
        doi = next((r.doi for r in match.group if r.doi), "")
        unpaywall_url = (sources.lookup_unpaywall(doi) if doi else None) or ""
        paper = _build_paper(
            match.group, input_line, unpaywall_url=unpaywall_url
        )
        result = epqtypes.ResolveResult(papers=(paper,), issues=())
    else:
        issue = _make_issue(
            input_line, match.status, match.reason, match.candidates
        )
        result = epqtypes.ResolveResult(papers=(), issues=(issue,))
    return result


def _resolve_lines(
    raw_lines: List[str], sources: epqbibso.BibSources
) -> epqtypes.ResolveResult:
    """
    Resolve all the lines of the input file.

    An error of a lookup stops the call: the pipeline calls `_resolve_line()`
    per line instead, to isolate the failures.

    :param raw_lines: lines of the input file
    :param sources: clients of the lookup APIs
    :return: deduped, sorted papers and the issues of the lines
    """
    papers: List[epqtypes.Paper] = []
    issues: List[epqtypes.Issue] = []
    for input_line in _parse_lines(raw_lines):
        result = _resolve_line(input_line, sources)
        papers.extend(result.papers)
        issues.extend(result.issues)
    return epqtypes.ResolveResult(
        papers=tuple(_finalize_papers(papers)), issues=tuple(issues)
    )


# #############################################################################
# Resolve the CSV
# #############################################################################


def _read_input_lines(input_csv: str) -> List[epqtypes.InputLine]:
    """
    Read and classify the rows of the input CSV.

    A blank row and a row that starts with `#` or `//` are skipped.

    :param input_csv: path of the input CSV, one paper per row
    :return: classified lines, numbered by their row in the CSV
    """
    input_lines = []
    for line_no, text in epqrepor.read_input_csv(input_csv):
        input_line = _parse_line(line_no, text)
        if input_line is not None:
            input_lines.append(input_line)
    _LOG.debug("return=%s lines", len(input_lines))
    return input_lines


def _copy_input(input_csv: str, output_dir: str, *, dry_run: bool) -> None:
    """
    Copy the input CSV to the output dir, so the run keeps the list it used.

    An earlier copy is overwritten, since a rerun can use a new list. The copy is
    skipped if the input is already that file.

    :param input_csv: path of the input CSV
    :param output_dir: dir of all the outputs of the run
    :param dry_run: if True, only log the copy
    """
    copy_path = epqpaths.get_input_copy_path(output_dir)
    if dry_run:
        _LOG.warning("[DRY_RUN] Would copy '%s' to '%s'", input_csv, copy_path)
        return
    if os.path.exists(copy_path) and os.path.samefile(input_csv, copy_path):
        _LOG.debug("The input is already '%s'", copy_path)
        return
    hio.create_dir(output_dir, incremental=True)
    shutil.copyfile(input_csv, copy_path)
    _LOG.info("Copied the input to '%s'", copy_path)


def _resolve(
    input_csv: str,
    output_dir: str,
    sources: epqbibso.BibSources,
) -> epqtypes.ResolveResult:
    """
    Resolve the rows of an input CSV into a papers file and an issues file.

    The stage writes the papers file, the issues file, and the summary in
    `output_dir`.

    :param input_csv: path of the input CSV, one paper per row
    :param output_dir: dir of all the outputs of the run
    :param sources: clients of the lookup APIs
    :return: deduped papers in reverse chronological order, and the issues
    """
    input_lines = _read_input_lines(input_csv)
    _LOG.info("Resolving %s lines", len(input_lines))
    papers: List[epqtypes.Paper] = []
    issues: List[epqtypes.Issue] = []
    for input_line in tqdm.tqdm(input_lines):
        try:
            result = _resolve_line(input_line, sources)
        except Exception as e:  # noqa: BLE001
            # Decision D21: an error of one line must not stop the run.
            _LOG.warning("Line %s failed: %s", input_line.line_no, e)
            issue = epqtypes.Issue(
                line_no=input_line.line_no,
                input=input_line.text,
                base_name="",
                stage="resolve",
                status="unresolved",
                reason=epqissue.get_last_line(str(e)),
                candidates=(),
            )
            result = epqtypes.ResolveResult(papers=(), issues=(issue,))
        papers.extend(result.papers)
        issues.extend(result.issues)
    papers = _finalize_papers(papers)
    papers_file = epqpaths.get_papers_path(output_dir)
    issues_file = epqpaths.get_issues_path(output_dir, "resolve")
    epqrepor.write_papers(papers_file, papers)
    epqrepor.write_issues(issues_file, issues)
    # Summarize the stage: each line gives one paper or one issue, and the papers
    # are deduped, so the count of papers can be lower than the count of lines.
    summary = epqtypes.StageSummary(
        stage="resolve",
        unit="line",
        n_total=len(input_lines),
        n_done_before=0,
        n_no_input=0,
        n_ok=len(input_lines) - len(issues),
        n_failed=len(issues),
        outputs=((papers_file, 1),),
        issues_path=issues_file,
        issue_counts=epqsumm.count_issues(issues, add_stage=False),
        extra=(f"Papers after the dedup: {len(papers)}",),
        dry_run=False,
    )
    epqsumm.emit_stage_summary(summary, output_dir)
    return epqtypes.ResolveResult(papers=tuple(papers), issues=tuple(issues))


def _parse() -> argparse.ArgumentParser:
    # No abbreviation of a flag: the old flag `--output` would otherwise be read
    # as `--output_dir` and make a dir with the name of the old papers CSV.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        required=True,
        help="Input CSV with a column `input`, one paper per row",
    )
    # The script writes several files, so it takes a dir and infers the names
    # (rule `ARCH3.7`).
    parser.add_argument(
        "--output_dir",
        action="store",
        required=True,
        help="Dir of all the outputs and of the state of the run",
    )
    parser.add_argument(
        "--cache_dir",
        action="store",
        default="",
        help="Dir of the cache of the API answers; empty is `state/cache/` in "
        "the output dir",
    )
    parser.add_argument(
        "--email",
        action="store",
        default="",
        help="Contact email for Unpaywall; empty skips Unpaywall",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show how each row is classified, with no lookup and no output",
    )
    hcacsimp.add_cache_control_arg(parser)
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    hcacsimp.parse_cache_control_args(args)
    input_csv = os.path.abspath(os.path.expanduser(args.input))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    cache_dir = args.cache_dir or epqpaths.get_cache_dir(output_dir)
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    hdbg.dassert_file_exists(input_csv)
    _copy_input(input_csv, output_dir, dry_run=args.dry_run)
    if args.dry_run:
        # Classify the rows only: no lookup, no cache dir, no output file.
        for input_line in _read_input_lines(input_csv):
            _LOG.warning(
                "[DRY_RUN] Would resolve row %s as %s: '%s'",
                input_line.line_no,
                input_line.kind,
                input_line.key,
            )
        return
    # The cache of the helpers lives with the cache of the API answers.
    hcacsimp.set_cache_dir(cache_dir)
    sources = epqbibso.BibSources(
        email=args.email,
        s2_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        cache_dir=cache_dir,
        cache_mode=args.cache_mode,
    )
    _resolve(input_csv, output_dir, sources)


if __name__ == "__main__":
    _main(_parse())
