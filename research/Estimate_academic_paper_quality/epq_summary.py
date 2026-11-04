"""
Write the summary of a stage: the counts, the output files, and the issues file.

Each executable ends its stage with a summary. The summary goes to the log and to
`<output_dir>/NN-<stage>.summary.txt`, with the same text (decision D33). The
text answers four questions:
- Which output files did the stage write?
- Where is the file with the issues?
- How many items did the stage process?
- How many items did it complete successfully?

Example, for the stage `download`:
```
Stage `download` summary
  Papers in input:         20
  Skipped, done before:    12
  Processed in this run:    8
  Completed successfully:   6
  Failed:                   2
  Completed in total:      18 of 20
  Output files written in this run:
    '/out/papers/<BASE>.pdf': 6 files
    '/out/papers/<BASE>.md': 6 files
  Issues file: '/out/02-download.issues.json' (2 issues)
    download_failed: 1
    no_download_link: 1
  Summary file: '/out/02-download.summary.txt'
```

Import as:

import research.Estimate_academic_paper_quality.epq_summary as epqsumm
"""

import collections
import logging
from typing import List, Tuple

import helpers.hdbg as hdbg
import helpers.hio as hio

import research.Estimate_academic_paper_quality.epq_paths as epqpaths
import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)


# #############################################################################
# Format
# #############################################################################


def count_issues(
    issues: List[epqtypes.Issue], *, add_stage: bool
) -> Tuple[Tuple[str, int], ...]:
    """
    Count the issues by status.

    - `add_stage=False` -> `(("download_failed", 1), ("no_download_link", 2))`
    - `add_stage=True` -> `(("download.download_failed", 1), ...)`

    :param issues: issues to count
    :param add_stage: if True, the label is `<stage>.<status>`, for the issues of
        many stages
    :return: pairs of a label and a count, sorted by label
    """
    labels = [f"{i.stage}.{i.status}" if add_stage else i.status for i in issues]
    counts = tuple(sorted(collections.Counter(labels).items()))
    return counts


def _plural(count: int, word: str) -> str:
    """
    Get a count with its word in singular or plural.

    - `(1, "file")` -> `1 file`
    - `(6, "file")` -> `6 files`
    """
    suffix = "" if count == 1 else "s"
    return f"{count} {word}{suffix}"


def _get_count_rows(
    summary: epqtypes.StageSummary,
) -> List[Tuple[str, int, str]]:
    """
    Get the rows of the counts, as a label, a count, and a note.

    :param summary: summary of the stage
    :return: rows to align in columns
    """
    n_skipped = summary.n_done_before + summary.n_no_input
    rows = [(f"{summary.unit.capitalize()}s in input", summary.n_total, "")]
    if summary.n_done_before > 0:
        rows.append(("Skipped, done before", summary.n_done_before, ""))
    if summary.n_no_input > 0:
        rows.append(("Skipped, no input", summary.n_no_input, ""))
    if summary.dry_run:
        rows.append(("Would process", summary.n_total - n_skipped, ""))
    else:
        n_processed = summary.n_ok + summary.n_failed
        n_done = summary.n_done_before + summary.n_ok
        rows.append(("Processed in this run", n_processed, ""))
        rows.append(("Completed successfully", summary.n_ok, ""))
        rows.append(("Failed", summary.n_failed, ""))
        rows.append(("Completed in total", n_done, f"of {summary.n_total}"))
    return rows


def format_stage_summary(
    summary: epqtypes.StageSummary, *, summary_path: str
) -> str:
    """
    Format the summary of a stage as the text of the log and of the file.

    :param summary: summary of the stage
    :param summary_path: path of the file that holds the text, empty in a dry
        run, where no file is written
    :return: text with no final newline
    """
    n_issues = sum(count for _, count in summary.issue_counts)
    if not summary.dry_run:
        n_items = (
            summary.n_done_before
            + summary.n_no_input
            + summary.n_ok
            + summary.n_failed
        )
        hdbg.dassert_eq(
            n_items,
            summary.n_total,
            "Every item of the stage '%s' must be in one group",
            summary.stage,
        )
    prefix = "[DRY_RUN] " if summary.dry_run else ""
    lines = [f"{prefix}Stage `{summary.stage}` summary"]
    # Align the values of the counts.
    rows = _get_count_rows(summary)
    label_width = max(len(label) for label, _, _ in rows) + 1
    value_width = max(len(str(count)) for _, count, _ in rows)
    for label, count, note in rows:
        label = f"{label}:".ljust(label_width)
        line = f"  {label} {str(count).rjust(value_width)} {note}"
        lines.append(line.rstrip())
    # List the files.
    if summary.dry_run:
        lines.append("  No file is written")
    else:
        lines.append("  Output files written in this run:")
        for path, n_files in summary.outputs:
            lines.append(f"    '{path}': {_plural(n_files, 'file')}")
        issues_text = _plural(n_issues, "issue")
        lines.append(f"  Issues file: '{summary.issues_path}' ({issues_text})")
        for label, count in summary.issue_counts:
            lines.append(f"    {label}: {count}")
    for line in summary.extra:
        lines.append(f"  {line}")
    if summary_path:
        lines.append(f"  Summary file: '{summary_path}'")
    text = "\n".join(lines)
    return text


# #############################################################################
# Write
# #############################################################################


def emit_stage_summary(summary: epqtypes.StageSummary, output_dir: str) -> None:
    """
    Log the summary of a stage and write it to `NN-<stage>.summary.txt`.

    A dry run writes no file.

    :param summary: summary of the stage
    :param output_dir: dir of all the outputs of the run
    """
    summary_path = ""
    if not summary.dry_run:
        summary_path = epqpaths.get_summary_path(output_dir, summary.stage)
    text = format_stage_summary(summary, summary_path=summary_path)
    _LOG.info("\n%s", text)
    if summary_path:
        hio.to_file(summary_path, text + "\n")
