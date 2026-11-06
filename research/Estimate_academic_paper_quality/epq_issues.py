"""
Build the issues that the stages report for one paper.

A stage must not stop on the failure of one paper (decision D21): it turns the
failure into an `Issue` and goes on.

Import as:

import research.Estimate_academic_paper_quality.epq_issues as epqissue
"""

import research.Estimate_academic_paper_quality.epq_types as epqtypes


def get_last_line(message: str) -> str:
    """
    Keep the last non-empty line of an error message as the reason.

    The frame lines of a failed `hdbg.dassert` are only `#`, and they are
    skipped, since the line before them has the message.

    :param message: text of an exception
    :return: last line, or `unknown error` if the message is empty
    """
    lines = [line.strip() for line in message.split("\n") if line.strip()]
    lines = [line for line in lines if set(line) != {"#"}]
    return lines[-1] if lines else "unknown error"


def make_paper_issue(
    paper: epqtypes.Paper, stage: str, status: str, reason: str
) -> epqtypes.Issue:
    """
    Build the issue of a paper that failed in a stage.

    :param paper: paper that failed
    :param stage: one of `download`, `extract`, `rate`
    :param status: status of the issue
    :param reason: why the paper failed
    :return: issue tied to the first line that gave the paper
    """
    return epqtypes.Issue(
        line_no=min(paper.line_nos),
        input=paper.reference,
        base_name=paper.base_name,
        stage=stage,
        status=status,
        reason=reason,
        candidates=(),
    )
