"""
Paths of the files that the executables of the project share.

The executables talk to each other only through files on disk (decision D26).
This module is the one place that names those files, so that the writer and the
reader of a file always agree on its path.

A file of a stage is named `NN-<stage>.<what>.<ext>`, where `NN` is the order of
the executable in the pipeline (decision D29). All the paths live under one
output dir:
```
<output_dir>/
    00-paper_list.csv
    01-resolve.paper_list.json
    01-resolve.issues.json
    01-resolve.summary.txt
    02-download.issues.json
    02-download.summary.txt
    03-extract.issues.json
    03-extract.summary.txt
    04-rate.issues.json
    04-rate.summary.txt
    04-report.report.json
    04-report.issues.json
    04-report.references.md
    04-report.summary.txt
    papers/<BASE>.pdf
    papers/<BASE>.md
    papers/<BASE>.features.json
    papers/<BASE>.rating.json
    state/cache/
    state/logs/
```

Import as:

import research.Estimate_academic_paper_quality.epq_paths as epqpaths
"""

import os

import helpers.hdbg as hdbg

# Order of the executable that runs each stage: `rate` and `report` are stages
# of `report_papers.py`, so they share a number.
_STAGE_NUMBERS = {
    "resolve": 1,
    "download": 2,
    "extract": 3,
    "rate": 4,
    "report": 4,
}


def get_papers_dir(output_dir: str) -> str:
    """
    Get the dir of the PDF, markdown, features, and rating files of the papers.

    :param output_dir: dir of all the outputs of a run
    :return: `<output_dir>/papers`
    """
    return os.path.join(output_dir, "papers")


def get_state_dir(output_dir: str) -> str:
    """
    Get the dir of the HTTP cache and the logs of the download tool.

    :param output_dir: dir of all the outputs of a run
    :return: `<output_dir>/state`
    """
    return os.path.join(output_dir, "state")


def get_cache_dir(output_dir: str) -> str:
    """
    Get the dir of the cache of the HTTP answers.

    :param output_dir: dir of all the outputs of a run
    :return: `<output_dir>/state/cache`
    """
    return os.path.join(get_state_dir(output_dir), "cache")


def get_log_dir(output_dir: str) -> str:
    """
    Get the dir of the logs of the download tool.

    :param output_dir: dir of all the outputs of a run
    :return: `<output_dir>/state/logs`
    """
    return os.path.join(get_state_dir(output_dir), "logs")


def get_markdown_path(output_dir: str, base_name: str) -> str:
    """
    Get the path of the markdown of a paper.

    :param output_dir: dir of all the outputs of a run
    :param base_name: base name of the paper
    :return: `<output_dir>/papers/<BASE>.md`
    """
    return os.path.join(get_papers_dir(output_dir), f"{base_name}.md")


def get_pdf_path(output_dir: str, base_name: str) -> str:
    """
    Get the path of the PDF of a paper.

    :param output_dir: dir of all the outputs of a run
    :param base_name: base name of the paper
    :return: `<output_dir>/papers/<BASE>.pdf`
    """
    return os.path.join(get_papers_dir(output_dir), f"{base_name}.pdf")


def get_features_path(output_dir: str, base_name: str) -> str:
    """
    Get the path of the features file of a paper.

    :param output_dir: dir of all the outputs of a run
    :param base_name: base name of the paper
    :return: `<output_dir>/papers/<BASE>.features.json`
    """
    return os.path.join(get_papers_dir(output_dir), f"{base_name}.features.json")


def get_rating_path(output_dir: str, base_name: str) -> str:
    """
    Get the path of the rating file of a paper.

    :param output_dir: dir of all the outputs of a run
    :param base_name: base name of the paper
    :return: `<output_dir>/papers/<BASE>.rating.json`
    """
    return os.path.join(get_papers_dir(output_dir), f"{base_name}.rating.json")


def get_stage_path(output_dir: str, stage: str, name: str) -> str:
    """
    Get the path of a file that a stage writes in the output dir.

    - `("run", "download", "issues.json")` -> `run/02-download.issues.json`

    :param output_dir: dir of all the outputs of a run
    :param stage: one of `resolve`, `download`, `extract`, `rate`, `report`
    :param name: what the file holds, with its extension, e.g., `issues.json`
    :return: `<output_dir>/NN-<stage>.<name>`
    """
    hdbg.dassert_in(stage, _STAGE_NUMBERS)
    number = _STAGE_NUMBERS[stage]
    return os.path.join(output_dir, f"{number:02d}-{stage}.{name}")


def get_input_copy_path(output_dir: str) -> str:
    """
    Get the path of the copy of the input CSV of the run.

    :param output_dir: dir of all the outputs of a run
    :return: `<output_dir>/00-paper_list.csv`
    """
    return os.path.join(output_dir, "00-paper_list.csv")


def get_papers_path(output_dir: str) -> str:
    """
    Get the path of the papers that `resolve_papers.py` wrote.

    :param output_dir: dir of all the outputs of a run
    :return: `<output_dir>/01-resolve.paper_list.json`
    """
    return get_stage_path(output_dir, "resolve", "paper_list.json")


def get_issues_path(output_dir: str, stage: str) -> str:
    """
    Get the path of the issues that a stage wrote.

    :param output_dir: dir of all the outputs of a run
    :param stage: one of `resolve`, `download`, `extract`, `rate`
    :return: `<output_dir>/NN-<stage>.issues.json`
    """
    hdbg.dassert_ne(stage, "report", "The stage `report` writes no stage issues")
    return get_stage_path(output_dir, stage, "issues.json")


def get_summary_path(output_dir: str, stage: str) -> str:
    """
    Get the path of the summary that a stage wrote.

    :param output_dir: dir of all the outputs of a run
    :param stage: one of `resolve`, `download`, `extract`, `rate`, `report`
    :return: `<output_dir>/NN-<stage>.summary.txt`
    """
    return get_stage_path(output_dir, stage, "summary.txt")


def get_report_path(output_dir: str) -> str:
    """
    Get the path of `report.json`.

    :param output_dir: dir of all the outputs of a run
    :return: `<output_dir>/04-report.report.json`
    """
    return get_stage_path(output_dir, "report", "report.json")


def get_all_issues_path(output_dir: str) -> str:
    """
    Get the path of `issues.json` of the stage `report`, which holds the issues
    of every stage.

    :param output_dir: dir of all the outputs of a run
    :return: `<output_dir>/04-report.issues.json`
    """
    return get_stage_path(output_dir, "report", "issues.json")


def get_references_path(output_dir: str) -> str:
    """
    Get the path of `references.md`.

    :param output_dir: dir of all the outputs of a run
    :return: `<output_dir>/04-report.references.md`
    """
    return get_stage_path(output_dir, "report", "references.md")
