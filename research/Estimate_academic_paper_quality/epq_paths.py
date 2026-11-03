"""
Paths of the files that the executables of the project share.

The executables talk to each other only through files on disk (decision D26).
This module is the one place that names those files, so that the writer and the
reader of a file always agree on its path.

All the paths live under one output dir:
```
<output_dir>/
    papers/<BASE>.pdf
    papers/<BASE>.md
    papers/<BASE>.features.json
    papers/<BASE>.rating.json
    state/issues.<stage>.json
    state/cache/
    state/logs/
```

Import as:

import research.Estimate_academic_paper_quality.epq_paths as epqpaths
"""

import os


def get_papers_dir(output_dir: str) -> str:
    """
    Get the dir of the PDF, markdown, features, and rating files of the papers.

    :param output_dir: dir of all the outputs of a run
    :return: `<output_dir>/papers`
    """
    return os.path.join(output_dir, "papers")


def get_state_dir(output_dir: str) -> str:
    """
    Get the dir of the issues of the stages, the logs, and the HTTP cache.

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


def get_issues_path(output_dir: str, stage: str) -> str:
    """
    Get the path of the issues that a stage wrote.

    :param output_dir: dir of all the outputs of a run
    :param stage: one of `download`, `extract`, `rate`
    :return: `<output_dir>/state/issues.<stage>.json`
    """
    return os.path.join(get_state_dir(output_dir), f"issues.{stage}.json")


def get_resolve_issues_path(papers_file: str) -> str:
    """
    Get the default path of the issues CSV that goes with a papers CSV.

    - `run/papers.csv` -> `run/papers.issues.csv`

    :param papers_file: path of the papers CSV
    :return: path of the issues CSV of `resolve_papers.py`
    """
    return os.path.splitext(papers_file)[0] + ".issues.csv"
