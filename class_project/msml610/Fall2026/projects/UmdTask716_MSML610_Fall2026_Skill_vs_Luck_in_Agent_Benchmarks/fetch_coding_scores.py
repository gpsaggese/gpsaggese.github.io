#!/usr/bin/env python
"""
Download the scores of the coding runs from Google Drive.

The scored answers of our runs and of the public EvalPlus samples
(`data/coding_runs/` and `data/coding_runs_pilot/`, 88 `.csv.gz` files) are
kept on Google Drive instead of the repo. This script downloads them as 1
archive, checks its SHA-256, and extracts it into the project dir, where the
notebooks look for them.

# Usage Example

- Download the scores into `data/`:
> fetch_coding_scores.py

Import as:

import fetch_coding_scores as fecosc
"""

import argparse
import hashlib
import logging
import os
import tarfile
from typing import List

import requests

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hretry as hretry

_LOG = logging.getLogger(__name__)

# Google Drive file with the archive of the scores, and its SHA-256. The
# file is in the Drive folder of the project:
# https://drive.google.com/drive/folders/1rgI_x5rfX9HXiP0kZUHfyq2Zg8WWuV4y
_ARCHIVE_ID = "15A_Oo4QA2G72yssbG7DK0NEiNlEDzvTd"
_ARCHIVE_SHA256 = (
    "f5ba29b72fefd85a675c442210ac4a2efe7ba4dad5364f81e8db664f4d31dcd5"
)
# Dirs that the archive holds, relative to the project dir.
SCORE_DIRS = ("data/coding_runs/", "data/coding_runs_pilot/")


# #############################################################################


@hretry.sync_retry(
    (requests.RequestException,), num_attempts=5, retry_delay_in_sec=5
)
def _download(url: str, dst: str) -> None:
    """
    Download a file, retrying a few times if the connection drops.

    :param url: URL to download
    :param dst: where to save it
    """
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(dst, "wb") as file:
            for chunk in response.iter_content(chunk_size=1 << 20):
                file.write(chunk)


def get_sha256(path: str) -> str:
    """
    Compute the SHA-256 of a file.

    :param path: file to hash
    :return: hex digest
    """
    with open(path, "rb") as file:
        digest = hashlib.sha256(file.read()).hexdigest()
    return digest


def extract_scores(path: str, dst_dir: str) -> List[str]:
    """
    Extract the archive of the scores.

    Every member must be a plain file in one of `SCORE_DIRS`, so the archive
    cannot write anywhere else.

    :param path: the `.tar.gz` archive
    :param dst_dir: dir to extract into, e.g., the project dir
    :return: the extracted files, e.g., `data/coding_runs/...csv.gz`
    """
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            hdbg.dassert(member.isfile(), "'%s' is not a file", member.name)
            hdbg.dassert(
                member.name.startswith(SCORE_DIRS),
                "'%s' is outside the score dirs",
                member.name,
            )
            hdbg.dassert_not_in("..", member.name.split("/"))
        archive.extractall(dst_dir, members=members, filter="data")
    names = [member.name for member in members]
    return names


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--archive_file",
        action="store",
        default="data/raw/coding_runs_scores.tar.gz",
        help="Where to save the downloaded archive",
    )
    parser.add_argument(
        "--dst_dir",
        action="store",
        default=".",
        help="Dir to extract the scores into, the project dir by default",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Download the archive, unless a good copy is already there.
    hio.create_dir(os.path.dirname(args.archive_file), incremental=True)
    if (
        not os.path.exists(args.archive_file)
        or get_sha256(args.archive_file) != _ARCHIVE_SHA256
    ):
        url = f"https://drive.google.com/uc?export=download&id={_ARCHIVE_ID}"
        _download(url, args.archive_file)
    # Check that it is the archive the notebooks were run on.
    hdbg.dassert_eq(
        get_sha256(args.archive_file),
        _ARCHIVE_SHA256,
        "'%s' is not the archive of the scores",
        args.archive_file,
    )
    names = extract_scores(args.archive_file, args.dst_dir)
    _LOG.info("Extracted '%s' files into '%s'", len(names), args.dst_dir)


if __name__ == "__main__":
    _main(_parse())
