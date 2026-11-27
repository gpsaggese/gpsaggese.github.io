#!/usr/bin/env python
"""
Download the scores of the coding runs from Google Drive.

The scored answers of our runs and of the public EvalPlus samples are kept on
Google Drive instead of the repo, as 2 archives:
- Phase 1: `data/coding_runs/` and `data/coding_runs_pilot/`, 88 `.csv.gz`
  files
- Phase 2: `data/coding_runs_phase2/`, 18 `.csv.gz` files

This script downloads each archive, checks its SHA-256, and extracts it into
the project dir, where the notebooks look for the scores.

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
from typing import Dict, List, Tuple

import requests

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hretry as hretry

_LOG = logging.getLogger(__name__)

# For each archive: its Google Drive file, its SHA-256, and the dirs that it
# holds, relative to the project dir. The files are in the Drive folder of the
# project:
# https://drive.google.com/drive/folders/1rgI_x5rfX9HXiP0kZUHfyq2Zg8WWuV4y
ARCHIVES: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {
    "coding_runs_scores.tar.gz": (
        "15A_Oo4QA2G72yssbG7DK0NEiNlEDzvTd",
        "f5ba29b72fefd85a675c442210ac4a2efe7ba4dad5364f81e8db664f4d31dcd5",
        ("data/coding_runs/", "data/coding_runs_pilot/"),
    ),
    "coding_runs_phase2_scores.tar.gz": (
        "1Pz5HgG_cqPnM3e5uTPiVZ7dPBb--_U-u",
        "2ff21a839f8db867e4410fc8ab1599fce3bb93ef2046ddc502dd73314e9771d7",
        ("data/coding_runs_phase2/",),
    ),
}


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


def extract_scores(
    path: str, dst_dir: str, score_dirs: Tuple[str, ...]
) -> List[str]:
    """
    Extract an archive of the scores.

    Every member must be a plain file in one of `score_dirs`, so the archive
    cannot write anywhere else.

    :param path: the `.tar.gz` archive
    :param dst_dir: dir to extract into, e.g., the project dir
    :param score_dirs: dirs that the archive may write, relative to `dst_dir`,
        e.g., `("data/coding_runs_phase2/",)`
    :return: the extracted files, e.g., `data/coding_runs/...csv.gz`
    """
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            hdbg.dassert(member.isfile(), "'%s' is not a file", member.name)
            hdbg.dassert(
                member.name.startswith(score_dirs),
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
        "--archive_dir",
        action="store",
        default="data/raw",
        help="Where to save the downloaded archives",
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
    hio.create_dir(args.archive_dir, incremental=True)
    for file_name, (file_id, sha256, score_dirs) in ARCHIVES.items():
        archive_file = os.path.join(args.archive_dir, file_name)
        # Download the archive, unless a good copy is already there.
        if (
            not os.path.exists(archive_file)
            or get_sha256(archive_file) != sha256
        ):
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            _download(url, archive_file)
        # Check that it is the archive the notebooks were run on.
        hdbg.dassert_eq(
            get_sha256(archive_file),
            sha256,
            "'%s' is not the archive of the scores",
            archive_file,
        )
        names = extract_scores(archive_file, args.dst_dir, score_dirs)
        _LOG.info(
            "Extracted '%s' files from '%s' into '%s'",
            len(names),
            file_name,
            args.dst_dir,
        )


if __name__ == "__main__":
    _main(_parse())
