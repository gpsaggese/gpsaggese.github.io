#!/usr/bin/env python

"""
Copy Docker-related files from the source directory to a destination directory.

This script copies all Docker configuration and utility files from
class_project/docker_common/ to a specified destination directory.

Usage examples:
    # Copy all files to a target directory.
    > ./copy_docker_files.py --dst_dir /path/to/destination

    # Copy with verbose logging.
    > ./copy_docker_files.py --dst_dir /path/to/destination -v DEBUG

Import as:

import class_project.docker_common.copy_docker_files as cpdccodo
"""

import argparse
import logging
import os
from typing import List

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# List of files to copy from the source directory.
_FILES_TO_COPY = [
    "bashrc",
    "docker_bash.sh",
    "docker_build.sh",
    "docker_clean.sh",
    "docker_cmd.sh",
    "docker_exec.sh",
    "docker_jupyter.sh",
    "docker_name.sh",
    "docker_push.sh",
    "etc_sudoers",
    "install_jupyter_extensions.sh",
    "run_jupyter.sh"
    "version.sh",
]


# #############################################################################
# Helper functions
# #############################################################################


def _get_source_dir() -> str:
    """
    Get the absolute path to the source directory containing Docker files.

    :return: absolute path to class_project/docker_common/
    """
    # Get the directory where this script is located.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    _LOG.debug("Script directory='%s'", script_dir)
    return script_dir


def _copy_files(
    *,
    src_dir: str,
    dst_dir: str,
    files: List[str],
) -> None:
    """
    Copy specified files from source directory to destination directory.

    :param src_dir: source directory path
    :param dst_dir: destination directory path
    :param files: list of filenames to copy
    """
    # Verify source directory exists.
    hdbg.dassert_dir_exists(src_dir, "Source directory does not exist:", src_dir)
    # Create destination directory if it doesn't exist.
    hio.create_dir(dst_dir, incremental=True)
    _LOG.info("Copying %d files from '%s' to '%s'", len(files), src_dir, dst_dir)
    # Copy each file.
    copied_count = 0
    for filename in files:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        # Verify source file exists.
        hdbg.dassert_path_exists(
            src_path, "Source file does not exist:", src_path
        )
        # Copy the file using cp -a to preserve all permissions and attributes.
        _LOG.debug("Copying '%s' -> '%s'", src_path, dst_path)
        cmd = f"cp -a {src_path} {dst_path}"
        hsystem.system(cmd)
        copied_count += 1
    #
    _LOG.info("Successfully copied %d files", copied_count)


# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dst_dir",
        action="store",
        required=True,
        help="Destination directory where files will be copied",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Get source directory.
    src_dir = _get_source_dir()
    # Copy files to destination.
    _copy_files(
        src_dir=src_dir,
        dst_dir=args.dst_dir,
        files=_FILES_TO_COPY,
    )


if __name__ == "__main__":
    _main(_parse())
