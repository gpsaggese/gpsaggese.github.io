#!/usr/bin/env python

"""
Process branches from input file and find project directories.

This script has two modes:

1. Branch processing mode (default):
   - Reads branch names from an input file.
   - Checks out each branch in the source directory.
   - Finds directories in class_project/MSML610/Fall2025/Projects.
   - Outputs branch names and their corresponding directories to a file.

2. Copy directories mode (--copy_dirs):
   - Reads branch-directory pairs from an input file.
   - For each pair, checks out the branch and copies the specific directory
     from source to destination.
   - Asserts if destination directory already exists.

Example usage (branch processing mode):
> classes_project/create_PR.py \
    --input_file class_project/fall2025_msml610_branches.txt \
    --source_dir /Users/saggese/src/umd_classes2 \
    --dst_dir /Users/saggese/src/umd_classes3 \
    --output_file output.txt

Example usage (copy directories mode):
> classes_project/create_PR.py \
    --input_file class_project/fall2025_msml610_branches_dirs.txt \
    --source_dir /Users/saggese/src/umd_classes2 \
    --dst_dir /Users/saggese/src/umd_classes3 \
    --copy_dirs

Import as:

import classes_project.create_PR as clprcrepr
"""

import argparse
import logging
import os
from typing import List, Tuple

from tqdm import tqdm

import helpers_root.helpers.hdbg as hdbg
import helpers_root.helpers.hio as hio
import helpers_root.helpers.hparser as hparser
import helpers_root.helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_file",
        action="store",
        required=True,
        help="Path to file containing list of branches",
    )
    parser.add_argument(
        "--source_dir",
        action="store",
        required=True,
        help="Source directory containing the git repository",
    )
    parser.add_argument(
        "--dst_dir",
        action="store",
        required=True,
        help="Destination directory (reserved for future use)",
    )
    parser.add_argument(
        "--output_file",
        action="store",
        required=False,
        help="Path to output file (required when not using --copy_dirs)",
    )
    parser.add_argument(
        "--start_from",
        action="store",
        type=int,
        default=0,
        help="Index to start processing branches from (0-indexed)",
    )
    parser.add_argument(
        "--copy_dirs",
        action="store_true",
        default=False,
        help="Enable directory copying mode",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _read_branches(input_file: str) -> List[str]:
    """
    Read branch names from input file.

    :param input_file: Path to file containing branch names
    :return: List of branch names
    """
    hdbg.dassert_path_exists(input_file)
    _LOG.info("Reading branches from: %s", input_file)
    branches = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                branches.append(line)
    _LOG.info("Found %d branches", len(branches))
    return branches


def _read_branch_directory_pairs(input_file: str) -> List[Tuple[str, str]]:
    """
    Read branch-directory pairs from input file.

    The input file format is:
    # BranchName
    DirectoryName

    :param input_file: Path to file containing branch-directory pairs
    :return: List of tuples (branch_name, directory_name)
    """
    hdbg.dassert_path_exists(input_file)
    _LOG.info("Reading branch-directory pairs from: %s", input_file)
    pairs = []
    current_branch = None
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines.
            if not line:
                continue
            # Lines starting with # are branch names.
            if line.startswith("# "):
                current_branch = line[2:].strip()
            elif current_branch is not None:
                # This is a directory name following a branch name.
                directory = line
                pairs.append((current_branch, directory))
                # Reset current_branch to ensure proper pairing.
                current_branch = None
    _LOG.info("Found %d branch-directory pairs", len(pairs))
    return pairs


def _checkout_branch(source_dir: str, branch: str) -> None:
    """
    Checkout a git branch in the source directory.

    :param source_dir: Directory containing the git repository
    :param branch: Branch name to checkout
    """
    hdbg.dassert_path_exists(source_dir)
    _LOG.debug("Checking out branch: %s", branch)
    # Change to source directory and checkout branch.
    cmd = f"cd {source_dir} && (cd helpers_root; chmod -R +w docs; git reset --hard origin/master; chmod -R +w docs)"
    hsystem.system(cmd, suppress_output=False)
    #
    cmd = f"cd {source_dir} && git clean -fd"
    hsystem.system(cmd, suppress_output=False)
    #
    cmd = f"cd {source_dir} && git reset --hard origin/master"
    hsystem.system(cmd, suppress_output=False)
    #
    cmd = f"cd {source_dir} && git clean -fd"
    hsystem.system(cmd, suppress_output=False)
    # Change to source directory and checkout branch.
    cmd = f"cd {source_dir} && git -c submodule.recurse=false checkout origin/{branch}"
    hsystem.system(cmd, suppress_output=False)


def _find_project_directories(source_dir: str) -> List[str]:
    """
    Find project directories in class_project/MSML610/Fall2025/Projects.

    Excludes specific directories that should not be included in the output.

    :param source_dir: Source directory containing the repository
    :return: List of project directory names
    """
    # Directories to exclude from the output.
    excluded_dirs = {
        "CLIP_ViT_Large_Task22",
        "Fall2025_CausalInference_Evaluating_the_Effect_of_Public_Health_Interventions_on_Disease_Spread",
        "TutorTask_67_Fall2025_AutoKeras_Electricity_Load_Forecasting",
        "TutorTask35_Fall2025_CausalML_Measuring_the_Impact_of_Lifestyle_Programs_on_Diabetes_Outcomes",
        "TutorTask60_Fall2025_Optuna_Customer_Segmentation_Using_Clustering",
        "UmdTask200_Fall2025_A_Causal_Analysis_of_Success_in_Modern_Society",
        "UmdTask27_Fall2025_HMMlearn_Anomaly_Detection_in_Network_Traffic",
    }
    projects_path = os.path.join(
        source_dir, "class_project", "MSML610", "Fall2025", "Projects"
    )
    directories = []
    if os.path.exists(projects_path):
        _LOG.debug("Searching for directories in: %s", projects_path)
        # Get all directories in the Projects folder.
        items = os.listdir(projects_path)
        for item in items:
            item_path = os.path.join(projects_path, item)
            if os.path.isdir(item_path) and item not in excluded_dirs:
                directories.append(item)
        _LOG.debug("Found %d directories (after filtering)", len(directories))
    else:
        _LOG.warning("Projects path does not exist: %s", projects_path)
    return directories


def _copy_project_directory(
    *,
    source_dir: str,
    dst_dir: str,
    directory_name: str,
) -> None:
    """
    Copy a project directory from source to destination.

    Asserts if the directory already exists in the destination.

    :param source_dir: Source directory containing the repository
    :param dst_dir: Destination directory
    :param directory_name: Name of the directory to copy
    """
    # Build source and destination paths.
    src_path = os.path.join(
        source_dir,
        "class_project",
        "MSML610",
        "Fall2025",
        "Projects",
        directory_name,
    )
    dst_projects_path = os.path.join(
        dst_dir, "class_project", "MSML610", "Fall2025", "Projects"
    )
    dst_path = os.path.join(dst_projects_path, directory_name)
    # Assert that source directory exists.
    hdbg.dassert_path_exists(
        src_path, "Source directory does not exist:", src_path
    )
    # Assert that destination directory does not exist.
    hdbg.dassert(
        not os.path.exists(dst_path),
        "Destination directory already exists:",
        dst_path,
    )
    # Create destination Projects directory if it doesn't exist.
    hio.create_dir(dst_projects_path, incremental=True)
    # Copy the directory.
    _LOG.info("Copying directory: %s -> %s", src_path, dst_path)
    cmd = f"cp -a '{src_path}' '{dst_path}'"
    hsystem.system(cmd, suppress_output=False)


def _process_branches(
    *,
    input_file: str,
    source_dir: str,
    dst_dir: str,
    output_file: str,
    start_from: int,
) -> None:
    """
    Process all branches and collect project directories.

    :param input_file: Path to file containing branch names
    :param source_dir: Source directory containing git repository
    :param dst_dir: Destination directory (reserved for future use)
    :param output_file: Path to output file
    :param start_from: Index to start processing branches from (0-indexed)
    """
    # Read all branches.
    branches = _read_branches(input_file)
    # Validate start_from index.
    hdbg.dassert_lte(
        0,
        start_from,
        "start_from must be non-negative: %s",
        start_from,
    )
    hdbg.dassert_lt(
        start_from,
        len(branches),
        "start_from exceeds number of branches: %s >= %s",
        start_from,
        len(branches),
    )
    # Initialize output file.
    _LOG.info("Writing results to: %s", output_file)
    with open(output_file, "w") as f:
        f.write("")
    # Process each branch with progress bar.
    _LOG.info("Processing branches starting from index %d...", start_from)
    for idx, branch in enumerate(tqdm(branches, desc="Processing branches")):
        # Skip branches before start_from index.
        if idx < start_from:
            _LOG.debug("Skipping branch %d: %s", idx, branch)
            continue
        _LOG.debug("Processing branch %d: %s", idx, branch)
        try:
            # Checkout branch.
            _checkout_branch(source_dir, branch)
            # Find project directories.
            directories = _find_project_directories(source_dir)
            # Format result in the required format: # Branch\ndir1\ndir2\n.
            with open(output_file, "a") as f:
                f.write(f"# {branch}\n")
                if directories:
                    for directory in sorted(directories):
                        f.write(f"{directory}\n")
                else:
                    f.write("(no directories found)\n")
                f.write("\n")
            _LOG.debug("Saved result for branch: %s", branch)
        except Exception as e:
            _LOG.error("Error processing branch %s: %s", branch, str(e))
            # Write error result to output file.
            with open(output_file, "a") as f:
                f.write(f"# {branch}\n")
                f.write(f"(error: {str(e)})\n")
                f.write("\n")
    _LOG.info("Processing complete. Results written to: %s", output_file)


def _process_branches_with_copy(
    *,
    input_file: str,
    source_dir: str,
    dst_dir: str,
    start_from: int,
) -> None:
    """
    Process all branch-directory pairs and copy directories.

    :param input_file: Path to file containing branch-directory pairs
    :param source_dir: Source directory containing git repository
    :param dst_dir: Destination directory
    :param start_from: Index to start processing branches from (0-indexed)
    """
    # Read all branch-directory pairs.
    pairs = _read_branch_directory_pairs(input_file)
    # Validate start_from index.
    hdbg.dassert_lte(
        0,
        start_from,
        "start_from must be non-negative:",
        start_from,
    )
    hdbg.dassert_lt(
        start_from,
        len(pairs),
        "start_from exceeds number of pairs:",
        start_from,
        len(pairs),
    )
    # Process each branch-directory pair with progress bar.
    _LOG.info("Processing branches starting from index %d...", start_from)
    for idx, (branch, directory) in enumerate(
        tqdm(pairs, desc="Processing branches")
    ):
        # Skip pairs before start_from index.
        if idx < start_from:
            _LOG.debug("Skipping pair %d: %s -> %s", idx, branch, directory)
            continue
        _LOG.debug("Processing pair %d: %s -> %s", idx, branch, directory)
        try:
            # Checkout branch.
            _checkout_branch(source_dir, branch)
            # Copy the directory.
            _copy_project_directory(
                source_dir=source_dir,
                dst_dir=dst_dir,
                directory_name=directory,
            )
            _LOG.info(
                "Successfully copied directory for branch %s: %s",
                branch,
                directory,
            )
        except Exception as e:
            _LOG.error(
                "Error processing branch %s (directory %s): %s",
                branch,
                directory,
                str(e),
            )
            raise
    _LOG.info("Processing complete.")


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Validate inputs.
    hdbg.dassert_path_exists(args.input_file)
    hdbg.dassert_path_exists(args.source_dir)
    # Choose processing mode based on copy_dirs flag.
    if args.copy_dirs:
        # Copy directories mode.
        _LOG.info("Running in copy directories mode")
        hdbg.dassert_path_exists(args.dst_dir)
        _process_branches_with_copy(
            input_file=args.input_file,
            source_dir=args.source_dir,
            dst_dir=args.dst_dir,
            start_from=args.start_from,
        )
    else:
        # Original mode: process branches and generate output.
        _LOG.info("Running in branch processing mode")
        hdbg.dassert(
            args.output_file is not None,
            "--output_file is required when not using --copy_dirs",
        )
        _process_branches(
            input_file=args.input_file,
            source_dir=args.source_dir,
            dst_dir=args.dst_dir,
            output_file=args.output_file,
            start_from=args.start_from,
        )


if __name__ == "__main__":
    _main(_parse())
