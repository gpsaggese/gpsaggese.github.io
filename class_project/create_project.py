#!/usr/bin/env -S uv run

# /// script
# dependencies = ["pandas"]
# ///

r"""
Create a project / tutorial to a specified destination directory and manage
its Docker infrastructure files relative to `class_project/project_template`.

# Usage Example

- Create a project in a target directory:
> create_project.py --dst_dir /path/to/destination

- Create a project, overwriting the destination directory if it exists:
> create_project.py --dst_dir /path/to/destination --overwrite

- Copy only the Docker files into an existing project directory:
> create_project.py --dst_dir /path/to/destination --action copy_docker_files

- Replace Docker files in a project directory with links back to the template,
  for the files that are unmodified:
> create_project.py --dst_dir /path/to/destination --action create_links

- Report which Docker files in a project directory are in sync with, linked
  to, or diverged from the template, and generate a script to `vimdiff` the
  files that are different or missing:
> create_project.py --dst_dir /path/to/destination \
    --action compare_docker_files

- Preview an action without changing anything on disk:
> create_project.py --dst_dir /path/to/destination --action create_links \
    --dry_run

- Use a custom template directory instead of the default one:
> create_project.py --src_dir /path/to/template --dst_dir /path/to/destination

For detailed documentation, architecture, and more usage examples, see:
`create_project.README.md`
"""

import argparse
import filecmp
import logging
import os
from typing import Optional

import pandas as pd

import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hprint as hprint
import helpers.hselect_action as hselacti
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)


# #############################################################################
# Constants
# #############################################################################


# Docker infrastructure files copied from `src_dir` into a project, excluding:
# - project-specific template files (e.g., # notebooks)
# - generated artifacts (e.g., `docker_build.version.log`)
# - # template's own documentation (`docker_scripts.README.md`)
_DOCKER_FILES = [
    ".dockerignore",
    "bashrc",
    "docker_bash.sh",
    "docker_build.sh",
    "docker_clean.sh",
    "docker_cmd.sh",
    "docker_exec.sh",
    "docker_jupyter.sh",
    "docker_name.sh",
    "docker_push.sh",
    "Dockerfile",
    "etc_sudoers",
    "requirements.txt",
    "run_jupyter.sh",
    "utils.sh",
    "version.sh",
]

# Template files renamed to use the project name when a new project is
# created (e.g., `template_utils.py` -> `my_project_utils.py`).
_TEMPLATE_FILES = [
    "template_utils.py",
    "template.API.ipynb",
    "template.API.py",
    "template.example.ipynb",
    "template.example.py",
]

# #############################################################################
# Helper functions
# #############################################################################


def _get_default_src_dir() -> str:
    """
    Get the default source directory containing the project template files.

    :return: absolute path to `$GIT_ROOT/class_project/project_template`
    """
    git_root = hgit.get_client_root(super_module=True)
    src_dir = os.path.join(git_root, "class_project", "project_template")
    _LOG.debug(hprint.to_str("git_root src_dir"))
    return src_dir


def _copy_files(
    src_dir: str,
    dst_dir: str,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Copy entire directory from source to destination.

    :param src_dir: source directory path
    :param dst_dir: destination directory path
    :param overwrite: whether to overwrite if destination already exists
    :param dry_run: if True, log the action without copying anything
    """
    # Verify source directory exists.
    hdbg.dassert_dir_exists(src_dir, "Source directory does not exist:", src_dir)
    # Check if destination directory already exists.
    if not overwrite:
        hdbg.dassert_path_not_exists(
            dst_dir,
            "Destination directory already exists (use --overwrite to replace)",
        )
    elif os.path.exists(dst_dir):
        # Remove existing directory to allow fresh copy.
        if dry_run:
            _LOG.info(
                "[dry_run] Would remove existing destination directory: '%s'",
                dst_dir,
            )
        else:
            _LOG.debug("Removing existing destination directory: '%s'", dst_dir)
            hsystem.system(f"rm -rf {dst_dir}")
    if dry_run:
        _LOG.info(
            "[dry_run] Would copy directory from '%s' to '%s'", src_dir, dst_dir
        )
        return
    # Copy the entire directory using cp -a.
    _LOG.info("Copying directory from '%s' to '%s'", src_dir, dst_dir)
    cmd = f"cp -a {src_dir} {dst_dir}"
    hsystem.system(cmd)
    _LOG.info("Successfully copied directory to '%s'", dst_dir)


def _rename_template_files(project_name: str, dst_dir: str) -> None:
    """
    Rename template files to use project name.

    Renames files containing "template" prefix to use the project name
    (derived from destination directory name) instead.

    :param project_name: project name to substitute for "template"
    :param dst_dir: destination directory path
    """
    for template_file in _TEMPLATE_FILES:
        src_path = os.path.join(dst_dir, template_file)
        # Skip if template file doesn't exist.
        hdbg.dassert_file_exists(src_path)
        # Create new filename by replacing template with project name.
        new_filename = template_file.replace("template", project_name)
        dst_path = os.path.join(dst_dir, new_filename)
        _LOG.info("Renaming '%s' -> '%s'", template_file, new_filename)
        hsystem.system(f"mv {src_path} {dst_path}")
    _LOG.info("Successfully renamed template files")


def _customize_files(project_name: str, dst_dir: str) -> None:
    """
    Customize files in the project directory.

    Updates docker_name.sh to use project-specific image name.

    :param project_name: project name
    :param dst_dir: destination directory path
    """
    docker_file = os.path.join(dst_dir, "docker_name.sh")
    if not os.path.exists(docker_file):
        _LOG.debug("docker_name.sh not found in '%s'", dst_dir)
        return
    # Read the file.
    with open(docker_file, "r") as f:
        content = f.read()
    # Replace IMAGE_NAME template with project-specific name.
    content = content.replace(
        "IMAGE_NAME=umd_project_template",
        f"IMAGE_NAME=umd_project_{project_name}",
    )
    # Write back the modified content.
    with open(docker_file, "w") as f:
        f.write(content)
    _LOG.info("Updated docker_name.sh with project name")


# #############################################################################


def _run_create_project(
    src_dir: str,
    dst_dir: str,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Create a new project directory from the template.

    Copies `src_dir` to `dst_dir`, renames the template files to use the
    project name, and customizes `docker_name.sh`.

    :param src_dir: source template directory
    :param dst_dir: destination directory to create
    :param overwrite: whether to overwrite `dst_dir` if it already exists
    :param dry_run: if True, log the plan without changing anything on disk
    """
    project_name = os.path.basename(dst_dir)
    _LOG.info("Project name='%s'", project_name)
    _copy_files(src_dir, dst_dir, overwrite=overwrite, dry_run=dry_run)
    if dry_run:
        # `dst_dir` was not created above, so validate renames against
        # `src_dir` instead (the two directories have the same layout).
        for template_file in _TEMPLATE_FILES:
            new_filename = template_file.replace("template", project_name)
            _LOG.info(
                "[dry_run] Would rename '%s' -> '%s'",
                template_file,
                new_filename,
            )
        _LOG.info(
            "[dry_run] Would update 'docker_name.sh' with project name '%s'",
            project_name,
        )
        return
    # Rename template files to use project name.
    _rename_template_files(project_name, dst_dir)
    # Customize project files.
    _customize_files(project_name, dst_dir)
    #
    text = f"""
    Next steps:
    - Commit the changes
      ```
      > git add tutorials/{project_name}
      > git commit -am "Add template"
      > git push
      ```
    - Change `tutorials/{project_name}/requirements.txt`
    - Edit `tutorials/{project_name}/{project_name}*.py
    """
    print(hprint.dedent(text))


# #############################################################################


def _copy_docker_files(
    src_dir: str, dst_dir: str, *, dry_run: bool = False
) -> None:
    """
    Copy all and only the Docker files from `src_dir` to `dst_dir`.

    :param src_dir: source template directory
    :param dst_dir: destination project directory
    :param dry_run: if True, log the plan without copying anything
    """
    hdbg.dassert_dir_exists(src_dir, "Source directory does not exist:", src_dir)
    if not dry_run:
        hio.create_dir(dst_dir, incremental=True)
    num_copied = 0
    for filename in _DOCKER_FILES:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        hdbg.dassert_file_exists(src_path)
        if dry_run:
            _LOG.info("[dry_run] Would copy '%s' -> '%s'", src_path, dst_path)
            num_copied += 1
            continue
        _LOG.debug("Copying '%s' -> '%s'", src_path, dst_path)
        cmd = f"cp -a {src_path} {dst_path}"
        hsystem.system(cmd)
        num_copied += 1
    _LOG.info("Copied %d Docker files to '%s'", num_copied, dst_dir)


def _create_links(src_dir: str, dst_dir: str, *, dry_run: bool = False) -> None:
    """
    Replace Docker files in `dst_dir` with links to `src_dir` when unmodified.

    For every file in `_DOCKER_FILES` that is present in `dst_dir` and is
    byte-identical to its counterpart in `src_dir`, remove the file in
    `dst_dir` and replace it with a soft link to the file in `src_dir`. Files
    that were customized (i.e., their content diverged from `src_dir`), that
    are missing from `dst_dir`, or that are already links are left untouched.

    :param src_dir: source template directory
    :param dst_dir: destination project directory
    :param dry_run: if True, log the plan without creating any link
    """
    hdbg.dassert_dir_exists(src_dir, "Source directory does not exist:", src_dir)
    hdbg.dassert_dir_exists(
        dst_dir, "Destination directory does not exist:", dst_dir
    )
    num_linked = 0
    for filename in _DOCKER_FILES:
        dst_path = os.path.join(dst_dir, filename)
        src_path = os.path.join(src_dir, filename)
        if not os.path.isfile(dst_path) or os.path.islink(dst_path):
            # Skip directories and files that are already links.
            continue
        if not os.path.isfile(src_path):
            # No corresponding file in `src_dir` to link to.
            continue
        if not filecmp.cmp(src_path, dst_path, shallow=False):
            # Content diverged: leave the customized file alone.
            continue
        if dry_run:
            _LOG.info(
                "[dry_run] Would replace '%s' with a link to '%s'",
                dst_path,
                src_path,
            )
            num_linked += 1
            continue
        os.remove(dst_path)
        hio.create_soft_link(src_path, dst_path, use_relative_path=True)
        _LOG.info("Replaced '%s' with a link to '%s'", dst_path, src_path)
        num_linked += 1
    _LOG.info("Created %d link(s) in '%s'", num_linked, dst_dir)


# #############################################################################


def _compare_docker_files(src_dir: str, dst_dir: str) -> pd.DataFrame:
    """
    Compare the Docker files in `src_dir` and `dst_dir`.

    :param src_dir: source template directory
    :param dst_dir: destination project directory
    :return: one row per file in `_DOCKER_FILES`, with columns:
        - `file`: filename
        - `in_src`: whether the file exists in `src_dir`
        - `in_dst`: whether the file exists in `dst_dir`
        - `status`: one of `same` (identical content), `link` (`dst_dir` file
          is a soft link to the `src_dir` file), `different` (content
          diverged), `missing_in_src`, `missing_in_dst`
        ```
                file  in_src  in_dst  status
        .dockerignore    True    True    same
             Dockerfile    True    True    link
        ```
    """
    hdbg.dassert_dir_exists(src_dir, "Source directory does not exist:", src_dir)
    hdbg.dassert_dir_exists(
        dst_dir, "Destination directory does not exist:", dst_dir
    )
    rows = []
    for filename in _DOCKER_FILES:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        in_src = os.path.exists(src_path)
        in_dst = os.path.exists(dst_path)
        if not in_dst:
            status = "missing_in_dst"
        elif not in_src:
            status = "missing_in_src"
        elif os.path.islink(dst_path):
            is_same_target = os.path.realpath(dst_path) == os.path.realpath(
                src_path
            )
            status = "link" if is_same_target else "different"
        elif filecmp.cmp(src_path, dst_path, shallow=False):
            status = "same"
        else:
            status = "different"
        rows.append(
            {
                "file": filename,
                "in_src": in_src,
                "in_dst": in_dst,
                "status": status,
            }
        )
    df = pd.DataFrame(rows)
    return df


def _create_vimdiff_script(
    df: pd.DataFrame,
    src_dir: str,
    dst_dir: str,
    *,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Create a script that runs `vimdiff` on the Docker files that are different
    or missing.

    :param df: comparison report, as returned by `_compare_docker_files()`
    :param src_dir: source template directory
    :param dst_dir: destination project directory
    :param dry_run: if True, log the plan without creating the script
    :return: path of the generated script, or `None` if there is nothing to
        diff
    """
    # Select the files that are different or missing on either side: files
    # that are `same` or `link` don't need to be diffed.
    mask = df["status"].isin(["different", "missing_in_src", "missing_in_dst"])
    rows = df[mask]
    if rows.empty:
        _LOG.info("No different or missing Docker files to diff")
        return None
    # Build one `vimdiff` command per file, preceded by a comment reporting
    # its status (e.g., `different`, `missing_in_dst`).
    lines = ["#!/bin/bash"]
    for _, row in rows.iterrows():
        filename = str(row["file"])
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        lines.append(f"# {filename}: {row['status']}")
        lines.append(f"vimdiff {src_path} {dst_path}")
    script_txt = "\n".join(lines) + "\n"
    script_name = "tmp.create_project.vimdiff.sh"
    if dry_run:
        _LOG.info(
            "[dry_run] Would create '%s' to vimdiff %d file(s)",
            script_name,
            len(rows),
        )
        return None
    hio.create_executable_script(
        script_name, script_txt, msg="Diff the Docker files with"
    )
    return script_name


# #############################################################################
# CLI
# #############################################################################


_VALID_ACTIONS = [
    "create_project",
    "copy_docker_files",
    "create_links",
    "compare_docker_files",
]
_DEFAULT_ACTIONS = ["create_project"]


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--src_dir",
        action="store",
        default="",
        help=(
            "Source template directory (default: "
            "$GIT_ROOT/class_project/project_template)"
        ),
    )
    parser.add_argument(
        "--dst_dir",
        action="store",
        required=True,
        help="Destination directory where files will be copied",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination directory if it already exists",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )
    hselacti.add_action_arg(parser, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Resolve the source template directory.
    src_dir = args.src_dir or _get_default_src_dir()
    hdbg.dassert_dir_exists(src_dir, "Source directory does not exist:", src_dir)
    # Select which actions to execute.
    default_actions = [] if args.action else _DEFAULT_ACTIONS
    actions = hselacti.select_actions(
        args,
        valid_actions=_VALID_ACTIONS,
        default_actions=default_actions,
    )
    _LOG.info(
        hselacti.actions_to_string(actions, _VALID_ACTIONS, add_frame=True)
    )
    while actions:
        action = actions[0]
        to_execute, actions = hselacti.mark_action(action, actions)
        if not to_execute:
            continue
        if action == "create_project":
            _run_create_project(
                src_dir,
                args.dst_dir,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        elif action == "copy_docker_files":
            _copy_docker_files(src_dir, args.dst_dir, dry_run=args.dry_run)
        elif action == "create_links":
            _create_links(src_dir, args.dst_dir, dry_run=args.dry_run)
        elif action == "compare_docker_files":
            df = _compare_docker_files(src_dir, args.dst_dir)
            print(df.to_string(index=False))
            _create_vimdiff_script(
                df, src_dir, args.dst_dir, dry_run=args.dry_run
            )
        else:
            raise ValueError(f"Invalid action='{action}'")
    hdbg.dassert_eq(
        len(actions), 0, "There are unprocessed actions: %s", str(actions)
    )


if __name__ == "__main__":
    _main(_parse())
