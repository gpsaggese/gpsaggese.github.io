#!/usr/bin/env python

"""
Run all the stages of the Estimate Paper Quality pipeline, in order.

Each stage is a separate executable. The executables talk to each other only
through files in `--output_dir`, so a stage can also run alone on the output of
an earlier run. The script runs each executable as a subprocess, and does not
import it. There is one action per executable, named after it:
- `resolve_papers`: `resolve_papers.py` turns a loose list of papers into
  `01-resolve.paper_list.json`
- `download_papers`: `download_papers.py` turns the papers file into the PDF and
  the markdown of each paper
- `extract_features`: `extract_features.py` turns the markdown of each paper into
  `*.features.json`
- `report_papers`: `report_papers.py` turns `*.features.json` into
  `*.rating.json`, `04-report.report.csv`, `04-report.references.md`, and
  `04-report.issues.csv`

The actions are selected with `--action`, `--skip_action`, `--only_action`,
`--clear_actions`, and `--all_actions`. The default is to run all of them, in the
order above. The script stops at the first action that fails.

A rerun on the same `--output_dir` does not download, convert, or extract again a
paper that is done. It rates every paper again, since the tier list and the
citation counts can change.

The script adds the repo root and `helpers_root` to `PYTHONPATH` for the
executables.

# Usage Example

- Run the pipeline on a CSV with one paper per row:
> estimate_paper_quality.py --input papers_in.csv --output_dir out

- Extract again with another model, and ask for fresh citation counts:
> estimate_paper_quality.py --input papers_in.csv --output_dir out --model gpt-4o-mini --force_extract --refresh_citations

- Run only the actions `extract_features` and `report_papers` on the output of an
  earlier run:
> estimate_paper_quality.py --output_dir out --only_action extract_features --only_action report_papers

- Show the command of each action, with no execution:
> estimate_paper_quality.py --input papers_in.csv --output_dir out --dry_run

For a description of this file, see the file
`research/Estimate_academic_paper_quality/estimate_paper_quality.py.README.md`
"""

import argparse
import logging
import os
import shlex

import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hparser as hparser
import helpers.hprint as hprint
import helpers.hselect_action as hselacti
import helpers.hsystem as hsystem

import research.Estimate_academic_paper_quality.epq_paths as epqpaths

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# One action per executable, named after it: the executable of an action is
# `<action>.py`. The actions are in the order the stages run.
_VALID_ACTIONS = [
    "resolve_papers",
    "download_papers",
    "extract_features",
    "report_papers",
]
_DEFAULT_ACTIONS = _VALID_ACTIONS


# #############################################################################
# Environment
# #############################################################################


def _add_repo_to_python_path() -> None:
    """
    Make the modules of the repo visible to the executables.

    The executables import `research.Estimate_academic_paper_quality` from the
    repo root and `helpers` from `helpers_root`. The subprocesses inherit
    `PYTHONPATH` from this process.
    """
    repo_root = hgit.get_client_root(super_module=True)
    new_paths = [repo_root, os.path.join(repo_root, "helpers_root")]
    old_paths = [
        path
        for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if path and path not in new_paths
    ]
    python_path = os.pathsep.join(new_paths + old_paths)
    _LOG.debug("PYTHONPATH='%s'", python_path)
    os.environ["PYTHONPATH"] = python_path


# #############################################################################
# Stages
# #############################################################################


def _build_command(
    action: str,
    executable: str,
    *,
    input_csv: str,
    output_dir: str,
    model: str,
    force_extract: bool,
    refresh_citations: bool,
    log_level: str,
) -> str:
    """
    Build the command line of one action.

    Every executable gets `--output_dir`, where it finds the files of the earlier
    stages. Only `resolve_papers.py` gets `--input`: the other executables find
    `01-resolve.paper_list.json` in the output dir.

    :param action: action to run, one of `_VALID_ACTIONS`
    :param executable: path of the executable of the action
    :param input_csv: CSV with one paper per row, for `resolve_papers.py`
    :param output_dir: dir of all the outputs and of the state of the run
    :param model: LLM of `extract_features.py`, empty for the default of the
        executable
    :param force_extract: if True, `extract_features.py` extracts again the papers
        that have a features file
    :param refresh_citations: if True, `report_papers.py` asks for fresh citation
        counts
    :param log_level: logging level to forward to the executable
    :return: command line
    """
    _LOG.debug(hprint.to_str("action executable input_csv output_dir model"))
    hdbg.dassert_in(action, _VALID_ACTIONS)
    cmd = [executable]
    if action == "resolve_papers":
        cmd.append(f"--input {shlex.quote(input_csv)}")
    cmd.append(f"--output_dir {shlex.quote(output_dir)}")
    if action == "extract_features":
        if model:
            cmd.append(f"--model {shlex.quote(model)}")
        if force_extract:
            cmd.append("--force_extract")
    elif action == "report_papers":
        if refresh_citations:
            cmd.append("--refresh_citations")
    # Forward the log level so that it applies to every executable.
    cmd.append(f"-v {log_level}")
    cmd_str = " ".join(cmd)
    _LOG.debug("return=%s", cmd_str)
    return cmd_str


def _run_action(action: str, cmd: str, *, dry_run: bool) -> None:
    """
    Run the command of one action and stop the pipeline if it fails.

    :param action: action to run, for the log
    :param cmd: command line of the action
    :param dry_run: if True, only log the command
    """
    _LOG.debug(hprint.to_str("action cmd dry_run"))
    if dry_run:
        _LOG.warning("[DRY_RUN] Would run the action '%s': %s", action, cmd)
    else:
        _LOG.info("Running the action '%s'", action)
        # Show the output of the executable, which `hsystem` hides by default.
        # An error of the executable raises and stops the pipeline.
        hsystem.system(cmd, suppress_output=False)


# #############################################################################
# Main
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        default="",
        help="CSV with one paper per row, no header needed: the input of the "
        "action `resolve_papers`",
    )
    # The flag `--output_dir` follows decision D10: the output dir is required
    # and the user decides where it lives.
    parser.add_argument(
        "--output_dir",
        action="store",
        required=True,
        help="Dir of all the outputs and of the state of the run",
    )
    parser.add_argument(
        "--model",
        action="store",
        default="",
        help="LLM of `extract_features.py`; empty is the default model of the "
        "script",
    )
    hparser.add_bool_arg(
        parser,
        "force_extract",
        default_value=False,
        help_="Extract again the papers that have a features file",
    )
    hparser.add_bool_arg(
        parser,
        "refresh_citations",
        default_value=False,
        help_="Ask for fresh citation counts in the action `report_papers`",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show the command of each action, with no execution",
    )
    hselacti.add_action_arg(parser, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Select the actions to run.
    actions = hselacti.select_actions(args, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    _LOG.info(
        "\n%s",
        hselacti.actions_to_string(actions, _VALID_ACTIONS, add_frame=True),
    )
    selected_actions = list(actions)
    # Check the boundary of the system before any work starts.
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    input_csv = ""
    if "resolve_papers" in selected_actions:
        hdbg.dassert_ne(
            args.input, "", "The action `resolve_papers` needs `--input`"
        )
        input_csv = os.path.abspath(os.path.expanduser(args.input))
        hdbg.dassert_file_exists(input_csv)
    elif args.input:
        _LOG.warning(
            "Ignore `--input` since the action `resolve_papers` is not "
            "selected: the other actions read the papers file from '%s'",
            output_dir,
        )
    _add_repo_to_python_path()
    # Run the actions in order.
    while actions:
        action = actions[0]
        to_execute, actions = hselacti.mark_action(action, actions)
        if not to_execute:
            continue
        executable = hgit.find_file_in_git_tree(f"{action}.py")
        cmd = _build_command(
            action,
            executable,
            input_csv=input_csv,
            output_dir=output_dir,
            model=args.model,
            force_extract=args.force_extract,
            refresh_citations=args.refresh_citations,
            log_level=args.log_level,
        )
        _run_action(action, cmd, dry_run=args.dry_run)
    hdbg.dassert_eq(actions, [], "Unprocessed actions: %s", str(actions))
    if "report_papers" in selected_actions and not args.dry_run:
        _LOG.info("The report is '%s'", epqpaths.get_report_path(output_dir))
    _LOG.info("Outputs are in '%s'", output_dir)


if __name__ == "__main__":
    _main(_parse())
