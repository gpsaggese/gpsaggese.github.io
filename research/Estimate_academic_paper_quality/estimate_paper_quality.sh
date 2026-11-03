#!/usr/bin/env bash
# """
# Run all the stages of the Estimate Paper Quality pipeline, in order.
#
# Each stage is a separate executable. The executables talk to each other only
# through files in the output dir, so a stage can also run alone on the output of
# an earlier run:
#   1. `resolve_papers.py`: loose list of papers -> `01-resolve.paper_list.json`
#   2. `download_papers.py`: the papers file -> PDF and markdown of each paper
#   3. `extract_features.py`: markdown of each paper -> `*.features.json`
#   4. `report_papers.py`: `*.features.json` -> `*.rating.json`,
#      `04-report.report.csv`, `04-report.references.md`, and
#      `04-report.issues.csv`
#
# A rerun on the same output dir does not download, convert, or extract again a
# paper that is done. It rates every paper again, since the tier list and the
# citation counts can change.
#
# The scripts need the repo root and `helpers_root` in `PYTHONPATH`: the script
# adds them.
#
# # Usage Example
#
# - Run the pipeline on a CSV with one paper per row:
# > estimate_paper_quality.sh -i papers_in.csv -o out
#
# - Extract again with another model, and ask for fresh citation counts:
# > estimate_paper_quality.sh -i papers_in.csv -o out -m gpt-4o-mini -f -r
# """

set -euo pipefail

# Absolute path of the dir of this script, where the executables live.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


usage() {
  # """
  # Print the usage of the script.
  # """
  cat <<EOF
Usage: $(basename "$0") -i INPUT_CSV -o OUTPUT_DIR [-m MODEL] [-v LOG_LEVEL] [-f] [-r]

Run resolve_papers.py, download_papers.py, extract_features.py, and
report_papers.py in order.

Options:
  -i INPUT_CSV   CSV with one paper per row, no header needed (required)
  -o OUTPUT_DIR  Dir of all the outputs and of the state of the run (required)
  -m MODEL       LLM of extract_features.py (default: DEFAULT_MODEL of helpers.hllm_cli)
  -v LOG_LEVEL   Logging level forwarded to every script (default: INFO)
  -f             Extract again the papers that have a features file
  -r             Ask for fresh citation counts
  -h             Print this help
EOF
}


run_stage() {
  # """
  # Run one stage and stop the pipeline if it fails.
  #
  # :param $1: name of the stage
  # :param $@: executable of the stage, followed by its arguments
  # """
  local stage="$1"
  shift
  echo
  echo "# ############################################################################"
  echo "# Stage: ${stage}"
  echo "# ############################################################################"
  if ! "$@"; then
    echo "ERROR: the stage '${stage}' failed: fix it and run the script again" >&2
    exit 1
  fi
}


main() {
  # """
  # Parse the arguments and run the four stages in order.
  # """
  local input_csv=""
  local output_dir=""
  local model=""
  local log_level="INFO"
  local force_extract=0
  local refresh_citations=0
  # Parse the arguments.
  local opt
  while getopts "i:o:m:v:frh" opt; do
    case "${opt}" in
      i) input_csv="${OPTARG}" ;;
      o) output_dir="${OPTARG}" ;;
      m) model="${OPTARG}" ;;
      v) log_level="${OPTARG}" ;;
      f) force_extract=1 ;;
      r) refresh_citations=1 ;;
      h) usage; exit 0 ;;
      *) usage >&2; exit 1 ;;
    esac
  done
  # Check the arguments before any work starts.
  if [[ -z "${input_csv}" || -z "${output_dir}" ]]; then
    echo "ERROR: -i and -o are required" >&2
    usage >&2
    exit 1
  fi
  if [[ ! -f "${input_csv}" ]]; then
    echo "ERROR: the input CSV '${input_csv}' does not exist" >&2
    exit 1
  fi
  # Make the modules of the repo visible to the executables.
  local repo_root
  repo_root="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
  export PYTHONPATH="${repo_root}:${repo_root}/helpers_root${PYTHONPATH:+:${PYTHONPATH}}"
  # Build the optional arguments of the stages.
  local extract_args=()
  if [[ -n "${model}" ]]; then
    extract_args+=(--model "${model}")
  fi
  if [[ "${force_extract}" -eq 1 ]]; then
    extract_args+=(--force_extract)
  fi
  local report_args=()
  if [[ "${refresh_citations}" -eq 1 ]]; then
    report_args+=(--refresh_citations)
  fi
  # Every stage writes its files in the output dir with names that it infers, and
  # reads the papers file of the stage `resolve` from there.
  mkdir -p "${output_dir}"
  # Stage 1: resolve the loose list into a papers file.
  run_stage "resolve" "${SCRIPT_DIR}/resolve_papers.py" \
    --input "${input_csv}" \
    --output_dir "${output_dir}" \
    -v "${log_level}"
  # Stage 2: download the PDF of each paper and convert it to markdown.
  run_stage "download" "${SCRIPT_DIR}/download_papers.py" \
    --output_dir "${output_dir}" \
    -v "${log_level}"
  # Stage 3: extract the features of each paper with the LLM.
  run_stage "extract" "${SCRIPT_DIR}/extract_features.py" \
    --output_dir "${output_dir}" \
    ${extract_args[@]+"${extract_args[@]}"} \
    -v "${log_level}"
  # Stage 4: rate the papers and write the report.
  run_stage "report" "${SCRIPT_DIR}/report_papers.py" \
    --output_dir "${output_dir}" \
    ${report_args[@]+"${report_args[@]}"} \
    -v "${log_level}"
  echo
  echo "Done: the report is '${output_dir}/04-report.report.csv'"
}


main "$@"
