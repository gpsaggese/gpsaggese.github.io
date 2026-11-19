#!/bin/bash
# """
# Download all the tabs of the Google Sheet "GP's Startup Network" to the CSV
# files `startup_resources/gp_resources.<ID>.<tab_name>.csv`.
#
# The command fails if a CSV file already exists. Pass `--overwrite` to replace
# the existing CSV files and discard the local changes that are not uploaded.
# """

set -euo pipefail

# Run from the root of the repo, so that the paths are the same as in the README.
cd "$(dirname "$0")/.."

SHEET_URL="https://docs.google.com/spreadsheets/d/1Qa9g-j0NSdStV46W_dc85TMrwywIEPvJAhzV4Ymokb0/edit"

# Download each tab to one CSV file, using `--output_file` as the file prefix.
helpers_root/dev_scripts_helpers/google/from_gsheet.py \
  --url "${SHEET_URL}" \
  --tabname _ALL_ \
  --output_file startup_resources/gp_resources.csv \
  --overwrite \
  "$@"
