#!/bin/bash
# """
# Upload the CSV files `startup_resources/gp_resources.<ID>.<tab_name>.csv` to
# the tabs of the Google Sheet "GP's Startup Network".
#
# The command replaces the content of the existing tabs. This cannot be undone,
# except with the version history of Google Sheets.
# """

set -euo pipefail

# Run from the root of the repo, so that the paths are the same as in the README.
cd "$(dirname "$0")/.."

SHEET_URL="https://docs.google.com/spreadsheets/d/1Qa9g-j0NSdStV46W_dc85TMrwywIEPvJAhzV4Ymokb0/edit"

# Upload each CSV file to one tab, using `--input_file` as the file prefix.
helpers_root/dev_scripts_helpers/google/to_gsheet.py \
  --input_file startup_resources/gp_resources.csv \
  --url "${SHEET_URL}" \
  --tabname _ALL_ \
  --overwrite \
  "$@"
