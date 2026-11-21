#!/bin/bash
# """
# Compile all the Typst book chapters in `msml610/book/*.typ` to PDF and
# compress each PDF, without opening the PDFs.
# """

set -euo pipefail

# Move to the root of the repo, since the `.typ` files import
# `/helpers_root/...` relative to it.
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# Collect all the Typst files as a single space-separated string, which is the
# format expected by `--files`.
FILES="$(ls msml610/book/*.typ)"

# Render the diagrams, compile each file to PDF, and compress the PDF.
# Skip `open_pdf` to avoid opening one viewer window per file.
run_typst.py \
    --files "${FILES}" \
    --skip_action open_pdf \
    --action compress_pdf
