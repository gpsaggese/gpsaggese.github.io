#!/bin/bash
# """
# Render all the slide sources in `msml610/lectures_source/*.smd` to PDF and
# compress each PDF, without opening the PDFs.
# """

set -euo pipefail

# Move to the root of the repo, since `gen_slides.py` resolves the lecture
# paths relative to it.
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# Collect all the slide sources as a single space-separated string, which is
# the format expected by `--files`.
FILES="$(ls msml610/lectures_source/*.smd | xargs)"

# Render each file to `msml610/lectures_pdf.tmp/`, then release it to
# `msml610/lectures_pdf/`, compressing the PDF in place.
# Skip `open_pdf` to avoid opening one viewer window per file.
# Continue past failures and print a summary of what failed at the end.
gen_slides.py \
    --files "${FILES}" \
    --action release \
    --notes_to_pdf_args="--skip_action open_pdf" \
    --no_abort_on_error
