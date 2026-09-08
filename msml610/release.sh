#!/bin/bash -xe
# `--action release` adds `release` on top of the default `generate` action,
# so each file is built fresh from source and then released in one pass.
# `--skip_action open_pdf` stops each build from popping a PDF viewer, since
# this runs generate across every file in one batch.
FILES=$(ls msml610/lectures_source/*.smd | xargs)
gen_slides.py -f "$FILES" --action release --notes_to_pdf_args="--skip_action open_pdf"

FILES=$(ls msml610/book/*.typ | xargs)
render_book_chapter.py -f "$FILES" --action release --run_typst_args="--skip_action open_pdf"
