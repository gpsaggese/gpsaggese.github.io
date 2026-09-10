#!/bin/bash -xe
# `--action release` adds `release` on top of the default `generate` action,
# so each file is built fresh from source and then released in one pass.
# `--skip_action open_pdf` stops each build from popping a PDF viewer, since
# this runs generate across every file in one batch.

if [[ 0 == 1 ]]; then

lint_text.py --files "$(ls msml610/lectures_source/Lesson0{1,2}*.smd | xargs)"

#FILES=$(ls msml610/lectures_source/Lesson0{1,2}*.smd | xargs)
#gen_slides.py -f "$FILES" --action release --notes_to_pdf_args="--skip_action open_pdf"

for_loop_lessons.py --class msml610 --lectures "01*:02*" --action release_slides_pdf

fi;

lint_text.py --files "$(ls msml610/book/Lesson0{1,2}*.typ | xargs)"

FILES=$(ls msml610/book/*.typ | xargs)
for_loop_lessons.py --class msml610 --lectures "01.2:02*" --action release_pdf
