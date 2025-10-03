#!/bin/bash -xe
LESSON=$1
DIR=msml610

shopt -s nullglob   # empty pattern expands to nothing instead of itself

files=($DIR/lectures_source/Lesson${LESSON}*)
if (( ${#files[@]} == 0 )); then
    echo "No files found"
    exit 1
else
    echo "Found files: ${files[*]}"
fi

SRC_NAME=$(cd $DIR/lectures_source; ls Lesson${LESSON}*); DST_NAME=$(echo $SRC_NAME | sed 's/\.txt$/.pdf/'); notes_to_pdf.py --input $DIR/lectures_source/$SRC_NAME --output $DIR/lectures/$DST_NAME --type slides --toc_type navigation --debug_on_error
