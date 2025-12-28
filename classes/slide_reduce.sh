#!/bin/bash -e
LESSON=$1
DIR=data605

shopt -s nullglob   # empty pattern expands to nothing instead of itself

files=($DIR/lectures_source/Lesson${LESSON}*)
if (( ${#files[@]} != 1 )); then
    echo "Need exactly one file"
    exit 1
else
    echo "Found file: ${files[*]}"
fi

SRC_NAME=$(ls $DIR/lectures_source/Lesson${LESSON}*)
OPTS=${@:2}

process_slides.py --in_file $SRC_NAME --action slide_reduce --out_file $SRC_NAME --use_llm_transform $OPTS
