#!/bin/bash -xe
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

OPTS=${@:2}

SRC_NAME=$(cd $DIR/lectures_source; ls Lesson${LESSON}*)
DST_NAME=$(echo $SRC_NAME | sed 's/\.txt$/.script.txt/')

uv run generate_slide_script.py \
  --in_file data605/lectures_source/$SRC_NAME \
  --out_file data605/lectures_script/$DST_NAME \
  --slides_per_group 3 \
  $OPTS

lint_txt.py \
    -i data605/lectures_script/$DST_NAME \
    -o data605/lectures_script/$DST_NAME \
    --use_dockerized_prettier \
    --action prettier \
    --action frame_chapters
