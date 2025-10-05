#!/bin/bash -xe
LESSON=$1
DIR=data605

shopt -s nullglob   # empty pattern expands to nothing instead of itself

files=($DIR/lectures_source/Lesson${LESSON}*)
if (( ${#files[@]} == 0 )); then
    echo "No files found"
    exit 1
else
    echo "Found files: ${files[*]}"
fi

OPTS=${@:2}

SRC_NAME=$(cd $DIR/lectures_source; ls Lesson${LESSON}*)
DST_NAME=$(echo $SRC_NAME | sed 's/\.txt$/.script.txt/')

generate_slide_script.py \
  --in_file $DIR/lectures_source/$SRC_NAME \
  --out_file $DIR/lectures_script/$DST_NAME \
  --slides_per_group 3 \
  $OPTS

perl -pi -e 's/^Transition: //g' $DIR/lectures_script/$DST_NAME

lint_txt.py -i $DIR/lectures_script/$DST_NAME --use_dockerized_prettier
