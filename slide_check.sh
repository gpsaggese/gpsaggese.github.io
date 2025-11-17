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

<<<<<<< HEAD
SRC_NAME=$(cd $DIR/lectures_source; ls Lesson${LESSON}*)
DIR_NAME="processed.txt"
OPTS=${@:2}

process_slides.py --in_file $SRC_NAME --action slide_check --out_file $DST_NAME --use_llm_transform $OPTS
=======
SRC_NAME=$(ls $DIR/lectures_source/Lesson${LESSON}*)
DST_NAME=$SRC_NAME
#DST_NAME="processed.txt"
OPTS=${@:2}

ACTION=text_check_fix
#ACTION=text_check
process_slides.py \
    --in_file $SRC_NAME \
    --action $ACTION \
    --out_file $DST_NAME \
    --use_llm_transform \
    $OPTS
>>>>>>> master
