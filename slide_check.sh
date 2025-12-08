#!/bin/bash -e
LESSON=$1
DIR=data605

shopt -s nullglob   # empty pattern expands to nothing instead of itself

files=($DIR/lectures_source/Lesson${LESSON}*)
if (( ${#files[@]} != 1 )); then
<<<<<<< HEAD
<<<<<<< HEAD
    echo "Need exactly one file"
=======
    echo "Need exactly one file. Found file: ${files[*]}"
>>>>>>> acc7434fbbd3a13e7d6fab709755918b215726e4
=======
    echo "Need exactly one file. Found file: ${files[*]}"
>>>>>>> 641d70d0fd5943e647d36753863ed7313ac79733
    exit 1
else
    echo "Found file: ${files[*]}"
fi

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
