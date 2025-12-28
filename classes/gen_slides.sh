#!/bin/bash -xe

# Check that exactly two arguments are provided.
if [ "$#" -ne 2 ]; then
    echo "Error: Expected exactly 2 arguments, got $#"
    echo ""
    echo "Usage: $0 <DIR> <LESSON>"
    echo ""
    echo "Arguments:"
    echo "  DIR     - Course directory (e.g., data605, msml610)"
    echo "  LESSON  - Lesson number (e.g., 01.1, 02.3)"
    echo ""
    echo "Example:"
    echo "  $0 data605 0.1"
    exit 1
fi

# E.g., data605, msml610
DIR=$1
# E.g., 01.1
LESSON=$2

files=($DIR/lectures_source/Lesson${LESSON}*)
if (( ${#files[@]} != 1 )); then
    echo "Need exactly one file"
    exit 1
else
    echo "Found file: ${files[*]}"
fi

OPTS=${@:3}

SRC_NAME=$(cd $DIR/lectures_source; ls Lesson${LESSON}*)
DST_NAME=$(echo $SRC_NAME | sed 's/\.txt$/.pdf/')
OPTS_DEBUG="--skip_action cleanup_before --skip_action cleanup_after"
notes_to_pdf.py \
    --input $DIR/lectures_source/$SRC_NAME \
    --output $DIR/lectures/$DST_NAME \
    --type slides --toc_type navigation --debug_on_error \
    $OPTS_DEBUG \
    $OPTS
