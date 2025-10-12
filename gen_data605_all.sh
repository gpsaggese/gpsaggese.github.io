#!/bin/bash -xe
DIR=data605

shopt -s nullglob   # empty pattern expands to nothing instead of itself

files=($DIR/lectures_source/Lesson${LESSON}*)
if (( ${#files[@]} != 1 )); then
    echo "Need exactly one file"
    exit 1
else
    echo "Found file: ${files[*]}"
fi

#FILES=$(ls -1 $DIR/lectures_source/Lesson*)
FILES=$(ls -1 $DIR/lectures_source/Lesson{01,02}*)

#FILES="
#Lesson01.1-Intro.txt
#Lesson01.2-Big_Data.txt
#Lesson01.3-Data_Models.txt
#Lesson02.1-Git.txt
#Lesson02.2-Data_Pipelines.txt
#"

echo $FILES

shopt -s nullglob   # empty pattern expands to nothing instead of itself

# Loop over each file
for SRC_PATH in ${FILES}; do
    SRC_NAME=$(basename "$SRC_PATH")
    DST_NAME=$(echo "$SRC_NAME" | sed 's/\.txt$/.pdf/')

    echo "Processing $SRC_NAME -> $DST_NAME"
    notes_to_pdf.py \
        --input "$SRC_PATH" \
        --output "$DIR/lectures/$DST_NAME" \
        --type slides \
        --toc_type navigation \
        --skip_action open \
        --debug_on_error \
        $OPTS
done
