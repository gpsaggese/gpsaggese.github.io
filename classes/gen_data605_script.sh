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

OUT_FILE="data605/lectures_script/$DST_NAME"

uv run generate_slide_script.py \
  --in_file data605/lectures_source/$SRC_NAME \
  --out_file $OUT_FILE \
  --slides_per_group 3 \
  $OPTS

PROMPT="You are a college professor and you need to do an introduction in 50 word the content of the slides starting with In this lesson we will discuss"
llm_cli.py -i $OUT_FILE -p "$PROMPT" -o intro.txt

PROMPT="You are a college professor and you need to summarize what was discussed in less than 50 word in the slides like In this lesson we have discussed"
llm_cli.py -i $OUT_FILE -p "$PROMPT" -o outro.txt

{
    printf '# Intro\n'
    cat intro.txt
    printf '\n'
    cat $OUT_FILE
    printf '\n# Outro\n'
    cat outro.txt
} > script.tmp && mv script.tmp $OUT_FILE

lint_txt.py \
    -i $OUT_FILE \
    -o $OUT_FILE \
    --use_dockerized_prettier \
    --action prettier \
    --action frame_chapters
