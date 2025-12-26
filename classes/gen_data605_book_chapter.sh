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

# E.g., data605/lectures_source/Lesson01.1-Intro.txt
INPUT_FILE="${files[0]}"
# E.g., data605/lectures/Lesson01.1-Intro.pdf
INPUT_PDF_FILE="${INPUT_FILE/lectures_source/lectures}"
INPUT_PDF_FILE="${INPUT_PDF_FILE/.txt/.pdf}"
OUT_DIR="book"

BASENAME=$(basename "$INPUT_FILE" .txt)

helpers_root/dev_scripts_helpers/slides/generate_book_chapter.py \
    --input_file "$INPUT_FILE" \
    --input_pdf_file "$INPUT_PDF_FILE" \
    --output_dir $OUT_DIR

pandoc "$OUT_DIR/${BASENAME}.book_chapter.txt" \
    -o "$OUT_DIR/${BASENAME}.pdf" \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    --highlight-style=tango

open -a /Applications/Skim.app "$OUT_DIR/${BASENAME}.pdf"
