#!/bin/bash -xe

# Check that exactly two arguments are provided
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

LESSON=$1
# E.g., data605, msml610
DIR=$2

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
OUT_DIR="data605/book"
echo "OUT_DIR=$OUT_DIR"

BASENAME=$(basename "$INPUT_FILE" .txt)

HELPERS_ROOT_DIR=$(find . -type d -path "./helpers_root/dev_scripts_helpers")
echo "HELPERS_ROOT_DIR=$HELPERS_ROOT_DIR"

$HELPERS_ROOT_DIR/slides/generate_book_chapter.py \
    --input_file "$INPUT_FILE" \
    --input_pdf_file "$INPUT_PDF_FILE" \
    --output_dir $OUT_DIR

pandoc "$OUT_DIR/${BASENAME}.book_chapter.txt" \
    -o "$OUT_DIR/${BASENAME}.pdf" \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    --highlight-style=tango \
    --include-in-header=$HELPERS_ROOT_DIR/slides/header-style.tex

open -a /Applications/Skim.app "$OUT_DIR/${BASENAME}.pdf"
