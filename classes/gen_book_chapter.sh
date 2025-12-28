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

# E.g., data605, msml610
DIR=$1
# E.g., 01.1
LESSON=$2

shopt -s nullglob   # empty pattern expands to nothing instead of itself

FILES_DIR="$DIR/lectures_source/Lesson${LESSON}*"
echo "FILES_DIR=$FILES_DIR"

files=$(ls -1 $FILES_DIR)
if (( ${#files[@]} != 1 )); then
    echo "Need exactly one file"
    exit 1
else
    echo "Found file: ${files[*]}"
fi

# 1) Generate the PDF.
SRC_NAME=$(cd $DIR/lectures_source; ls Lesson${LESSON}*)
DST_NAME=$(echo tmp.$SRC_NAME | sed 's/\.txt$/.pdf/')
notes_to_pdf.py \
    --input $DIR/lectures_source/$SRC_NAME \
    --output $DST_NAME \
    --type slides \
    --toc_type remove_headers 

# 2) Generate book chapter.
HELPERS_ROOT_DIR=$(find . -type d -path "./helpers_root/dev_scripts_helpers")
echo "HELPERS_ROOT_DIR=$HELPERS_ROOT_DIR"

# E.g., Lesson01.1-Intro.txt
INPUT_FILE="${files[0]}"

OUT_DIR="$DIR/book"
echo "OUT_DIR=$OUT_DIR"

$HELPERS_ROOT_DIR/slides/generate_book_chapter.py \
    --input_file "$INPUT_FILE" \
    --input_pdf_file $DST_NAME \
    --output_dir $OUT_DIR

# 3) Convert to PDF.
BASENAME=$(basename "$INPUT_FILE" .txt)
PDF_FILE_NAME="$OUT_DIR/${BASENAME}.book_chapter.pdf"
pandoc "$OUT_DIR/${BASENAME}.book_chapter.txt" \
    -o $PDF_FILE_NAME \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    --highlight-style=tango \
    --include-in-header=$HELPERS_ROOT_DIR/slides/header-style.tex

# 4) Open the PDF.
open -a /Applications/Skim.app $PDF_FILE_NAME
