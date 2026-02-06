#!/bin/bash -e

# Find the actual git root directory.
export GIT_ROOT=$(git rev-parse --show-toplevel)
if [[ -z $GIT_ROOT ]]; then
    echo "Can't find GIT_ROOT=$GIT_ROOT"
    exit 1
fi;
echo "GIT_ROOT=$GIT_ROOT"

SCRIPT_SOURCE=$0
if [[ -z $1 ]]; then
    NUM_PASSES=1
else
    NUM_PASSES=$1
fi;
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)
echo $SCRIPT_DIR
echo "SCRIPT_DIR=$SCRIPT_DIR"

if [[ -d ./figs ]]; then
    rm -rf figs
fi;
EXEC=$GIT_ROOT/helpers_root/dev_scripts_helpers/documentation/render_images.py
for FILE_NAME in $SCRIPT_DIR/*.tex; do
  echo "Processing $FILE_NAME"
  "$EXEC" -i "$FILE_NAME"
done

#if [[ ./figs ]]; then
#    cp figs/* $SCRIPT_DIR/figs
#fi;

LATEX_NAME=$SCRIPT_DIR/paper.tex
echo "LATEX_NAME=$LATEX_NAME"

PDF_NAME=$(basename $SCRIPT_DIR).pdf
echo "PDF_NAME=$PDF_NAME"

PDF_FILE_NAME=$SCRIPT_DIR/$PDF_NAME

DOCKERIZED_LATEX=$GIT_ROOT/helpers_root/dev_scripts_helpers/documentation/dockerized_latex.py

# First pdflatex pass - generates .aux file
if [[ $NUM_PASSES -ge 1 ]]; then
    $DOCKERIZED_LATEX -i ${LATEX_NAME} -o $PDF_FILE_NAME
fi;

if [[ $NUM_PASSES -ge 2 ]]; then
    $DOCKERIZED_LATEX -i ${LATEX_NAME} -o $PDF_FILE_NAME
fi;

if [[ $NUM_PASSES -ge 3 ]]; then
    if [[ -f $SCRIPT_DIR/references.bib ]]; then
        # Run bibtex to process bibliography
        # Extract directory and base name for bibtex
        TEX_DIR=$(dirname $LATEX_NAME)
        TEX_BASE=$(basename $LATEX_NAME .tex)
        cp $SCRIPT_DIR/references.bib .
        bibtex $TEX_BASE

        # Third pdflatex pass - resolves all cross-references
        $DOCKERIZED_LATEX -i ${LATEX_NAME} -o $PDF_FILE_NAME
        $DOCKERIZED_LATEX -i ${LATEX_NAME} -o $PDF_FILE_NAME
    fi;
fi;

LOGFILE=paper.log
grep -E "LaTeX Warning:|Package .* Warning:|Class .* Warning:" "$LOGFILE" || true

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running on macOS"

    # Check if Skim.app exists (system-wide or in user's Applications)
    if [[ -d "/Applications/Skim.app" ]] || [[ -d "$HOME/Applications/Skim.app" ]]; then
        echo "Skim is installed."
        # do something that requires Skim here
        # From open_file_cmd.sh
        /usr/bin/osascript << EOF
set theFile to POSIX file "$PDF_FILE_NAME" as alias
tell application "Skim"
activate
set theDocs to get documents whose path is (get POSIX path of theFile)
if (count of theDocs) > 0 then revert theDocs
open theFile
end tell
EOF
    else
        echo "Skim is not installed."
    fi
else
    echo "Not running on macOS."
fi

cp $PDF_FILE_NAME website/docs/papers/gp_saggese_quant_research_cv.pdf
