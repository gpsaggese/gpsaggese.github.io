#!/bin/bash -e

# Find the actual git root directory.
export GIT_ROOT=$(git rev-parse --show-toplevel)
if [[ -z $GIT_ROOT ]]; then
    echo "Can't find GIT_ROOT=$GIT_ROOT"
    exit 1
fi;
echo "GIT_ROOT=$GIT_ROOT"

PDF_FILE_NAME=gp_resume/gp_saggese_cv.pdf

dockerized_latex.py -i gp_resume/gp_saggese_cv.tex -o $PDF_FILE_NAME

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
