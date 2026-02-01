#!/bin/bash
#
# Common utility functions for classes scripts.
#

# Validate DIR and LESSON arguments and find the matching lecture file.
#
# This function:
# - Validates that exactly 2 arguments are provided (DIR and LESSON)
# - Sets DIR and LESSON variables in the caller's scope
# - Finds the matching file in DIR/lectures_source/Lesson${LESSON}*
# - Validates that exactly one matching file exists
# - Sets the FILES variable with the found file path
#
# Usage:
#   validate_dir_lesson_args "$#" "$@"
#   # After calling, use $DIR, $LESSON, and $FILES
#
# Arguments:
#   $1 - Argument count from caller ($#)
#   $2 - DIR (course directory, e.g., data605, msml610)
#   $3 - LESSON (lesson number, e.g., 01.1, 02.3)
#
# Returns:
#   0 on success
#   1 on error (with error message printed to stderr)
#
validate_dir_lesson_args() {
    local argc=$1
    local script_name=$0

    # Check that exactly two arguments are provided
    if [ "$argc" -ne 2 ]; then
        echo "Error: Expected exactly 2 arguments, got $argc" >&2
        echo "" >&2
        echo "Usage: $script_name <DIR> <LESSON>" >&2
        echo "" >&2
        echo "Arguments:" >&2
        echo "  DIR     - Course directory (e.g., data605, msml610)" >&2
        echo "  LESSON  - Lesson number (e.g., 01.1, 02.3)" >&2
        echo "" >&2
        echo "Example:" >&2
        echo "  $script_name data605 0.1" >&2
        return 1
    fi

    # Set DIR and LESSON in caller's scope
    DIR=$2
    LESSON=$3

    # Find matching files using bash glob expansion
    FILES_DIR="$DIR/lectures_source/Lesson${LESSON}*"
    echo "FILES_DIR=$FILES_DIR"

    # Use bash array to collect matching files
    local files_array=($FILES_DIR)
    local file_count=${#files_array[@]}

    if [ "$file_count" -ne 1 ]; then
        echo "Error: Need exactly one file, found $file_count" >&2
        if [ "$file_count" -gt 0 ]; then
            echo "Found files: ${files_array[*]}" >&2
        fi
        return 1
    else
        echo "Found file: ${files_array[*]}"
        # Set FILES to the first (and only) matched file
        FILES="${files_array[0]}"
    fi

    return 0
}
