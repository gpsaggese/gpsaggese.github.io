#!/bin/bash -e

# Check that the right number of arguments are provided.
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

PROMPT="
You are a college professor teaching a class.

Given the content below:
- Write 20 multiple choice questions
- Each question has 5 possible answers with only one correct answer
- Make sure to focus on concept and understanding of the material rather than memorization.
- Mark the correct answer in bold.

The output should be in Markdown code without having page separators, any
comment, or divved fence, just the questions and the answers.
"

# Save the prompt to a file
PROMPT_FILE="tmp.gen_quizzes_prompt.txt"
echo "$PROMPT" > $PROMPT_FILE

# Enable command printing from this point
set -x

SRC_NAME=$(cd $DIR/lectures_source; ls Lesson${LESSON}*)
DST_NAME=$(echo $SRC_NAME | sed 's/\.txt$/.quizzes.md/')

llm_cli.py --input $DIR/lectures_source/$SRC_NAME --output $DIR/lectures_quizzes/$DST_NAME --system_prompt_file $PROMPT_FILE
