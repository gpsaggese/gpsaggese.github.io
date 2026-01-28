#!/bin/bash -e

# Check that exactly two arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Error: Expected exactly 1 argument, got $#"
    echo ""
    echo "Usage: $0 <DIR>"
    echo ""
    echo "Arguments:"
    echo "  DIR     - Course directory (e.g., data605, msml610)"
    echo ""
    echo "Example:"
    echo "  $0 data605"
    exit 1
fi

# E.g., data605, msml610
DIR=$1/book
echo "DIR=$DIR"

cd $DIR
find Lesson*.pdf -type f -name "*.pdf" -print0 | while IFS= read -r -d '' file; do     pages=$(mdls -name kMDItemNumberOfPages "$file" | awk -F'= ' '{print $2}');     echo -e "${file}\t${pages}"; done
