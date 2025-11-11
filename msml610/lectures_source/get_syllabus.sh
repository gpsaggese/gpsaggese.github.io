#!/bin/bash -e

# lectures_source/example.txt
# lectures_source/Gallery.txt

# > ls -1 lectures_source/Lesson*
FILES=$(ls -1 lectures_source/Lesson*)

for FILE in $FILES; do
    echo
    echo "# #############################################################################"
    echo "# $FILE"
    echo "# #############################################################################"
    cmd="extract_headers_from_markdown.py -i $FILE"
    eval $cmd
done
