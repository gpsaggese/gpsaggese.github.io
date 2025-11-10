#!/bin/bash -xe
FILE=$1

# Remove 3/57
perl -ne 'print unless /^[[:space:]]*\d+\/\d+[[:space:]]*$/;' $FILE

# Remove the tag
perl -i -ne 'print unless /UMD DATA605/;' $FILE

# Remove duplicated lines
perl -i -ne 'print if !defined($prev) || $_ ne $prev; $prev = $_' $FILE

# Merge
# ```
# - **Challenges
# **
# ```
# into
# - **Challenges**
perl -i -0777 -pe 's/\*\*\s*\n\s*\*\*/\*\*/g' $FILE
perl -i -0777 -pe 's/\*\*([^\n]+?)\s*\n\s*\*\*/**$1**/g' $FILE

# Clean up 
perl -i -0777 -pe 's/\n{3,}/\n\n/g' $FILE

perl -i -pe 's/→/\$\\to\$/g' $FILE
perl -i -pe "s/’/'/g" $FILE

# Convert '**git pull**: short hand from **git fetch origin**'
# to
# `git pull`: short hand from `git fetch origin`
perl -i -pe 's/\*\*git ([^*]+)\*\*/`git \1`/g' $FILE
perl -i -pe 's/\*\*> git ([^*]+)\*\*/`> git \1`/g' $FILE

# Remove trailing \*\*
perl -pe 's/\*\*$//' $FILE

# Reflow
lint_txt.py -i $FILE --use_dockerized_prettier
