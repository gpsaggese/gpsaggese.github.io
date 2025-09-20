#!/bin/bash -xe
FILE=$1

# Remove 3/57
perl -ne 'print unless /^[[:space:]]*\d+\/\d+[[:space:]]*$/;' $FILE

# Remove the tag
perl -i -ne 'print unless /UMD DATA605/;' $FILE

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

# Reflow
lint_txt.py -i $FILE --use_dockerized_prettier
