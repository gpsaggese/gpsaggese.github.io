# Remove 3/57
perl -ne 'print unless /^[[:space:]]*\d+\/\d+[[:space:]]*$/;' $*

# Remove the tag
perl -i -ne 'print unless /UMD DATA605/;' $*

# Merge
# ```
# - **Challenges
# **
# ```
# into
# - **Challenges**
perl -i -0777 -pe 's/\*\*\s*\n\s*\*\*/\*\*/g' $*
perl -i -0777 -pe 's/\*\*([^\n]+?)\s*\n\s*\*\*/**$1**/g' $*

# Clean up 
perl -i -0777 -pe 's/\n{3,}/\n\n/g' $*
