#!/bin/bash -xe
# To generate output for the tutorial run:
# ```
# > merge_conflict.sh 2>&1 | tee /tmp/log.txt
# > cat /tmp/log.txt | perl -p -e 's/\+\+/+/; s/^\+ /\n> /'
# ```

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$GIT_ROOT/tutorials/tutorial_git/restart.sh"

# Work on issue53.
git checkout -b iss53
echo "hello from iss53" >feature.py
git add feature.py
git status -s
git commit -am "Add feature.py"
git log --graph --oneline -3

# Fix in hot-fix.
git checkout main
git checkout -b hotfix
echo "hello from hotfix" > feature.py
git add feature.py
git status -s
git commit -am "Add feature.py"

# Merge hot-fix into main. 
git checkout main
git merge hotfix -m "Merge hotfix"
git log --graph --oneline -3

# Merge iss53 back to main. 
git checkout main
git merge iss53 -m "Merge iss53" || true

# Resolve.
git status -s
git diff

echo "hello from iss53 and hotfix" > feature.py
git add feature.py
git status -s

cat feature.py

git commit -m "Merge"
git status -s
git log --graph --oneline -5
