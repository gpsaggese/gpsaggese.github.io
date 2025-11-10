#!/bin/bash -xe
# /Users/saggese/src/umd_data605/projects/tutorial_git/hot_fix.sh 2>&1 | tee /tmp/log.txt
# cat /tmp/log.txt | perl -p -e 's/\+\+/+/; s/^\+ /\n> /'

GIT_ROOT=/Users/saggese/src/umd_data605
source $GIT_ROOT/tutorials/tutorial_git/restart.sh

# Work on issue53.
git checkout -b iss53
touch feature.py
git add feature.py
git status -s
git commit -am "Add feature.py"
git log --graph --oneline -3

# Hot-fix to main.
git checkout main
git checkout -b hotfix
touch hot_fix.py
git add hot_fix.py
git status -s
git commit -am "Add hot_fix.py"
git checkout main
git merge hotfix -m "Merge hot_fix.py"
git log --graph --oneline -3

# Divergent history.
git checkout iss53
git log --graph --oneline -3

# Branch keeps diverging.
touch feature2.py
git add feature2.py
git commit -am "Add feature2.py"

# Merge iss53 back to main. 
git checkout main
git merge iss53 -m "Merge iss53"
