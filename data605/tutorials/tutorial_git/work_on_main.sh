#!/bin/bash -xe

# To generate the output for the markdown run:
# > /Users/saggese/src/umd_data605/tutorials/tutorial_git/work_on_main.sh 2>&1 | tee /tmp/log.txt
# > cat /tmp/log.txt | perl -p -e 's/\+\+/+/; s/^\+ /\n> /'

GIT_ROOT=/Users/saggese/src/umd_data605
source $GIT_ROOT/tutorials/tutorial_git/restart.sh

git status -s

touch work_main.py

git status -s

git add work_main.py

git status -s

git log --graph --oneline -3

git commit -am "Add work_main.py"

git log --graph --oneline -3
