#!/bin/bash -xe

# To generate output for the tutorial run:
# ```
# > bash work_on_main.sh 2>&1 | tee /tmp/log.txt
# > cat /tmp/log.txt | perl -p -e 's/\+\+/+/; s/^\+ /\n> /'
# ```

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$GIT_ROOT/tutorials/tutorial_git/restart.sh"

git status -s

touch work_main.py

git status -s

git add work_main.py

git status -s

git log --graph --oneline -3

git commit -am "Add work_main.py"

git log --graph --oneline -3
