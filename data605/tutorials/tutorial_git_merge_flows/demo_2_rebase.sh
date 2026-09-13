#!/bin/bash -xe
# Flow 2: rebase.
# `feature`'s 5 commits are replayed one at a time onto `main`. Every one of
# them touches the same line `main` also changed, so the rebase stops on
# *every single commit*: 5 conflicts instead of 1.
#
# To generate output for the tutorial run:
# ```
# > ./demo_2_rebase.sh 2>&1 | tee /tmp/log.txt
# ```

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/restart.sh"

git checkout -q feature

# This is NOT what `invoke git_merge_master` runs: it never rebases.
git rebase main || true

# Resolve the same conflict 5 times, once per replayed commit.
for i in 1 2 3 4 5; do
    echo "=== conflict $i/5 ==="
    git status -s
    cat shared.py

    cat >shared.py <<EOF
def process(rows):
    # Step 1: validate input
    # Step 2: normalize input rows (v$i)
    # Step 3: dedupe
    # Step 4: write output
    return rows
EOF
    git add shared.py
    GIT_EDITOR=true git rebase --continue || true
done

echo "=== resulting history (linear, 5 conflicts resolved) ==="
git log --oneline --graph
