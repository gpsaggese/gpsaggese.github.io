#!/bin/bash -xe
# Flow 1: plain merge.
# `feature` has 5 commits that all touch the same line as `main`'s 1 commit.
# A merge resolves that overlap exactly once, in a single merge commit.
#
# To generate output for the tutorial run:
# ```
# > ./demo_1_merge.sh 2>&1 | tee /tmp/log.txt
# ```

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/restart.sh"

git checkout -q feature

# This is what `invoke git_merge_master` runs under the hood: `git merge master`.
git merge main -m "Merge main into feature" || true

# Both sides touched the same line: exactly 1 conflict, in 1 file.
git status -s
cat shared.py

# Resolve by hand, keeping both edits.
cat >shared.py <<'EOF'
def process(rows):
    # Step 1: validate input
    # Step 2: normalize input rows (v5)
    # Step 3: dedupe
    # Step 4: write output
    return rows
EOF
git add shared.py
git commit -q -m "Merge main into feature"

echo "=== resulting history (non-linear, 1 conflict resolved) ==="
git log --oneline --graph
