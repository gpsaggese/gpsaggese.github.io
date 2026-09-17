#!/bin/bash -xe
#
# Flow 1: plain merge.
# `feature` has 5 commits that all touch the same line as `main`'s 1 commit.
# A merge resolves that overlap exactly once, in a single merge commit.

echo "=== phase: restore the identical starting point ==="
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/restart.sh"

echo "=== phase: attempt the merge and inspect the conflict ==="
git checkout feature

# This is what `invoke git_merge_master` runs under the hood: `git merge master`.
# -m: merge commit message given inline.
# || true: the merge stops on a conflict (non-zero exit), which would abort
# the script under `set -e`; `|| true` lets the script continue so the
# conflict can be resolved by hand below.
git merge main -m "Merge main into feature" || true

# Both sides touched the same line: exactly 1 conflict, in 1 file.
# -s: short status format, one line per changed file.
git status -s
cat shared.py

echo "=== phase: resolve the conflict and finish the merge ==="
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
git commit -m "Merge main into feature"

echo "=== resulting history (non-linear, 1 conflict resolved) ==="
# --oneline: one line per commit
# --graph: draw the branch/merge structure as ASCII art next to the log
git log --oneline --graph
