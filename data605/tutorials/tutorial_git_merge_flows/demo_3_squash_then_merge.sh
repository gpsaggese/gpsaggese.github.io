#!/bin/bash -xe
#
# Flow 3: squash `feature`'s commits first, then merge.
# Collapse the 5 small commits into 1 before merging `main`. This keeps the
# conflict count at 1 (like a plain merge) while also cleaning up the
# feature branch's history.

echo "=== phase: restore the identical starting point ==="
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/restart.sh"

echo "=== phase: squash feature's five commits into one ==="
git checkout feature

# Squash the 5 "Update" commits into 1, keeping the working tree unchanged.
# merge-base: find the commit where `feature` and `main` last shared history.
BASE=$(git merge-base feature main)
# --soft: move the branch pointer to BASE but leave the index and working
# tree untouched, so the 5 commits' combined changes stay staged.
git reset --soft "$BASE"
git commit -m "Update (squashed)"
# --oneline: one line per commit, for the "feature" branch.
git log --oneline feature

echo "=== phase: merge main into the squashed feature branch ==="
# Now merge `main`: same 1 conflict as flow 1, but `feature`'s own history
# is a single clean commit instead of 5 "Update" commits.
# -m: merge commit message given inline.
git merge main -m "Merge main into feature" || true
# -s: short status format, one line per changed file.
git status -s
cat shared.py

echo "=== phase: resolve the conflict and finish the merge ==="
cat >shared.py <<'EOF'
def process(rows):
    # Step 1: validate input
    # Step 2: normalize input rows (v5)
    # Step 3: dedupe
    # Step 4: write output
    return rows
EOF
git add shared.py
# -m: commit message given inline.
git commit -m "Merge main into feature"

echo "=== resulting history (linear feature branch, 1 conflict resolved) ==="
# --oneline: one line per commit
# --graph: draw the branch/merge structure as ASCII art next to the log.
git log --oneline --graph
