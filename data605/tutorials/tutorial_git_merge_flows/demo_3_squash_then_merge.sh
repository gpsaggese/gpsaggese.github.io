#!/bin/bash -xe
# Flow 3: squash `feature`'s commits first, then merge.
# Collapse the 5 small commits into 1 before merging `main`. This keeps the
# conflict count at 1 (like a plain merge) while also cleaning up the
# feature branch's history.
#
# To generate output for the tutorial run:
# ```
# > ./demo_3_squash_then_merge.sh 2>&1 | tee /tmp/log.txt
# ```

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/restart.sh"

git checkout -q feature

# Squash the 5 "Update" commits into 1, keeping the working tree unchanged.
BASE=$(git merge-base feature main)
git reset -q --soft "$BASE"
git commit -q -m "Update (squashed)"
git log --oneline feature

# Now merge `main`: same 1 conflict as flow 1, but `feature`'s own history
# is a single clean commit instead of 5 "Update" commits.
git merge main -m "Merge main into feature" || true
git status -s
cat shared.py

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

echo "=== resulting history (linear feature branch, 1 conflict resolved) ==="
git log --oneline --graph
