#!/bin/bash -xe
# Build a small local scratch repo with a diverging history:
# - `main` advances by one commit that edits a shared line.
# - `feature` branches off *before* that commit, then edits the *same* line
#   five times, in five small commits (mimicking a branch with many tiny
#   "Update" commits).
#
# Source this before each demo script to get an identical starting point:
# ```
# > source restart.sh
# > ./demo_1_merge.sh
# ```

cd /tmp
if [[ -d /tmp/merge_flows_demo ]]; then
    rm -rf /tmp/merge_flows_demo
fi
mkdir /tmp/merge_flows_demo
cd /tmp/merge_flows_demo

git init -q
git config user.email "demo@example.com"
git config user.name "Demo"

cat >shared.py <<'EOF'
def process(rows):
    # Step 1: validate input
    # Step 2: normalize
    # Step 3: dedupe
    # Step 4: write output
    return rows
EOF
git add shared.py
git commit -q -m "Initial commit"
git branch -M main
git checkout -q -b feature

# `main` moves forward with one commit that rewrites the shared line.
git checkout -q main
sed -i.bak 's/# Step 2: normalize/# Step 2: normalize input rows/' shared.py
rm -f shared.py.bak
git commit -q -am "Update"
git checkout -q feature

# `feature` moves forward with 5 small commits, each rewriting the *same*
# line (e.g., a developer polishing the same comment across many commits).
for i in 1 2 3 4 5; do
    sed -i.bak "s/# Step 2:.*/# Step 2: normalize (v$i)/" shared.py
    rm -f shared.py.bak
    git commit -q -am "Update"
done

echo "=== main log ==="
git log --oneline main
echo "=== feature log ==="
git log --oneline feature
