#!/bin/bash -xe
#
# Build a small local scratch repo with a diverging history:
# - `main` advances by one commit that edits a shared line.
# - `feature` branches off *before* that commit, then edits the *same* line
#   five times, in five small commits (mimicking a branch with many tiny
#   "Update" commits).
# ```

echo "=== phase: reset the scratch repo directory ==="
# Work from a fixed, predictable location on disk.
cd /tmp
# -d: true if the path exists and is a directory.
if [[ -d /tmp/merge_flows_demo ]]; then
    # -r: remove directories and their contents recursively.
    # -f: never prompt, ignore a missing path.
    rm -rf /tmp/merge_flows_demo
fi
mkdir /tmp/merge_flows_demo
cd /tmp/merge_flows_demo

echo "=== phase: initialize the git repository and identity ==="
git init
# Local (repo only) identity, so the commits do not depend on global config.
git config user.email "demo@example.com"
git config user.name "Demo"

echo "=== phase: create the initial commit and branches ==="
# Create first version of the file.
# <<'EOF' (quoted delimiter): the heredoc body is written verbatim, with no
# variable or command substitution.
cat >shared.py <<'EOF'
def process(rows):
    # Step 1: validate input
    # Step 2: normalize
    # Step 3: dedupe
    # Step 4: write output
    return rows
EOF
git add shared.py
# -m: commit message given inline.
git commit -m "Update #1"
# -M: rename the current branch to "main", overwriting it if it exists.
git branch -M main
# -b: create branch "feature" and switch to it.
git checkout -b feature

echo "=== phase: advance main with one commit ==="
# `main` moves forward with one commit that rewrites the shared line.
git checkout main
sed -i.bak 's/# Step 2: normalize/# Step 2: normalize input rows/' shared.py
rm -f shared.py.bak
# -a: stage all tracked, modified files. -m: message inline.
git commit -am "Update #2"
git checkout feature

echo "=== phase: advance feature with five commits on the same line ==="
# `feature` moves forward with 5 small commits, each rewriting the *same*
# line (e.g., a developer polishing the same comment across many commits).
for i in 1 2 3 4 5; do
    sed -i.bak "s/# Step 2:.*/# Step 2: normalize (v$i)/" shared.py
    rm -f shared.py.bak
    # -a: stage all tracked, modified files
    # -m: message inline, numbered sequentially from the 2 commits already on
    #     `main`
    git commit -am "Update #$((i + 2))"
done

echo "=== phase: show the two starting histories ==="
echo "=== main log ==="
git log --oneline main
echo "=== feature log ==="
git log --oneline feature
