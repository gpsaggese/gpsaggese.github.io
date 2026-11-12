#!/bin/bash -xe
#
# Flow 2: rebase.
# `feature`'s 5 commits are replayed one at a time onto `main`. Every one of
# them touches the same line `main` also changed, so the rebase stops on
# *every single commit*: 5 conflicts instead of 1.

echo "=== phase: restore the identical starting point ==="
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/restart.sh"

echo "=== phase: start the rebase (triggers the first conflict) ==="
git checkout feature

# This is NOT what `invoke git_merge_master` runs: it never rebases.
git rebase main || true

echo "=== phase: resolve all five conflicts, one per replayed commit ==="
# Resolve the same conflict 5 times, once per replayed commit.
for i in 1 2 3 4 5; do
    echo "=== conflict $i/5 ==="
    # -s: short status format, one line per changed file.
    git status -s
    cat shared.py

    # <<EOF (unquoted delimiter): unlike the other demos, this heredoc is
    # NOT quoted, so `$i` expands to the current loop value.
    cat >shared.py <<EOF
def process(rows):
    # Step 1: validate input
    # Step 2: normalize input rows (v$i)
    # Step 3: dedupe
    # Step 4: write output
    return rows
EOF
    git add shared.py
    # GIT_EDITOR=true: skip the interactive commit-message editor that
    # `rebase --continue` would otherwise open, keeping the replayed
    # commit's original message.
    # --continue: resume the rebase after the conflict is resolved.
    GIT_EDITOR=true git rebase --continue || true
done

echo "=== resulting history (linear, 5 conflicts resolved) ==="
# --oneline: one line per commit
# --graph: draw the branch structure as ASCII art next to the log.
git log --oneline --graph
