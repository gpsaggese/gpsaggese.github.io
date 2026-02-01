#!/usr/bin/env bash
set -euo pipefail

BRANCH="$1"
BASE_BRANCH="origin/master"
#DIR=class_project
DIR=class_project/MSML610/Fall2025/Projects/UmdTask78_Fall2025_Ax_Multi_Objective_Optimization_for_Marketing_Campaigns

echo "▶ Target branch: $BRANCH"
echo "▶ Base branch:   $BASE_BRANCH"
echo "▶ Keep dir:      $DIR/"

# Ensure we are inside a git repo
git rev-parse --is-inside-work-tree >/dev/null

# 1. Handle in-progress merge
if git rev-parse -q --verify MERGE_HEAD >/dev/null; then
  echo "⚠ Merge in progress detected"

  echo "→ Aborting merge"
  git merge --abort || {
    echo "❌ Failed to abort merge"
    exit 1
  }
fi

# 2. Switch to branch
git switch "$BRANCH"

# 3. Ensure clean working tree
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "⚠ Working tree not clean; stashing"
  git stash push -m "pre-cleanup stash"
  STASHED=1
else
  STASHED=0
fi

# 4. Save DIR from branch
git stash push -m "keep $DIR" -- "$DIR/"

# 5. Reset everything to master
git fetch origin
git restore --source="$BASE_BRANCH" --staged --worktree -- .

# 6. Restore DIR
git stash pop

# 7. Restore pre-cleanup stash if needed
if [[ "$STASHED" -eq 1 ]]; then
  echo "→ Restoring previous stash"
  git stash pop || true
fi

# 8. Commit
git add -A
git commit -m "Keep only $DIR changes; everything else matches $BASE_BRANCH"

echo "✅ Done"
