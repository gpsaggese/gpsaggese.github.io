> (cd helpers_root; chmod -R +w docs; git reset --hard origin/master; chmod -R +w docs)
> git -c submodule.recurse=false checkout UmdTask27_Fall2025_HMMlearn_Anomaly_Detection_in_Network_Traffic

- Look for different files
> git diff --name-status master...

- Only files inside the class_project should be modified
```
> git restore --source origin/master --worktree --staged -- . ':(exclude)class_project/**'
```

git fetch origin
gh pr checkout $1
git merge -X theirs master

find . -name "tmp.*" | xargs rm -rf
find . -name ".DS_Store" | xargs rm -rf
find . -name "__pycache__" | xargs rm -rf

rsync -av --delete \
  --exclude='class_project/**' \
  --exclude='.git' \
  /Users/saggese/src/umd_classes1/ \
  /Users/saggese/src/umd_class_scripts/

git status

git add data605 msml610
git add README.slides.md
git add -u

# Check that the files from the students are in the right dir
# git diff --name-only origin/master...HEAD
#
git commit -m "Merge" && git push




find class_project/MSML610/Fall2025/Projects/UmdTask78_Fall2025_Ax_Multi_Objective_Optimization_for_Marketing_Campaigns/  -type f -exec du -h {} + | sort -hr | head -n 10

class_project/create_PR.sh

class_project/create_PR.py \
  --input_file class_project/fall2025_msml610_branches_dirs.txt \
  --source_dir /Users/saggese/src/umd_class_scripts \
  --dst_dir /Users/saggese/src/umd_classes3
  --copy_dirs
