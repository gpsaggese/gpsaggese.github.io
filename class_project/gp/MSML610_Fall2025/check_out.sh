#!/bin/bash -xe
(cd helpers_root; chmod -R +w docs; git reset --hard origin/master; chmod -R +w docs)

git fetch origin
#git switch $1 || git switch -c $1  --track origin/$1
git -c submodule.recurse=false checkout $1

git diff --name-status master...
