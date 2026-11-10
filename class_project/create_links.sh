#!/bin/bash -xe
# """
# Rewire all the docker projects together.
#
# Extra args (e.g., `--dry_run`) are passed to `create_links.py`.
# """

# The paths below are relative to the root of the repo.
cd $(git rev-parse --show-toplevel)

SRC_DIR="class_project/project_template"

DIRS=$(ls -1 -d msml610/tutorials/L*)
# E.g.,
# msml610/tutorials/L03_knowledge_representation
# msml610/tutorials/L05_statistical_learning

# Add the docker projects in `tutorials/`, skipping symlinks (e.g.,
# `tutorials/project_template`) and dirs that are not docker projects.
for DIR in $(ls -1 -d tutorials/*/); do
    DIR=${DIR%/}
    if [[ ! -L $DIR && -e $DIR/docker_build.sh ]]; then
        DIRS="$DIRS $DIR"
    fi
done;

# Replace the files in each dir with links to the ones in the template.
for DIR in $DIRS; do
    echo "DIR=$DIR"
    create_links.py --src_dir $SRC_DIR --dst_dir $DIR --replace_links "$@"
done;
