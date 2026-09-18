#!/bin/bash -xe
# """
# Rewire all the docker projects together.
# """

DIRS=$(ls -1 -d msml610/tutorials/L*)
# E.g.,
# msml610/tutorials/L03_knowledge_representation
# msml610/tutorials/L05_statistical_learning
#
# TODO(ai_gp): Add also the dirs in tutorials/

for DIR in DIRS; do;
    echo "DIR=$DIR"
    create_links.py --src_dir class_project/project_template --dst_dir $DIR --replace_links
done;
