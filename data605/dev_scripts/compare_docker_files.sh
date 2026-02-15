#!/bin/bash -xe

if [[ -z $GIT_REPO ]]; then
    echo "GIT_REPO is not defined: exiting"
    exit -1
fi;
echo "GIT_REPO=$GIT_REPO"

SRC_DIR=$GIT_REPO/tutorials/tutorial_postgres
echo "SRC_DIR=$SRC_DIR"
DST_DIR=$GIT_REPO/tutorials/tutorial_pandas
echo "DST_DIR=$DST_DIR"

FILES=$(cd $SRC_DIR; ls -1 docker_*.sh)
echo $FILES

for FILE in $FILES
do
    echo "FILE=$FILE"
    vimdiff $SRC_DIR/$FILE $DST_DIR/$FILE
done;

if [[ 0 == 1 ]]; then
    ln -sf ../../project_template/bashrc $DST_DIR
    ln -sf ../../project_template/etc_sudoers $DST_DIR
    ln -sf ../../project_template/install_jupyter_extensions $DST_DIR
    ln -sf ../../project_template/version.sh $DST_DIR
fi;
