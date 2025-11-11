#!/bin/bash -xe
ROOT_DIR=/Users/saggese/src/cmamp2

DIR=sorrentum_sandbox
SRC_DIR=$ROOT_DIR/$DIR
#(cd $SRC_DIR; invoke git_clean)
rm -rf $DIR
rsync -arhv $SRC_DIR/ $DIR/
git add $DIR
