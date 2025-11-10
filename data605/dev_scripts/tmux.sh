#!/bin/bash -e
#
# Create a standard tmux session for this repo.
#
# Create an amp tmux session for $HOME/src/umd_data605
# > dev_scripts/tmux.sh 1
#
# Kill the session
# > tmux kill-session -t umd_data605_1
#
# Create the link with:
# > ln -sf ~/src/umd_data605_1/dev_scripts/tmux.sh ~/go_umd_data605.sh; ls -l ~/go_umd_data605.sh
#

echo "##> umd_data605/dev_scripts/tmux.sh"

#set -x

SERVER_NAME=$(uname -n)
echo "SERVER_NAME=$SERVER_NAME"

# Try macOS setup.
DIR_NAME="/Users/$USER"
if [[ -d $DIR_NAME ]]; then
  echo "Inferred macOS setup"
  HOME_DIR=$DIR_NAME
else
  # Try AWS setup.
  DIR_NAME="/data/$USER"
  if [[ -d $DIR_NAME ]]; then
    echo "Inferred AWS setup"
    HOME_DIR=$DIR_NAME
  else
    if [[ $SERVER_NAME == "cf-spm-dev4" ]]; then
      HOME_DIR=$HOME
    elif [[ $SERVER_NAME == "cf-spm-dev8" ]]; then
      HOME_DIR=$HOME
    fi;
  fi;
fi;

if [[ -z $HOME_DIR ]]; then
    echo "ERROR: Can't infer where your home dir is located"
    exit -1
fi;
echo "HOME_DIR=$HOME_DIR"

# #############################################################################
# Parse command options.
# #############################################################################

IDX=$1
if [[ -z $IDX ]]; then
  echo "ERROR: You need to specify IDX={1,2,3}"
  exit -1
fi;

DIR_PREFIX="umd_data605"
GIT_ROOT="${HOME_DIR}/src/${DIR_PREFIX}_${IDX}"
echo "GIT_ROOT=$GIT_ROOT"

# #############################################################################
# Open the tmux session.
# #############################################################################

SETENV="dev_scripts/setenv.sh"
if [[ ! -f $GIT_ROOT/$SETENV ]]; then
    echo "ERROR: Can't find file $GIT_ROOT/$SETENV"
    exit -1
fi;

# No `clear` since we want to see issues, if any.
#CMD="source ${SETENV} && reset && clear"
CMD="source ${SETENV}"
TMUX_NAME="${DIR_PREFIX}_${IDX}"

tmux new-session -d -s $TMUX_NAME -n "---${TMUX_NAME}---"

# The first one window seems a problem.
tmux send-keys "white; cd ${GIT_ROOT} && $CMD" C-m C-m
#
tmux new-window -n "dbash"
tmux send-keys "green; cd ${GIT_ROOT} && $CMD" C-m C-m
#
tmux new-window -n "regr"
tmux send-keys "yellow; cd ${GIT_ROOT} && $CMD" C-m C-m
#
tmux new-window -n "jupyter"
tmux send-keys "yellow; cd ${GIT_ROOT} && $CMD" C-m C-m

# Go to the first tab.
tmux select-window -t $TMUX_NAME:0
tmux -2 attach-session -t $TMUX_NAME
