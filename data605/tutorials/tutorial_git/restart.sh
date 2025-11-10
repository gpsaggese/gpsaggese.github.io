#!/bin/bash -xe
cd /tmp
# Delete dir, if needed.
if [[ -d /tmp/umd_data605_tmp ]]; then
    rm -rf /tmp/umd_data605_tmp
fi;
# Checkout the repo.
git clone git@github.com:gpsaggese/umd_data605.git /tmp/umd_data605_tmp
cd /tmp/umd_data605_tmp

# Remove remote to avoid to push to the actual repo.
git remote rm origin

# Print current state.
ls 
git status
git log --graph --oneline -3
