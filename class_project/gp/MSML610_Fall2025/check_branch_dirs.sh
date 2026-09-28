#!/bin/bash -xe
git fetch origin
git diff --name-only master...origin/$1 \
  | xargs -n1 dirname \
  | sort -u
