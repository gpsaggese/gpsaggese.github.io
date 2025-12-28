#!/usr/bin/env bash
set -ex

# Use classic builder so logs are very explicit
export DOCKER_BUILDKIT=0

docker build -t umdtask88-dgl-fraud .
