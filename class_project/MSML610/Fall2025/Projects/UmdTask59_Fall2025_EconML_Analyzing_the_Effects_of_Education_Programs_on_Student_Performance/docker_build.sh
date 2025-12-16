#!/usr/bin/env bash
set -euo pipefail

IMAGE="umdtask59_econml"

docker build -t "${IMAGE}" .
