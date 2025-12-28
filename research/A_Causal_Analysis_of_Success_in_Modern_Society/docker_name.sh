#!/usr/bin/env bash
set -e

echo "============================================"
echo "Running Docker containers (names only)"
echo "============================================"

docker ps --format "Container: {{.Names}}"

echo "============================================"