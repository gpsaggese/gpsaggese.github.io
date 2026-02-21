#!/bin/bash
# Docker image naming configuration.
# This file is sourced by docker_*.sh scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"

IMAGE_NAME="${IMAGE_NAME:-langchain_langgraph}"
REPO_NAME="${REPO_NAME:-${USER:-local}}"
FULL_IMAGE_NAME="${FULL_IMAGE_NAME:-$REPO_NAME/$IMAGE_NAME}"
CONTAINER_NAME="${CONTAINER_NAME:-$IMAGE_NAME}"
