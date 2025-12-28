#!/usr/bin/env bash
set -e

echo "============================================"
echo "Cleaning Docker resources"
echo "============================================"

# Remove stopped containers, unused networks, dangling images
docker system prune -f

# Remove unused volumes
docker volume prune -f

echo "============================================"
echo "Docker cleanup complete"
echo "============================================"
