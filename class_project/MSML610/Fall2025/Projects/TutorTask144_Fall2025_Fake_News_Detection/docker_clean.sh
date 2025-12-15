#!/bin/bash

set -e

echo "Cleaning Docker resources..."
docker-compose -f docker-compose.mcp.yml down
docker rmi bert-fake-news-mcp:latest 2>/dev/null || true
echo "Cleanup complete."
