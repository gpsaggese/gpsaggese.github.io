#!/bin/bash

set -e

echo "Starting server..."
docker-compose -f docker-compose.mcp.yml up -d
sleep 2

echo "Server is running at http://localhost:9090/"
echo "View logs with: ./docker_logs.sh"
echo "Stop with: ./docker_stop.sh"
