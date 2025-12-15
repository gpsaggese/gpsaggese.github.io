#!/bin/bash

set -e

echo "Restarting server..."
docker-compose -f docker-compose.mcp.yml down
sleep 1
docker-compose -f docker-compose.mcp.yml up -d
sleep 2
echo "Server restarted at http://localhost:9090/"
