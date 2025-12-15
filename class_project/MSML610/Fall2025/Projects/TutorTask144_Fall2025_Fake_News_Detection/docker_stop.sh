#!/bin/bash

set -e

echo "Stopping server..."
docker-compose -f docker-compose.mcp.yml down
echo "Server stopped."
