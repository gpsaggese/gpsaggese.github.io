#!/bin/bash

echo "Building Docker container for causal success analysis..."
docker build -t causal_success_analysis .
echo "Build complete!"
