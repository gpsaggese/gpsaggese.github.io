#!/bin/bash
# Run Jupyter Lab inside container
docker run -p 8888:8888 -v $(pwd):/app onnx_fake_news_detection jupyter lab --ip=0.0.0.0 --allow-root
