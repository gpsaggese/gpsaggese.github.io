#!/bin/bash
# Start Jupyter Notebook from the container
docker exec -it anomaly_container jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root